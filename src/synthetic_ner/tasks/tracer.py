"""Langfuse-backed tracing helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langfuse import get_client
from src.synthetic_ner.types.app_config import LangfuseConfig


@dataclass(slots=True)
class TraceHandle:
    trace_id: str | None
    observation: Any
    doc_id: str
    task_id: str
    stage: str
    parent_task_id: str | None


@dataclass(slots=True)
class DocumentTraceSession:
    trace_id: str | None
    trace_url: str | None
    path: Path
    enabled: bool


class TraceStore:
    def __init__(self, base_dir: Path, cfg: LangfuseConfig) -> None:
        self.base_dir = base_dir
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.base_url = cfg.base_url
        self.public_key_env = cfg.public_key_env
        self.secret_key_env = cfg.secret_key_env

        public_key = os.getenv(self.public_key_env)
        secret_key = os.getenv(self.secret_key_env)
        if public_key and "LANGFUSE_PUBLIC_KEY" not in os.environ:
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        if secret_key and "LANGFUSE_SECRET_KEY" not in os.environ:
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        if self.base_url and "LANGFUSE_BASE_URL" not in os.environ:
            os.environ["LANGFUSE_BASE_URL"] = self.base_url

        self.client_enabled = self.enabled and self._has_credentials()
        self.client = get_client() if self.client_enabled else None
        self._document_context = None
        self._document_observation = None
        self._current_session: DocumentTraceSession | None = None

    def start_document_run(
        self,
        *,
        doc_id: str,
        name: str,
        input_payload: dict,
        metadata: dict,
    ) -> DocumentTraceSession:
        doc_dir = self.base_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        index_path = doc_dir / "TRACE_INDEX.md"

        trace_id = None
        trace_url = None
        if self.client_enabled:
            self._document_context = self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_payload,
                metadata=metadata,
            )
            self._document_observation = self._document_context.__enter__()
            trace_id = self.client.get_current_trace_id()
            if trace_id:
                trace_url = self.client.get_trace_url(trace_id=trace_id)

        self._current_session = DocumentTraceSession(
            trace_id=trace_id,
            trace_url=trace_url,
            path=index_path,
            enabled=self.client_enabled,
        )
        self._write_index()
        return self._current_session

    def end_document_run(self, *, output_payload: dict | None = None) -> None:
        if self._document_observation is not None and output_payload is not None:
            self._document_observation.update(output=output_payload)

        if self._document_context is not None:
            self._document_context.__exit__(None, None, None)

        if self.client_enabled:
            self.client.flush()

        self._write_index()
        self._document_context = None
        self._document_observation = None

    def start_trace(
        self,
        *,
        doc_id: str,
        task_id: str,
        stage: str,
        model: str,
        parent_task_id: str | None = None,
        prompt: str | None = None,
        metadata: dict | None = None,
    ) -> TraceHandle:
        observation = None
        trace_id = self._current_session.trace_id if self._current_session else None
        if self.client_enabled:
            observation = self.client.start_observation(
                name=task_id,
                as_type="generation",
                model=model,
                input=prompt,
                metadata={
                    "doc_id": doc_id,
                    "task_id": task_id,
                    "stage": stage,
                    "parent_task_id": parent_task_id,
                    **(metadata or {}),
                },
            )
        return TraceHandle(
            trace_id=trace_id,
            observation=observation,
            doc_id=doc_id,
            task_id=task_id,
            stage=stage,
            parent_task_id=parent_task_id,
        )

    def record_llm_call(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        response: str,
        metadata: dict,
    ) -> None:
        if handle.observation is not None:
            handle.observation.update(
                input=prompt,
                output=response,
                metadata=metadata,
            )
            handle.observation.end()

    def record_error(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        error_message: str,
        metadata: dict,
    ) -> None:
        if handle.observation is not None:
            handle.observation.update(
                input=prompt,
                output=f"[error] {error_message}",
                metadata=metadata,
                level="ERROR",
                status_message=error_message,
            )
            handle.observation.end()

    def get_trace_info(self, doc_id: str) -> DocumentTraceSession:
        doc_dir = self.base_dir / doc_id
        index_path = doc_dir / "TRACE_INDEX.md"
        if self._current_session and self._current_session.path == index_path:
            return self._current_session
        return DocumentTraceSession(
            trace_id=None,
            trace_url=None,
            path=index_path,
            enabled=self.client_enabled,
        )

    def write_trace_index(self, doc_id: str) -> Path:
        self._write_index(doc_id)
        return self.base_dir / doc_id / "TRACE_INDEX.md"

    def _write_index(self, doc_id: str | None = None) -> None:
        session = self._current_session
        if doc_id is not None and (session is None or session.path.parent.name != doc_id):
            session = DocumentTraceSession(
                trace_id=None,
                trace_url=None,
                path=self.base_dir / doc_id / "TRACE_INDEX.md",
                enabled=self.client_enabled,
            )
        if session is None:
            return

        session.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": session.enabled,
            "trace_id": session.trace_id,
            "trace_url": session.trace_url,
            "backend": "langfuse",
        }
        lines = [
            "# Trace Index",
            "",
            "- Backend: `langfuse`",
            f"- Enabled: `{str(session.enabled).lower()}`",
            f"- Trace ID: `{session.trace_id or 'n/a'}`",
            f"- Trace URL: {session.trace_url or 'n/a'}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
        session.path.write_text("\n".join(lines), encoding="utf-8")

    def _has_credentials(self) -> bool:
        return bool(
            os.getenv(self.public_key_env)
            and os.getenv(self.secret_key_env)
        )
