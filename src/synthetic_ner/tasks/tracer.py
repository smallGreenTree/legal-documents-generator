"""Langfuse-backed tracing helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse
from src.synthetic_ner.types.app_config import LangfuseConfig


@dataclass(slots=True)
class TraceHandle:
    observation: Any


@dataclass(slots=True)
class DocumentTraceSession:
    enabled: bool
    trace_id: str | None
    trace_url: str | None


class TraceStore:
    def __init__(self, cfg: LangfuseConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled
        self._document_context = None
        self._document_observation = None
        self._current_session = DocumentTraceSession(
            enabled=self.enabled,
            trace_id=None,
            trace_url=None,
        )

        if not self.enabled:
            self.client = None
            return

        public_key = os.getenv(cfg.public_key_env)
        secret_key = os.getenv(cfg.secret_key_env)
        if not public_key or not secret_key:
            raise ValueError(
                "Langfuse is enabled but credentials are missing. "
                f"Expected env vars '{cfg.public_key_env}' and '{cfg.secret_key_env}'."
            )

        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=cfg.host,
        )

    def start_document_run(
        self,
        *,
        doc_id: str,
        name: str,
        input_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> DocumentTraceSession:
        del doc_id
        if not self.enabled or self.client is None:
            return self._current_session

        self._document_context = self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload,
            metadata=metadata,
        )
        self._document_observation = self._document_context.__enter__()
        trace_id = self.client.get_current_trace_id()
        trace_url = self.client.get_trace_url(trace_id=trace_id) if trace_id else None
        self._current_session = DocumentTraceSession(
            enabled=True,
            trace_id=trace_id,
            trace_url=trace_url,
        )
        return self._current_session

    def end_document_run(self, *, output_payload: dict[str, Any] | None = None) -> None:
        if self._document_observation is not None and output_payload is not None:
            self._document_observation.update(output=output_payload)

        if self._document_context is not None:
            self._document_context.__exit__(None, None, None)

        if self.enabled and self.client is not None:
            self.client.flush()

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
        metadata: dict[str, Any] | None = None,
    ) -> TraceHandle:
        del doc_id
        if not self.enabled or self.client is None:
            return TraceHandle(observation=None)

        observation = self.client.start_observation(
            name=task_id,
            as_type="generation",
            model=model,
            input=prompt,
            metadata={
                "stage": stage,
                "parent_task_id": parent_task_id,
                **(metadata or {}),
            },
            model_parameters=(metadata or {}).get("model_parameters"),
        )
        return TraceHandle(observation=observation)

    def record_llm_call(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        response: str,
        metadata: dict[str, Any],
    ) -> None:
        if handle.observation is None:
            return
        handle.observation.update(
            input=prompt,
            output=response,
            metadata=metadata,
            usage_details=_build_usage_details(metadata),
        )
        handle.observation.end()

    def record_error(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        error_message: str,
        metadata: dict[str, Any],
    ) -> None:
        if handle.observation is None:
            return
        handle.observation.update(
            input=prompt,
            output=f"[error] {error_message}",
            metadata=metadata,
            level="ERROR",
            status_message=error_message,
        )
        handle.observation.end()

    def get_trace_info(self) -> DocumentTraceSession:
        return self._current_session


def _build_usage_details(metadata: dict[str, Any]) -> dict[str, int] | None:
    prompt_tokens = metadata.get("tokens_prompt")
    response_tokens = metadata.get("tokens_response")
    usage_details: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        usage_details["input"] = prompt_tokens
    if isinstance(response_tokens, int):
        usage_details["output"] = response_tokens
    if isinstance(prompt_tokens, int) and isinstance(response_tokens, int):
        usage_details["total"] = prompt_tokens + response_tokens
    return usage_details or None
