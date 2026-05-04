"""Traced Ollama client."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import requests
from src.synthetic_ner.tasks.tracer import TraceStore
from src.synthetic_ner.types.app_config import OllamaConfig


@dataclass(slots=True)
class OllamaCallResult:
    text: str
    metadata: dict


class TracedOllamaClient:
    """Small wrapper around Ollama's generate API with Langfuse tracing."""

    def __init__(self, config: OllamaConfig, tracer: TraceStore) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.timeout = config.timeout
        self.tracer = tracer

    def invoke(
        self,
        *,
        doc_id: str,
        task_id: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        parent_task_id: str | None = None,
        temperature: float,
        max_output_tokens: int | None = None,
        prompt_object: Any | None = None,
    ) -> OllamaCallResult:
        full_prompt = (
            f"[SYSTEM]\n{system_prompt.strip()}\n\n"
            f"[USER]\n{user_prompt.strip()}\n"
        )
        prompt_payload = {
            "system_prompt": system_prompt.strip(),
            "user_prompt": user_prompt.strip(),
        }
        trace = self.tracer.start_trace(
            doc_id=doc_id,
            task_id=task_id,
            stage=stage,
            model=self.model,
            parent_task_id=parent_task_id,
            prompt=full_prompt,
            prompt_payload=prompt_payload,
            prompt_object=prompt_object,
            metadata={
                "model_parameters": {
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                }
            },
        )
        started = perf_counter()
        try:
            options: dict[str, Any] = {"temperature": temperature}
            if isinstance(max_output_tokens, int) and max_output_tokens > 0:
                options["num_predict"] = max_output_tokens
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": options,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("response", "").strip()
            latency_ms = round((perf_counter() - started) * 1000)
            metadata = self._build_metadata(
                payload,
                stage,
                temperature,
                latency_ms,
                task_id=task_id,
                model=self.model,
                prompt=full_prompt,
                response=text,
                options=options,
            )
            self.tracer.record_llm_call(
                trace,
                prompt=full_prompt,
                response=text,
                metadata=metadata,
            )
            return OllamaCallResult(text=text, metadata=metadata)
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            self.tracer.record_error(
                trace,
                prompt=full_prompt,
                error_message=str(exc),
                metadata={
                    "stage": stage,
                    "task_id": task_id,
                    "model": self.model,
                    "temperature": temperature,
                    "latency_ms": latency_ms,
                    "prompt_chars": len(full_prompt),
                    "response_chars": 0,
                    "response_empty": True,
                },
            )
            raise

    def _build_metadata(
        self,
        payload: dict,
        stage: str,
        temperature: float,
        latency_ms: int,
        *,
        task_id: str,
        model: str,
        prompt: str,
        response: str,
        options: dict[str, Any],
    ) -> dict:
        total_duration = payload.get("total_duration")
        if isinstance(total_duration, int):
            latency_ms = round(total_duration / 1_000_000)

        return {
            "stage": stage,
            "task_id": task_id,
            "section_name": _extract_section_name(task_id),
            "revision_round": _extract_revision_round(task_id),
            "model": model,
            "temperature": temperature,
            "output_budget": options.get("num_predict"),
            "latency_ms": latency_ms,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "response_empty": not bool(response.strip()),
            "tokens_prompt": payload.get("prompt_eval_count"),
            "tokens_response": payload.get("eval_count"),
            "done_reason": payload.get("done_reason"),
            "load_duration_ns": payload.get("load_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
        }


def _extract_section_name(task_id: str) -> str | None:
    for prefix in ("section_planner_", "writer_", "critic_"):
        if not task_id.startswith(prefix):
            continue
        tail = task_id.removeprefix(prefix)
        revision_marker = tail.find("_r")
        return tail[:revision_marker] if revision_marker != -1 else tail
    return None


def _extract_revision_round(task_id: str) -> int | None:
    match = re.search(r"_r(\d+)", task_id)
    return int(match.group(1)) if match else None
