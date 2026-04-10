"""Traced Ollama client."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import requests
from src.synthetic_ner.tasks.tracer import TraceStore
from src.synthetic_ner.types.app_config import OllamaConfig


@dataclass(slots=True)
class OllamaCallResult:
    text: str
    trace_id: str | None
    metadata: dict


class TracedOllamaClient:
    """Small wrapper around Ollama's generate API with full prompt tracing."""

    def __init__(
        self,
        config: OllamaConfig,
        tracer: TraceStore,
    ) -> None:
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
    ) -> OllamaCallResult:
        full_prompt = (
            f"[SYSTEM]\n{system_prompt.strip()}\n\n"
            f"[USER]\n{user_prompt.strip()}\n"
        )
        trace = self.tracer.start_trace(
            doc_id=doc_id,
            task_id=task_id,
            stage=stage,
            model=self.model,
            parent_task_id=parent_task_id,
            prompt=full_prompt,
            metadata={"temperature": temperature},
        )

        started = perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("response", "").strip()
            latency_ms = round((perf_counter() - started) * 1000)
            metadata = self._build_metadata(payload, temperature, latency_ms)
            self.tracer.record_llm_call(
                trace,
                prompt=full_prompt,
                response=text,
                metadata=metadata,
            )
            return OllamaCallResult(text=text, trace_id=trace.trace_id, metadata=metadata)
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            metadata = {
                "temperature": temperature,
                "latency_ms": latency_ms,
                "error": str(exc),
            }
            self.tracer.record_error(
                trace,
                prompt=full_prompt,
                error_message=str(exc),
                metadata=metadata,
            )
            raise

    def _build_metadata(
        self,
        payload: dict,
        temperature: float,
        latency_ms: int,
    ) -> dict:
        total_duration = payload.get("total_duration")
        if isinstance(total_duration, int):
            latency_ms = round(total_duration / 1_000_000)

        return {
            "temperature": temperature,
            "latency_ms": latency_ms,
            "tokens_prompt": payload.get("prompt_eval_count"),
            "tokens_response": payload.get("eval_count"),
            "done_reason": payload.get("done_reason"),
            "load_duration_ns": payload.get("load_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
        }
