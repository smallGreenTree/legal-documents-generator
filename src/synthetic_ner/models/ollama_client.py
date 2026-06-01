"""Traced Ollama client."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Callable

import requests
from src.synthetic_ner.tasks.document_generation.tracer import TraceStore
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
        self.num_ctx = config.num_ctx
        self.think = config.think
        self.recovery = config.recovery
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
        on_partial_text: Callable[[str], None] | None = None,
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
                    "num_ctx": self.num_ctx,
                    "think": self.think,
                }
            },
        )
        started = perf_counter()
        partial_text = ""
        options: dict[str, Any] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            options["num_predict"] = max_output_tokens

        def remember_partial_text(value: str) -> None:
            nonlocal partial_text
            partial_text = value
            if on_partial_text is not None:
                on_partial_text(value)

        try:
            payload, text = self._generate_with_retries(
                full_prompt=full_prompt,
                options=options,
                on_partial_text=remember_partial_text if on_partial_text is not None else None,
            )
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
            if on_partial_text is not None and partial_text.strip():
                text = partial_text.strip()
                metadata = {
                    "stage": stage,
                    "task_id": task_id,
                    "section_name": _extract_section_name(task_id),
                    "revision_round": _extract_revision_round(task_id),
                    "model": self.model,
                    "temperature": temperature,
                    "output_budget": options.get("num_predict") if "options" in locals() else None,
                    "context_window": options.get("num_ctx") if "options" in locals() else None,
                    "latency_ms": latency_ms,
                    "prompt_chars": len(full_prompt),
                    "response_chars": len(text),
                    "response_empty": False,
                    "tokens_prompt": None,
                    "tokens_response": None,
                    "done_reason": "partial_error",
                    "error": True,
                    "error_message": str(exc),
                }
                self.tracer.record_llm_call(
                    trace,
                    prompt=full_prompt,
                    response=text,
                    metadata=metadata,
                )
                return OllamaCallResult(text=text, metadata=metadata)
            if stage == "writer":
                metadata = {
                    "stage": stage,
                    "task_id": task_id,
                    "section_name": _extract_section_name(task_id),
                    "revision_round": _extract_revision_round(task_id),
                    "model": self.model,
                    "temperature": temperature,
                    "output_budget": options.get("num_predict"),
                    "context_window": options.get("num_ctx"),
                    "latency_ms": latency_ms,
                    "prompt_chars": len(full_prompt),
                    "response_chars": len(self.recovery.controlled_empty_section),
                    "response_empty": False,
                    "tokens_prompt": None,
                    "tokens_response": None,
                    "done_reason": "controlled_writer_fallback",
                    "error": True,
                    "error_message": str(exc),
                }
                self.tracer.record_llm_call(
                    trace,
                    prompt=full_prompt,
                    response=self.recovery.controlled_empty_section,
                    metadata=metadata,
                )
                return OllamaCallResult(
                    text=self.recovery.controlled_empty_section,
                    metadata=metadata,
                )
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

    def _generate_with_retries(
        self,
        *,
        full_prompt: str,
        options: dict[str, Any],
        on_partial_text: Callable[[str], None] | None,
    ) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        for attempt in range(1, self.recovery.max_generate_attempts + 1):
            try:
                return self._generate(
                    full_prompt=full_prompt,
                    options=options,
                    on_partial_text=on_partial_text,
                )
            except Exception as exc:
                last_error = exc
                if (
                    attempt >= self.recovery.max_generate_attempts
                    or not _is_retryable_error(exc)
                ):
                    break
                sleep(self.recovery.retry_backoff_seconds * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama generation failed without an exception")

    def _generate(
        self,
        *,
        full_prompt: str,
        options: dict[str, Any],
        on_partial_text: Callable[[str], None] | None,
    ) -> tuple[dict[str, Any], str]:
        if on_partial_text is None:
            request_json: dict[str, Any] = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": options,
            }
            if self.think is not None:
                request_json["think"] = self.think
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_json,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return payload, payload.get("response", "").strip()

        request_json = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": options,
        }
        if self.think is not None:
            request_json["think"] = self.think
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=request_json,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        chunks: list[str] = []
        final_payload: dict[str, Any] = {}
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if payload.get("thinking"):
                final_payload["thinking"] = (
                    final_payload.get("thinking", "") + payload["thinking"]
                )
            piece = payload.get("response", "")
            if piece:
                chunks.append(piece)
                on_partial_text("".join(chunks))
            if payload.get("done") is True:
                if final_payload.get("thinking") and not payload.get("thinking"):
                    payload["thinking"] = final_payload["thinking"]
                final_payload = payload
                break
        text = "".join(chunks).strip()
        final_payload.setdefault("response", text)
        return final_payload, text

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
            "context_window": options.get("num_ctx"),
            "latency_ms": latency_ms,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "response_empty": not bool(response.strip()),
            "thinking_chars": len(str(payload.get("thinking") or "")),
            "tokens_prompt": payload.get("prompt_eval_count"),
            "tokens_response": payload.get("eval_count"),
            "done_reason": payload.get("done_reason"),
            "load_duration_ns": payload.get("load_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
        }


def _extract_section_name(task_id: str) -> str | None:
    for prefix in ("section_planner_", "writer_", "polish_", "critic_"):
        if not task_id.startswith(prefix):
            continue
        tail = task_id.removeprefix(prefix)
        revision_marker = tail.find("_r")
        return tail[:revision_marker] if revision_marker != -1 else tail
    return None


def _extract_revision_round(task_id: str) -> int | None:
    match = re.search(r"_r(\d+)", task_id)
    return int(match.group(1)) if match else None


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and 500 <= response.status_code < 600
    return False
