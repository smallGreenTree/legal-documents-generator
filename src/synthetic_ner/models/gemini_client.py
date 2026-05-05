"""Traced Gemini client."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Callable

import requests
from src.synthetic_ner.tasks.tracer import TraceStore
from src.synthetic_ner.types.app_config import ModelProviderConfig


@dataclass(slots=True)
class GeminiCallResult:
    text: str
    metadata: dict


class TracedGeminiClient:
    """Gemini generateContent wrapper with the same invoke contract as Ollama."""

    def __init__(self, config: ModelProviderConfig, tracer: TraceStore) -> None:
        if not config.api_key_env:
            raise ValueError("Gemini provider requires api_key_env")
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini API key is missing. Expected env var '{config.api_key_env}'."
            )
        self.api_key = api_key
        self.base_url = (config.base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("Gemini provider requires base_url")
        self.model = config.model
        self.timeout = config.timeout
        self.thinking_budget = config.thinking_budget
        self.max_generate_attempts = config.max_generate_attempts
        self.retry_backoff_seconds = config.retry_backoff_seconds
        self.min_interval_seconds = config.min_interval_seconds
        self._last_request_at: float | None = None
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
    ) -> GeminiCallResult:
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
                "provider": "gemini",
                "model_parameters": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            },
        )
        started = perf_counter()
        try:
            payload = self._generate_with_retries(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            text = _extract_text(payload)
            if on_partial_text is not None and text:
                on_partial_text(text)
            latency_ms = round((perf_counter() - started) * 1000)
            metadata = _build_metadata(
                payload,
                stage=stage,
                task_id=task_id,
                model=self.model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                prompt=full_prompt,
                response=text,
                latency_ms=latency_ms,
            )
            self.tracer.record_llm_call(
                trace,
                prompt=full_prompt,
                response=text,
                metadata=metadata,
            )
            return GeminiCallResult(text=text, metadata=metadata)
        except Exception as exc:
            latency_ms = round((perf_counter() - started) * 1000)
            self.tracer.record_error(
                trace,
                prompt=full_prompt,
                error_message=_safe_error_message(exc),
                metadata={
                    "provider": "gemini",
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
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_generate_attempts + 1):
            try:
                return self._generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_generate_attempts or not _is_retryable_error(exc):
                    break
                sleep(max(self.retry_backoff_seconds * attempt, _retry_after_seconds(exc)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini generation failed without an exception")

    def _generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {"temperature": temperature}
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            generation_config["maxOutputTokens"] = max_output_tokens
        if self.thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget,
            }
        self._respect_min_interval()
        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt.strip()}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt.strip()}],
                    }
                ],
                "generationConfig": generation_config,
            },
            timeout=self.timeout,
        )
        self._last_request_at = perf_counter()
        response.raise_for_status()
        return response.json()

    def _respect_min_interval(self) -> None:
        if self.min_interval_seconds <= 0 or self._last_request_at is None:
            return
        elapsed = perf_counter() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            sleep(remaining)


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "\n".join(texts).strip()


def _build_metadata(
    payload: dict[str, Any],
    *,
    stage: str,
    task_id: str,
    model: str,
    temperature: float,
    max_output_tokens: int | None,
    prompt: str,
    response: str,
    latency_ms: int,
) -> dict[str, Any]:
    usage = payload.get("usageMetadata", {})
    finish_reason = None
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        finish_reason = candidates[0].get("finishReason")
    return {
        "provider": "gemini",
        "stage": stage,
        "task_id": task_id,
        "section_name": _extract_section_name(task_id),
        "revision_round": _extract_revision_round(task_id),
        "model": model,
        "temperature": temperature,
        "output_budget": max_output_tokens,
        "latency_ms": latency_ms,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "response_empty": not bool(response.strip()),
        "tokens_prompt": usage.get("promptTokenCount"),
        "tokens_response": usage.get("candidatesTokenCount"),
        "tokens_total": usage.get("totalTokenCount"),
        "done_reason": finish_reason,
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


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and (
            response.status_code == 429 or 500 <= response.status_code < 600
        )
    return False


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"{exc.response.status_code} {exc.response.reason}".strip()
    return str(exc)


def _retry_after_seconds(exc: Exception) -> float:
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return 0.0
    raw_value = exc.response.headers.get("Retry-After")
    if raw_value is None:
        return 0.0
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        return 0.0
