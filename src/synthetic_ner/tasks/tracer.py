"""Langfuse-backed tracing helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from langfuse import Langfuse
from src.synthetic_ner.types.app_config import LangfuseConfig, WorkflowPromptsConfig
from src.synthetic_ner.types.trace import (
    DocumentTraceSession,
    NodeExecutionRecord,
    ResolvedWorkflowPrompts,
    TraceHandle,
)

_RUBRIC_LINE_RE = re.compile(
    r"^\s*-\s*([a-zA-Z][a-zA-Z0-9_ -]{1,40})\s*:\s*([1-5])(?:\s*/\s*5)?\s*$",
    re.MULTILINE,
)


class TraceStore:
    def __init__(self, cfg: LangfuseConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled
        self._document_context = None
        self._document_observation = None
        self._node_runs: list[NodeExecutionRecord] = []
        self._prompt_sync_summary = "Langfuse prompts: not resolved"
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

    def run_langgraph_node(
        self,
        *,
        doc_id: str,
        node_name: str,
        state: Mapping[str, Any],
        runner: Callable[[], dict[str, Any]],
        next_node_resolver: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> dict[str, Any]:
        input_summary = _summarize_state(state)
        started = perf_counter()

        if not self.enabled or self.client is None:
            return self._run_langgraph_node_without_langfuse(
                doc_id=doc_id,
                node_name=node_name,
                state=state,
                runner=runner,
                started=started,
                next_node_resolver=next_node_resolver,
            )

        with self.client.start_as_current_observation(
            name=node_name,
            as_type="span",
            input=input_summary,
            metadata=_build_langgraph_node_metadata(
                doc_id=doc_id,
                node_name=node_name,
                state=state,
                status="running",
            ),
        ) as observation:
            try:
                result = runner()
            except Exception as exc:
                latency_ms = round((perf_counter() - started) * 1000)
                error_message = str(exc)
                observation.update(
                    output={"error": error_message},
                    metadata=_build_langgraph_node_metadata(
                        doc_id=doc_id,
                        node_name=node_name,
                        state=state,
                        latency_ms=latency_ms,
                        status="error",
                    ),
                    level="ERROR",
                    status_message=error_message,
                )
                self._record_node_run(
                    node_name=node_name,
                    state=state,
                    status="error",
                    latency_ms=latency_ms,
                    next_node=None,
                )
                raise

            combined_state = _merge_state(state, result)
            next_node = (
                next_node_resolver(combined_state) if next_node_resolver is not None else None
            )
            latency_ms = round((perf_counter() - started) * 1000)
            observation.update(
                output=_summarize_state(result),
                metadata=_build_langgraph_node_metadata(
                    doc_id=doc_id,
                    node_name=node_name,
                    state=combined_state,
                    latency_ms=latency_ms,
                    next_node=next_node,
                    status="completed",
                ),
            )
            self._record_node_run(
                node_name=node_name,
                state=combined_state,
                status="completed",
                latency_ms=latency_ms,
                next_node=next_node,
            )
            return result

    def start_trace(
        self,
        *,
        doc_id: str,
        task_id: str,
        stage: str,
        model: str,
        parent_task_id: str | None = None,
        prompt: str | None = None,
        prompt_payload: dict[str, Any] | None = None,
        prompt_object: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceHandle:
        del doc_id
        if not self.enabled or self.client is None:
            return TraceHandle(observation=None)

        prompt_metadata = _build_prompt_metadata(prompt_object)
        observation = self.client.start_observation(
            name=task_id,
            as_type="generation",
            model=model,
            input=prompt_payload if prompt_payload is not None else prompt,
            metadata={
                "stage": stage,
                "parent_task_id": parent_task_id,
                **prompt_metadata,
                **(metadata or {}),
            },
            prompt=None,
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
        enriched_metadata = dict(metadata)
        rubrics = _extract_rubric_scores(response) if metadata.get("stage") == "critic" else {}
        if rubrics:
            enriched_metadata["critic_rubrics"] = rubrics
        handle.observation.update(
            input=prompt,
            output=response,
            metadata=enriched_metadata,
            usage_details=_build_usage_details(enriched_metadata),
        )
        if rubrics:
            self._record_rubric_scores(handle, rubrics)
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

    def resolve_workflow_prompts(
        self,
        fallback_prompts: WorkflowPromptsConfig,
    ) -> ResolvedWorkflowPrompts:
        prompt_templates = asdict(fallback_prompts)
        resolved_templates: dict[str, str] = dict(prompt_templates)
        prompt_clients: dict[str, Any] = {}
        managed_count = 0
        seeded_count = 0
        fallback_count = 0
        error_count = 0

        if not self.enabled or self.client is None:
            self._prompt_sync_summary = "Langfuse prompts disabled: using config.yaml prompts only"
            return ResolvedWorkflowPrompts(
                prompts=WorkflowPromptsConfig(**resolved_templates),
                prompt_clients=prompt_clients,
                sync_summary=self._prompt_sync_summary,
            )

        for key, fallback_template in prompt_templates.items():
            prompt_name = f"synthetic_ner.{key}"
            prompt_client, status = self._get_or_seed_prompt(
                name=prompt_name,
                fallback_template=fallback_template,
            )
            if status == "managed":
                managed_count += 1
            elif status == "seeded":
                seeded_count += 1
            elif status == "fallback":
                fallback_count += 1
            else:
                error_count += 1
            if prompt_client is None:
                continue

            prompt_clients[key] = prompt_client
            prompt_text = getattr(prompt_client, "prompt", None)
            if isinstance(prompt_text, str) and prompt_text.strip():
                resolved_templates[key] = prompt_text

        self._prompt_sync_summary = (
            "Langfuse prompt sync: "
            f"managed={managed_count}, seeded={seeded_count}, "
            f"fallback={fallback_count}, errors={error_count}"
        )
        return ResolvedWorkflowPrompts(
            prompts=WorkflowPromptsConfig(**resolved_templates),
            prompt_clients=prompt_clients,
            sync_summary=self._prompt_sync_summary,
        )

    def get_langgraph_node_summary(self) -> list[dict[str, Any]]:
        summary_by_node: dict[str, dict[str, Any]] = {}
        execution_order: list[str] = []

        for run in self._node_runs:
            if run.node_name not in summary_by_node:
                summary_by_node[run.node_name] = {
                    "executions": 0,
                    "total_latency_ms": 0,
                    "max_latency_ms": 0,
                    "errors": 0,
                    "next_nodes": set(),
                }
                execution_order.append(run.node_name)

            bucket = summary_by_node[run.node_name]
            bucket["executions"] += 1
            bucket["total_latency_ms"] += run.latency_ms
            bucket["max_latency_ms"] = max(bucket["max_latency_ms"], run.latency_ms)
            if run.status == "error":
                bucket["errors"] += 1
            if run.next_node:
                bucket["next_nodes"].add(run.next_node)

        rows: list[dict[str, Any]] = []
        for node_name in execution_order:
            bucket = summary_by_node[node_name]
            executions = bucket["executions"]
            avg_latency_ms = round(bucket["total_latency_ms"] / executions) if executions else 0
            rows.append(
                {
                    "node_name": node_name,
                    "executions": executions,
                    "avg_latency_ms": avg_latency_ms,
                    "max_latency_ms": bucket["max_latency_ms"],
                    "errors": bucket["errors"],
                    "next_nodes": sorted(bucket["next_nodes"]),
                }
            )

        return rows

    def _run_langgraph_node_without_langfuse(
        self,
        *,
        doc_id: str,
        node_name: str,
        state: Mapping[str, Any],
        runner: Callable[[], dict[str, Any]],
        started: float,
        next_node_resolver: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> dict[str, Any]:
        del doc_id
        try:
            result = runner()
        except Exception:
            latency_ms = round((perf_counter() - started) * 1000)
            self._record_node_run(
                node_name=node_name,
                state=state,
                status="error",
                latency_ms=latency_ms,
                next_node=None,
            )
            raise

        combined_state = _merge_state(state, result)
        next_node = next_node_resolver(combined_state) if next_node_resolver is not None else None
        latency_ms = round((perf_counter() - started) * 1000)
        self._record_node_run(
            node_name=node_name,
            state=combined_state,
            status="completed",
            latency_ms=latency_ms,
            next_node=next_node,
        )
        return result

    def _record_node_run(
        self,
        *,
        node_name: str,
        state: Mapping[str, Any],
        status: str,
        latency_ms: int,
        next_node: str | None,
    ) -> None:
        section_name = state.get("current_section")
        self._node_runs.append(
            NodeExecutionRecord(
                node_name=node_name,
                status=status,
                latency_ms=latency_ms,
                next_node=next_node,
                section_name=section_name if isinstance(section_name, str) else None,
            )
        )

    def _get_or_seed_prompt(
        self,
        *,
        name: str,
        fallback_template: str,
    ) -> tuple[Any | None, str]:
        if self.client is None:
            return None, "fallback"

        label = _optional_env("LANGFUSE_PROMPT_LABEL")
        auto_seed = os.getenv("LANGFUSE_PROMPT_AUTOSEED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        try:
            get_prompt_kwargs: dict[str, Any] = {
                "name": name,
                "type": "text",
                "cache_ttl_seconds": 300,
                "fetch_timeout_seconds": 3,
                "max_retries": 1,
            }
            if label:
                get_prompt_kwargs["label"] = label
            return self.client.get_prompt(**get_prompt_kwargs), "managed"
        except Exception as exc:
            get_error = exc
            if not auto_seed:
                print(f"  Prompts : failed to fetch '{name}' ({exc}); using config fallback")
                return None, "fallback"

        try:
            labels = [label] if label else ["production"]
            prompt_client = self.client.create_prompt(
                name=name,
                prompt=fallback_template,
                labels=labels,
                type="text",
                commit_message="Seeded from synthetic-ner fallback prompt",
            )
            print(f"  Prompts : seeded '{name}' in Langfuse with labels={labels}")
            return prompt_client, "seeded"
        except Exception as exc:
            print(
                f"  Prompts : failed to seed '{name}' "
                f"(fetch_error={get_error}; seed_error={exc}); using config fallback"
            )
            return None, "error"

    def _record_rubric_scores(self, handle: TraceHandle, rubrics: dict[str, int]) -> None:
        observation = handle.observation
        if observation is None:
            return

        score_values: list[float] = []
        for metric, score in sorted(rubrics.items()):
            if not (1 <= score <= 5):
                continue
            score_value = float(score)
            score_values.append(score_value)
            observation.score(
                name=f"rubric.{metric}",
                value=score_value,
                data_type="NUMERIC",
                comment=f"{score}/5",
            )

        if score_values:
            overall = round(sum(score_values) / len(score_values), 2)
            observation.score(
                name="rubric.overall",
                value=overall,
                data_type="NUMERIC",
                comment="Average rubric score (1-5)",
            )


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


def _build_prompt_metadata(prompt_object: Any | None) -> dict[str, Any]:
    if prompt_object is None:
        return {}

    metadata: dict[str, Any] = {"managed_prompt_attached": False}
    for attr_name, metadata_key in (
        ("name", "managed_prompt_name"),
        ("version", "managed_prompt_version"),
        ("variables", "managed_prompt_variables"),
    ):
        value = getattr(prompt_object, attr_name, None)
        if value is not None:
            metadata[metadata_key] = value
    return metadata


def _build_langgraph_node_metadata(
    *,
    doc_id: str,
    node_name: str,
    state: Mapping[str, Any],
    latency_ms: int | None = None,
    next_node: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "component": "langgraph_node",
        "workflow_mode": "langgraph",
        "node_name": node_name,
        "doc_id": _string_value(state.get("doc_id")) or doc_id,
    }

    current_section = _string_value(state.get("current_section"))
    if current_section:
        metadata["current_section"] = current_section

    _add_state_counts(metadata, state)
    _add_text_lengths(metadata, state)

    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms
    if next_node is not None:
        metadata["next_node"] = next_node
    if status is not None:
        metadata["status"] = status

    return metadata


def _merge_state(state: Mapping[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(state)
    if updates is not None:
        merged.update(dict(updates))
    return merged


def _add_state_counts(metadata: dict[str, Any], state: Mapping[str, Any]) -> None:
    for field_name in ("section_index", "revision_count"):
        field_value = state.get(field_name)
        if isinstance(field_value, int):
            metadata[field_name] = field_value

    collection_lengths = {
        "issues_count": state.get("current_section_issues"),
        "section_order_count": state.get("section_order"),
        "section_outputs_count": state.get("section_outputs"),
        "section_plans_count": state.get("section_plans"),
        "section_reviews_count": state.get("section_reviews"),
    }
    for field_name, field_value in collection_lengths.items():
        if isinstance(field_value, (list, dict)):
            metadata[field_name] = len(field_value)


def _add_text_lengths(metadata: dict[str, Any], state: Mapping[str, Any]) -> None:
    for field_name in (
        "memory_text",
        "document_plan",
        "current_section_plan",
        "current_section_text",
        "current_revision_instruction",
        "final_text",
    ):
        field_value = state.get(field_name)
        if isinstance(field_value, str):
            metadata[f"{field_name}_chars"] = len(field_value)


def _summarize_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {key: _summarize_value(value) for key, value in state.items()}


def _summarize_value(value: Any) -> Any:
    if isinstance(value, str):
        return {
            "type": "str",
            "chars": len(value),
            "preview": _preview_text(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        items = list(value.items())
        summary: dict[str, Any] = {
            "type": "dict",
            "size": len(value),
            "keys": [str(key) for key, _ in items[:8]],
        }
        string_values = {
            str(key): len(item_value)
            for key, item_value in items[:8]
            if isinstance(item_value, str)
        }
        if string_values:
            summary["value_chars"] = string_values
        return summary
    if isinstance(value, (list, tuple)):
        summary = {
            "type": "list",
            "size": len(value),
        }
        if value and all(isinstance(item, str) for item in value[:8]):
            summary["items"] = [_preview_text(item, limit=80) for item in value[:8]]
        return summary
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return repr(value)


def _preview_text(value: str, *, limit: int = 180) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    trimmed = value.strip()
    return trimmed or None


def _extract_rubric_scores(raw_text: str) -> dict[str, int]:
    rubrics_marker = raw_text.find("RUBRICS:")
    if rubrics_marker == -1:
        return {}

    issues_marker = raw_text.find("ISSUES:")
    revision_marker = raw_text.find("REVISION:")
    block_end = (
        issues_marker
        if issues_marker != -1
        else (revision_marker if revision_marker != -1 else len(raw_text))
    )
    rubric_block = raw_text[rubrics_marker + len("RUBRICS:"):block_end]

    rubrics: dict[str, int] = {}
    for metric, raw_score in _RUBRIC_LINE_RE.findall(rubric_block):
        key = metric.strip().lower().replace(" ", "_").replace("-", "_")
        score = int(raw_score)
        if key and 1 <= score <= 5:
            rubrics[key] = score
    return rubrics
