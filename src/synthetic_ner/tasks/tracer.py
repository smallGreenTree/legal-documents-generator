"""Langfuse-backed tracing helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict
from time import perf_counter
from typing import Any

from langfuse import Langfuse
from src.synthetic_ner.tasks.trace_metrics import (
    build_langgraph_node_metadata,
    build_prompt_metadata,
    build_usage_details,
    extract_rubric_scores,
    merge_state,
    optional_env,
    summarize_llm_calls,
    summarize_node_runs,
    summarize_state,
)
from src.synthetic_ner.types.app_config import LangfuseConfig, WorkflowPromptsConfig
from src.synthetic_ner.types.trace import (
    DocumentTraceSession,
    NodeExecutionRecord,
    ResolvedWorkflowPrompts,
    TraceHandle,
)


class TraceStore:
    def __init__(
        self,
        cfg: LangfuseConfig,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.run_metadata = dict(run_metadata or {})
        self._document_context = None
        self._document_observation = None
        self._document_trace_context: dict[str, str] | None = None
        self._node_runs: list[NodeExecutionRecord] = []
        self._llm_calls: list[dict[str, Any]] = []
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
        self.run_metadata.setdefault("doc_id", doc_id)
        self.run_metadata.setdefault("langfuse_group_id", doc_id)
        self.run_metadata.setdefault("langfuse_trace_seed", doc_id)
        self.run_metadata.setdefault("workflow_run_id", doc_id)
        if not self.enabled or self.client is None:
            return self._current_session

        trace_id = self.client.create_trace_id(seed=doc_id)
        self._document_trace_context = {"trace_id": trace_id}
        self._document_context = self.client.start_as_current_observation(
            trace_context=self._document_trace_context,
            as_type="span",
            name=f"{name}:{doc_id}",
            input=input_payload,
            metadata=self._metadata(metadata),
        )
        self._document_observation = self._document_context.__enter__()
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
        input_summary = summarize_state(state)
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
            trace_context=self._trace_context(doc_id),
            name=node_name,
            as_type="span",
            input=input_summary,
            metadata=self._metadata(
                build_langgraph_node_metadata(
                    doc_id=doc_id,
                    node_name=node_name,
                    state=state,
                    status="running",
                )
            ),
        ) as observation:
            try:
                result = runner()
            except Exception as exc:
                latency_ms = round((perf_counter() - started) * 1000)
                error_message = str(exc)
                observation.update(
                    output={"error": error_message},
                    metadata=self._metadata(
                        build_langgraph_node_metadata(
                            doc_id=doc_id,
                            node_name=node_name,
                            state=state,
                            latency_ms=latency_ms,
                            status="error",
                        )
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

            combined_state = merge_state(state, result)
            next_node = (
                next_node_resolver(combined_state) if next_node_resolver is not None else None
            )
            latency_ms = round((perf_counter() - started) * 1000)
            observation.update(
                output=summarize_state(result),
                metadata=self._metadata(
                    build_langgraph_node_metadata(
                        doc_id=doc_id,
                        node_name=node_name,
                        state=combined_state,
                        latency_ms=latency_ms,
                        next_node=next_node,
                        status="completed",
                    )
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
        trace_metadata = self._metadata(
            {
                "doc_id": doc_id,
                "stage": stage,
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                **build_prompt_metadata(prompt_object),
                **(metadata or {}),
            }
        )
        if not self.enabled or self.client is None:
            return TraceHandle(observation=None, metadata=trace_metadata)

        observation = self.client.start_observation(
            trace_context=self._trace_context(doc_id),
            name=task_id,
            as_type="generation",
            model=model,
            input=prompt_payload if prompt_payload is not None else prompt,
            metadata=trace_metadata,
            prompt=None,
            model_parameters=(metadata or {}).get("model_parameters"),
        )
        return TraceHandle(observation=observation, metadata=trace_metadata)

    def record_llm_call(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        response: str,
        metadata: dict[str, Any],
    ) -> None:
        enriched_metadata = self._metadata({**handle.metadata, **metadata})
        rubrics = extract_rubric_scores(response) if metadata.get("stage") == "critic" else {}
        if rubrics:
            enriched_metadata["critic_rubrics"] = rubrics
            enriched_metadata.update(_flatten_rubrics(rubrics))
        self._record_llm_call_metadata(enriched_metadata)
        if handle.observation is None:
            return
        handle.observation.update(
            input=prompt,
            output=response,
            metadata=enriched_metadata,
            usage_details=build_usage_details(enriched_metadata),
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
        enriched_metadata = self._metadata({
            **handle.metadata,
            **metadata,
            "error": True,
            "error_message": error_message,
        })
        self._record_llm_call_metadata(enriched_metadata)
        if handle.observation is None:
            return
        handle.observation.update(
            input=prompt,
            output=f"[error] {error_message}",
            metadata=enriched_metadata,
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
        return summarize_node_runs(self._node_runs)

    def get_llm_call_records(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self._llm_calls]

    def get_llm_run_summary(self) -> dict[str, Any]:
        return summarize_llm_calls(self.get_llm_call_records())

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

        combined_state = merge_state(state, result)
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

    def _record_llm_call_metadata(self, metadata: dict[str, Any]) -> None:
        self._llm_calls.append(
            {
                "task_id": metadata.get("task_id"),
                "stage": metadata.get("stage"),
                "section_name": metadata.get("section_name"),
                "revision_round": metadata.get("revision_round"),
                "model": metadata.get("model"),
                "latency_ms": metadata.get("latency_ms"),
                "prompt_chars": metadata.get("prompt_chars"),
                "response_chars": metadata.get("response_chars"),
                "tokens_prompt": metadata.get("tokens_prompt"),
                "tokens_response": metadata.get("tokens_response"),
                "output_budget": metadata.get("output_budget"),
                "done_reason": metadata.get("done_reason"),
                "response_empty": metadata.get("response_empty"),
                "error": metadata.get("error", False),
                "error_message": metadata.get("error_message"),
                "workflow_run_id": metadata.get("workflow_run_id"),
                "prefect_flow_run_id": metadata.get("prefect_flow_run_id"),
                "doc_id": metadata.get("doc_id"),
                "langfuse_group_id": metadata.get("langfuse_group_id"),
                "critic_rubrics": metadata.get("critic_rubrics"),
                **{
                    key: value
                    for key, value in metadata.items()
                    if key.startswith("rubric_")
                },
            }
        )

    def _get_or_seed_prompt(
        self,
        *,
        name: str,
        fallback_template: str,
    ) -> tuple[Any | None, str]:
        if self.client is None:
            return None, "fallback"

        label = optional_env("LANGFUSE_PROMPT_LABEL")
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

    def _metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            key: value
            for key, value in {**self.run_metadata, **(metadata or {})}.items()
            if value is not None
        }

    def _trace_context(self, doc_id: str) -> dict[str, str] | None:
        if not self.enabled or self.client is None:
            return None
        if self._document_trace_context is not None:
            return self._document_trace_context
        return {"trace_id": self.client.create_trace_id(seed=doc_id)}


def _flatten_rubrics(rubrics: dict[str, int]) -> dict[str, int | float]:
    flattened = {f"rubric_{metric}": score for metric, score in rubrics.items()}
    valid_scores = [score for score in rubrics.values() if 1 <= score <= 5]
    if valid_scores:
        flattened["rubric_overall"] = round(sum(valid_scores) / len(valid_scores), 2)
    return flattened
