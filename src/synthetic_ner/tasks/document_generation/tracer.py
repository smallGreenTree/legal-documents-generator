"""MLflow-backed tracing and prompt-registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from time import perf_counter
from typing import Any

import mlflow

from src.synthetic_ner.tasks.document_generation.trace_metrics import (
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
from src.synthetic_ner.types.app_config import MlflowConfig, WorkflowPromptsConfig
from src.synthetic_ner.types.trace import (
    DocumentTraceSession,
    NodeExecutionRecord,
    ResolvedWorkflowPrompts,
    TraceHandle,
)


class TraceStore:
    def __init__(
        self,
        cfg: MlflowConfig,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled
        self.run_metadata = dict(run_metadata or {})
        self._document_context = None
        self._document_observation = None
        self._node_runs: list[NodeExecutionRecord] = []
        self._llm_calls: list[dict[str, Any]] = []
        self._prompt_sync_summary = "MLflow prompts: not resolved"
        self.experiment_id: str | None = None
        self._current_session = DocumentTraceSession(
            enabled=self.enabled,
            trace_id=None,
            trace_url=None,
        )

        if not self.enabled:
            self.client = None
            return

        mlflow.set_tracking_uri(cfg.tracking_uri)
        experiment = mlflow.set_experiment(cfg.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.client = mlflow

    def start_document_run(
        self,
        *,
        doc_id: str,
        input_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> DocumentTraceSession:
        self.run_metadata.setdefault("doc_id", doc_id)
        self.run_metadata.setdefault("workflow_run_id", doc_id)
        session_id = str(
            self.run_metadata.setdefault(
                "mlflow_session_id",
                self.run_metadata["workflow_run_id"],
            )
        )
        if not self.enabled or self.client is None:
            return self._current_session

        self._document_context = self.client.start_span(
            name=f"{self.cfg.trace_name}:{doc_id}",
            span_type="WORKFLOW",
            attributes=self._metadata(metadata),
        )
        self._document_observation = self._document_context.__enter__()
        self.client.update_current_trace(
            session_id=session_id,
            tags={
                "service.name": self.cfg.service_name,
                "pipeline.stage": self.cfg.pipeline_stage,
            },
            metadata={"service.name": self.cfg.service_name},
        )
        self._document_observation.set_inputs(input_payload)
        trace_id = self._document_observation.trace_id
        trace_url = self._trace_url(trace_id)
        self._current_session = DocumentTraceSession(
            enabled=True,
            trace_id=trace_id,
            trace_url=trace_url,
        )
        self._flush()
        return self._current_session

    def end_document_run(self, *, output_payload: dict[str, Any] | None = None) -> None:
        if self._document_observation is not None and output_payload is not None:
            self._document_observation.set_outputs(output_payload)

        if self._document_context is not None:
            self._document_context.__exit__(None, None, None)

        self._flush()

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
            return self._run_langgraph_node_without_mlflow(
                doc_id=doc_id,
                node_name=node_name,
                state=state,
                runner=runner,
                started=started,
                next_node_resolver=next_node_resolver,
            )

        with self.client.start_span(
            name=node_name,
            span_type="CHAIN",
            attributes=self._metadata(
                build_langgraph_node_metadata(
                    doc_id=doc_id,
                    node_name=node_name,
                    state=state,
                    status="running",
                )
            ),
        ) as observation:
            observation.set_inputs(input_summary)
            try:
                result = runner()
            except Exception as exc:
                latency_ms = round((perf_counter() - started) * 1000)
                error_message = str(exc)
                observation.set_outputs({"error": error_message})
                observation.set_attributes(
                    self._metadata(
                        build_langgraph_node_metadata(
                            doc_id=doc_id,
                            node_name=node_name,
                            state=state,
                            latency_ms=latency_ms,
                            status="error",
                        )
                    )
                )
                observation.record_exception(exc)
                self._record_node_run(
                    node_name=node_name,
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
            observation.set_outputs(summarize_state(result))
            observation.set_attributes(
                self._metadata(
                    build_langgraph_node_metadata(
                        doc_id=doc_id,
                        node_name=node_name,
                        state=combined_state,
                        latency_ms=latency_ms,
                        next_node=next_node,
                        status="completed",
                    )
                )
            )
            self._record_node_run(
                node_name=node_name,
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

        span_context = self.client.start_span(
            name=task_id,
            span_type="LLM",
            attributes={**trace_metadata, "model": model},
        )
        observation = span_context.__enter__()
        observation.set_inputs(prompt_payload if prompt_payload is not None else prompt)
        self._flush()
        return TraceHandle(
            observation=observation,
            metadata=trace_metadata,
            context=span_context,
        )

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
        handle.observation.set_inputs(prompt)
        handle.observation.set_outputs(response)
        usage_details = build_usage_details(enriched_metadata) or {}
        handle.observation.set_attributes({**enriched_metadata, **usage_details})
        if rubrics:
            self._record_rubric_scores(handle, rubrics)
        if handle.context is not None:
            handle.context.__exit__(None, None, None)
        else:
            handle.observation.end()
        self._flush()

    def record_error(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        error_message: str,
        metadata: dict[str, Any],
    ) -> None:
        enriched_metadata = self._metadata(
            {
                **handle.metadata,
                **metadata,
                "error": True,
                "error_message": error_message,
            }
        )
        self._record_llm_call_metadata(enriched_metadata)
        if handle.observation is None:
            return
        handle.observation.set_inputs(prompt)
        error = RuntimeError(error_message)
        handle.observation.set_outputs(f"[error] {error_message}")
        handle.observation.set_attributes(enriched_metadata)
        handle.observation.record_exception(error)
        if handle.context is not None:
            handle.context.__exit__(type(error), error, error.__traceback__)
        else:
            handle.observation.end(status="ERROR")
        self._flush()

    def get_trace_info(self) -> DocumentTraceSession:
        return self._current_session

    def resolve_workflow_prompts(
        self,
        fallback_prompts: WorkflowPromptsConfig,
    ) -> ResolvedWorkflowPrompts:
        prompt_templates = {
            key: value
            for key, value in asdict(fallback_prompts).items()
            if isinstance(value, str) and value.strip()
        }
        resolved_templates: dict[str, str] = dict(prompt_templates)
        prompt_clients: dict[str, Any] = {}
        managed_count = 0
        seeded_count = 0
        fallback_count = 0
        error_count = 0

        if not self.enabled or self.client is None:
            self._prompt_sync_summary = "MLflow prompts disabled: using config.yaml prompts only"
            return ResolvedWorkflowPrompts(
                prompts=WorkflowPromptsConfig(**resolved_templates),
                prompt_clients=prompt_clients,
                sync_summary=self._prompt_sync_summary,
            )

        for key, fallback_template in prompt_templates.items():
            prompt_name = f"{self.cfg.prompt_name_prefix}.{key}"
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
            prompt_text = getattr(prompt_client, "template", None)
            if isinstance(prompt_text, str) and prompt_text.strip():
                resolved_templates[key] = prompt_text

        self._prompt_sync_summary = (
            "MLflow prompt sync: "
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

    def _run_langgraph_node_without_mlflow(
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
            status="completed",
            latency_ms=latency_ms,
            next_node=next_node,
        )
        return result

    def _record_node_run(
        self,
        *,
        node_name: str,
        status: str,
        latency_ms: int,
        next_node: str | None,
    ) -> None:
        self._node_runs.append(
            NodeExecutionRecord(
                node_name=node_name,
                status=status,
                latency_ms=latency_ms,
                next_node=next_node,
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
                "mlflow_session_id": metadata.get("mlflow_session_id"),
                "critic_rubrics": metadata.get("critic_rubrics"),
                **{key: value for key, value in metadata.items() if key.startswith("rubric_")},
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

        alias = optional_env("MLFLOW_PROMPT_ALIAS") or self.cfg.prompt_alias

        try:
            prompt = self.client.genai.load_prompt(
                f"prompts:/{name}@{alias}",
                cache_ttl_seconds=300,
                link_to_model=False,
            )
            return prompt, "managed"
        except Exception as exc:
            get_error = exc

        try:
            prompt_client = self.client.genai.register_prompt(
                name=name,
                template=fallback_template,
                commit_message="Seeded from synthetic-ner fallback prompt",
                tags={"source": "synthetic-ner"},
            )
            self.client.genai.set_prompt_alias(name, alias, prompt_client.version)
            print(f"  Prompts : seeded '{name}' in MLflow with alias={alias}")
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
            observation.set_attribute(f"rubric.{metric}", score_value)

        if score_values:
            overall = round(sum(score_values) / len(score_values), 2)
            observation.set_attribute("rubric.overall", overall)

    def _metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            key: value
            for key, value in {**self.run_metadata, **(metadata or {})}.items()
            if value is not None
        }

    def _trace_url(self, trace_id: str) -> str:
        base = self.cfg.tracking_uri.rstrip("/")
        return f"{base}/#/experiments/{self.experiment_id}/traces/{trace_id}"

    def _flush(self) -> None:
        if self.client is None:
            return
        flush = getattr(self.client, "flush_trace_async_logging", None)
        if callable(flush):
            flush()


def _flatten_rubrics(rubrics: dict[str, int]) -> dict[str, int | float]:
    flattened = {f"rubric_{metric}": score for metric, score in rubrics.items()}
    valid_scores = [score for score in rubrics.values() if 1 <= score <= 5]
    if valid_scores:
        flattened["rubric_overall"] = round(sum(valid_scores) / len(valid_scores), 2)
    return flattened
