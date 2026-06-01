"""Pure tracing summary and metadata helpers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.synthetic_ner.types.trace import NodeExecutionRecord

_RUBRIC_LINE_RE = re.compile(
    r"^\s*-\s*([a-zA-Z][a-zA-Z0-9_ -]{1,40})\s*:\s*([1-5])(?:\s*/\s*5)?\s*$",
    re.MULTILINE,
)


def summarize_node_runs(runs: list[NodeExecutionRecord]) -> list[dict[str, Any]]:
    summary_by_node: dict[str, dict[str, Any]] = {}
    execution_order: list[str] = []

    for run in runs:
        bucket = summary_by_node.setdefault(
            run.node_name,
            {
                "executions": 0,
                "total_latency_ms": 0,
                "max_latency_ms": 0,
                "errors": 0,
                "next_nodes": set(),
            },
        )
        if bucket["executions"] == 0:
            execution_order.append(run.node_name)
        _add_node_run(bucket, run)

    return [
        _node_summary_row(node_name, summary_by_node[node_name])
        for node_name in execution_order
    ]


def summarize_llm_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    revised_sections: set[str] = set()

    for call in calls:
        _add_llm_call(by_stage, revised_sections, call)

    return {
        "total_llm_calls": len(calls),
        "total_latency_ms": sum(_int_value(call.get("latency_ms")) for call in calls),
        "total_prompt_tokens": sum(_int_value(call.get("tokens_prompt")) for call in calls),
        "total_response_tokens": sum(
            _int_value(call.get("tokens_response")) for call in calls
        ),
        "empty_responses": sum(1 for call in calls if call.get("response_empty") is True),
        "truncated_calls": sum(1 for call in calls if call.get("done_reason") == "length"),
        "error_calls": sum(1 for call in calls if call.get("error") is True),
        "sections_with_revisions": sorted(revised_sections),
        "largest_prompt": _max_call(calls, "prompt_chars"),
        "largest_response": _max_call(calls, "response_chars"),
        "slowest_call": _max_call(calls, "latency_ms"),
        "by_stage": _stage_rows(by_stage),
    }


def build_usage_details(metadata: dict[str, Any]) -> dict[str, int] | None:
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


def build_prompt_metadata(prompt_object: Any | None) -> dict[str, Any]:
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


def build_langgraph_node_metadata(
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


def merge_state(state: Mapping[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(state)
    if updates is not None:
        merged.update(dict(updates))
    return merged


def summarize_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {key: _summarize_value(value) for key, value in state.items()}


def optional_env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    trimmed = value.strip()
    return trimmed or None


def extract_rubric_scores(raw_text: str) -> dict[str, int]:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        rubrics = _rubrics_from_mapping(payload.get("rubrics"))
        if rubrics:
            return rubrics

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
    rubric_block = raw_text[rubrics_marker + len("RUBRICS:") : block_end]

    rubrics: dict[str, int] = {}
    for metric, raw_score in _RUBRIC_LINE_RE.findall(rubric_block):
        key = metric.strip().lower().replace(" ", "_").replace("-", "_")
        score = int(raw_score)
        if key and 1 <= score <= 5:
            rubrics[key] = score
    return rubrics


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    return text[start : end + 1]


def _rubrics_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    rubrics: dict[str, int] = {}
    for metric, raw_score in value.items():
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            continue
        key = str(metric).strip().lower().replace(" ", "_").replace("-", "_")
        if key and 1 <= raw_score <= 5:
            rubrics[key] = raw_score
    return rubrics


def _add_node_run(bucket: dict[str, Any], run: NodeExecutionRecord) -> None:
    bucket["executions"] += 1
    bucket["total_latency_ms"] += run.latency_ms
    bucket["max_latency_ms"] = max(bucket["max_latency_ms"], run.latency_ms)
    if run.status == "error":
        bucket["errors"] += 1
    if run.next_node:
        bucket["next_nodes"].add(run.next_node)


def _node_summary_row(node_name: str, bucket: dict[str, Any]) -> dict[str, Any]:
    executions = bucket["executions"]
    avg_latency_ms = round(bucket["total_latency_ms"] / executions) if executions else 0
    return {
        "node_name": node_name,
        "executions": executions,
        "avg_latency_ms": avg_latency_ms,
        "max_latency_ms": bucket["max_latency_ms"],
        "errors": bucket["errors"],
        "next_nodes": sorted(bucket["next_nodes"]),
    }


def _add_llm_call(
    by_stage: dict[str, dict[str, Any]],
    revised_sections: set[str],
    call: dict[str, Any],
) -> None:
    stage = _string_value(call.get("stage")) or "unknown"
    bucket = by_stage.setdefault(stage, _stage_bucket(stage))
    bucket["calls"] += 1
    bucket["total_latency_ms"] += _int_value(call.get("latency_ms"))
    bucket["prompt_tokens"] += _int_value(call.get("tokens_prompt"))
    bucket["response_tokens"] += _int_value(call.get("tokens_response"))
    if call.get("response_empty") is True:
        bucket["empty_responses"] += 1
    if call.get("done_reason") == "length":
        bucket["truncated_calls"] += 1
    if call.get("error") is True:
        bucket["errors"] += 1

    revision_round = call.get("revision_round")
    section_name = _string_value(call.get("section_name"))
    if isinstance(revision_round, int) and revision_round > 0 and section_name:
        revised_sections.add(section_name)


def _stage_bucket(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "calls": 0,
        "total_latency_ms": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "empty_responses": 0,
        "truncated_calls": 0,
        "errors": 0,
    }


def _stage_rows(by_stage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage in sorted(by_stage):
        row = by_stage[stage]
        calls_count = row["calls"]
        row["avg_latency_ms"] = (
            round(row["total_latency_ms"] / calls_count) if calls_count else 0
        )
        rows.append(row)
    return rows


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
        return _summarize_mapping(value)
    if isinstance(value, (list, tuple)):
        return _summarize_sequence(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return repr(value)


def _summarize_mapping(value: dict[Any, Any]) -> dict[str, Any]:
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


def _summarize_sequence(value: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    summary = {
        "type": "list",
        "size": len(value),
    }
    if value and all(isinstance(item, str) for item in value[:8]):
        summary["items"] = [_preview_text(item, limit=80) for item in value[:8]]
    return summary


def _preview_text(value: str, *, limit: int = 180) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def _max_call(calls: list[dict[str, Any]], field_name: str) -> dict[str, Any] | None:
    candidates = [call for call in calls if isinstance(call.get(field_name), int)]
    if not candidates:
        return None
    call = max(candidates, key=lambda item: item[field_name])
    return {
        "task_id": call.get("task_id"),
        "stage": call.get("stage"),
        "section_name": call.get("section_name"),
        "revision_round": call.get("revision_round"),
        field_name: call.get(field_name),
        "done_reason": call.get("done_reason"),
        "response_empty": call.get("response_empty"),
    }
