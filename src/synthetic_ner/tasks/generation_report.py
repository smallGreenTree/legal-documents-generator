"""Generation report formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.tracer import TraceStore


def write_generation_report(
    *,
    context,
    doc_id: str,
    memory_path: Path,
    document_plan: str,
    section_contracts: dict[str, str],
    section_plans: dict[str, str],
    section_reviews: dict[str, list[str]],
    revision_counts: dict[str, int],
    trace_store: TraceStore,
) -> Path:
    trace_info = trace_store.get_trace_info()
    node_summary = trace_store.get_langgraph_node_summary()
    llm_summary = trace_store.get_llm_run_summary()
    llm_calls = trace_store.get_llm_call_records()
    report_path = context.output_dir / doc_id / "generation_report.md"
    lines = _report_header(
        doc_id=doc_id,
        context=context,
        memory_path=memory_path,
        trace_info=trace_info,
        llm_summary=llm_summary,
        document_plan=document_plan,
    )
    lines.extend(
        _format_section_results(
            section_contracts=section_contracts,
            section_plans=section_plans,
            section_reviews=section_reviews,
            revision_counts=revision_counts,
        )
    )
    if node_summary:
        lines.extend(_format_node_analytics(node_summary))
    if llm_calls:
        lines.extend(_format_llm_analytics(llm_summary, llm_calls))
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _report_header(
    *,
    doc_id: str,
    context,
    memory_path: Path,
    trace_info,
    llm_summary: dict[str, Any],
    document_plan: str,
) -> list[str]:
    return [
        f"# Generation Report: {doc_id}",
        "",
        f"- Workflow mode: {context.workflow_cfg.mode}",
        f"- Max revision rounds: {context.workflow_cfg.max_revisions}",
        f"- Memory file: {memory_path}",
        f"- Langfuse enabled: {str(trace_info.enabled).lower()}",
        f"- Langfuse trace id: {trace_info.trace_id or 'n/a'}",
        f"- Langfuse trace url: {trace_info.trace_url or 'n/a'}",
        f"- Total LLM calls: {llm_summary['total_llm_calls']}",
        f"- Total LLM latency ms: {llm_summary['total_latency_ms']}",
        f"- Empty LLM responses: {llm_summary['empty_responses']}",
        f"- Truncated LLM calls: {llm_summary['truncated_calls']}",
        "",
        "## Document Plan",
        document_plan or "- none",
        "",
        "## Section Results",
    ]


def _format_section_results(
    *,
    section_contracts: dict[str, str],
    section_plans: dict[str, str],
    section_reviews: dict[str, list[str]],
    revision_counts: dict[str, int],
) -> list[str]:
    lines = []
    for section_name in section_plans:
        issues = section_reviews.get(section_name, [])
        lines.extend(
            [
                f"### {section_name}",
                "",
                "Contract:",
                section_contracts.get(section_name, "- none"),
                "",
                "Plan:",
                section_plans[section_name],
                "",
                "Issues:",
                *([f"- {issue}" for issue in issues] or ["- none"]),
                "",
                f"Revisions: {revision_counts.get(section_name, 0)}",
                "",
            ]
        )
    return lines


def _format_node_analytics(node_summary: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## LangGraph Node Analytics",
        "",
        "Each node below also appears as its own child span in Langfuse.",
        "",
        "| Node | Executions | Avg ms | Max ms | Errors | Next Nodes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in node_summary:
        next_nodes = ", ".join(row["next_nodes"]) if row["next_nodes"] else "-"
        lines.append(
            "| "
            f"{row['node_name']} | "
            f"{row['executions']} | "
            f"{row['avg_latency_ms']} | "
            f"{row['max_latency_ms']} | "
            f"{row['errors']} | "
            f"{next_nodes} |"
        )
    lines.append("")
    return lines


def _format_llm_analytics(
    llm_summary: dict[str, Any],
    llm_calls: list[dict[str, Any]],
) -> list[str]:
    lines = _format_stage_totals(llm_summary)
    lines.extend(_format_bottlenecks(llm_summary))
    lines.extend(_format_llm_calls(llm_calls))
    return lines


def _format_stage_totals(llm_summary: dict[str, Any]) -> list[str]:
    lines = [
        "## LLM Run Analytics",
        "",
        "### Stage Totals",
        "",
        (
            "| Stage | Calls | Total ms | Avg ms | Prompt Tokens | Response Tokens | "
            "Empty | Truncated | Errors |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in llm_summary["by_stage"]:
        lines.append(
            "| "
            f"{row['stage']} | "
            f"{row['calls']} | "
            f"{row['total_latency_ms']} | "
            f"{row['avg_latency_ms']} | "
            f"{row['prompt_tokens']} | "
            f"{row['response_tokens']} | "
            f"{row['empty_responses']} | "
            f"{row['truncated_calls']} | "
            f"{row['errors']} |"
        )
    return lines


def _format_bottlenecks(llm_summary: dict[str, Any]) -> list[str]:
    revised_sections = llm_summary["sections_with_revisions"]
    return [
        "",
        "### Bottleneck Candidates",
        "",
        f"- Slowest call: {_format_call_summary(llm_summary['slowest_call'], 'latency_ms')}",
        (
            "- Largest prompt: "
            f"{_format_call_summary(llm_summary['largest_prompt'], 'prompt_chars')}"
        ),
        (
            "- Largest response: "
            f"{_format_call_summary(llm_summary['largest_response'], 'response_chars')}"
        ),
        "- Sections with revisions: "
        + (", ".join(revised_sections) if revised_sections else "none"),
        "",
    ]


def _format_llm_calls(llm_calls: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### LLM Calls",
        "",
        (
            "| Task | Stage | Section | Rev | Prompt chars | Response chars | "
            "Prompt tokens | Response tokens | Budget | ms | Done | Empty |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for call in llm_calls:
        lines.append(_format_llm_call_row(call))
    lines.append("")
    return lines


def _format_llm_call_row(call: dict[str, Any]) -> str:
    return (
        "| "
        f"{call.get('task_id') or '-'} | "
        f"{call.get('stage') or '-'} | "
        f"{call.get('section_name') or '-'} | "
        f"{_format_optional_int(call.get('revision_round'))} | "
        f"{_format_optional_int(call.get('prompt_chars'))} | "
        f"{_format_optional_int(call.get('response_chars'))} | "
        f"{_format_optional_int(call.get('tokens_prompt'))} | "
        f"{_format_optional_int(call.get('tokens_response'))} | "
        f"{_format_optional_int(call.get('output_budget'))} | "
        f"{_format_optional_int(call.get('latency_ms'))} | "
        f"{call.get('done_reason') or '-'} | "
        f"{str(call.get('response_empty')).lower()} |"
    )


def _format_optional_int(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    return str(value) if isinstance(value, int) else "-"


def _format_call_summary(call: dict[str, Any] | None, field_name: str) -> str:
    if not call:
        return "n/a"
    task_id = call.get("task_id") or "-"
    stage = call.get("stage") or "-"
    section = call.get("section_name") or "-"
    value = call.get(field_name)
    return f"{task_id} ({stage}/{section}, {field_name}={value})"
