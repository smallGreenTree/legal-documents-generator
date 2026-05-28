"""Business-friendly Prefect views for existing document quality evidence."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

LATENCY_BANDS = (
    ("very fast", 30_000),
    ("fast", 90_000),
    ("slow", 180_000),
)


def build_quality_overview(
    *,
    context: Any,
    doc_id: str,
    quality_report: dict[str, Any],
    rubric_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact overview from existing quality and generation evidence."""
    generation_report = _parse_generation_report(_generation_report_path(context, doc_id))
    sections = quality_report.get("sections", [])
    missing_sections = [
        section["section"]
        for section in sections
        if section.get("verdict") == "missing"
    ]
    sections_with_issues = [
        section["section"]
        for section in sections
        if section.get("issues")
    ]
    revision_count = sum(_int_value(section.get("revision")) for section in sections)
    final_document_exists = _final_document_path(context, doc_id).exists()
    readiness = _readiness(
        final_document_exists=final_document_exists,
        missing_sections=missing_sections,
        quality_score=float(quality_report.get("overall_score") or 0),
        revision_count=revision_count,
        truncated_calls=_int_value(generation_report.get("truncated_calls")),
        rubric_summary=rubric_summary or {},
    )
    stage_rows = generation_report.get("stage_rows", [])
    total_prompt_tokens = sum(_int_value(row.get("prompt_tokens")) for row in stage_rows)
    total_response_tokens = sum(_int_value(row.get("response_tokens")) for row in stage_rows)
    overview = {
        "doc_id": doc_id,
        "readiness": readiness,
        "diagnosis": _diagnosis(
            readiness=readiness,
            final_document_exists=final_document_exists,
            missing_sections=missing_sections,
            sections_with_issues=sections_with_issues,
            revision_count=revision_count,
            generation_report=generation_report,
            quality_report=quality_report,
            rubric_summary=rubric_summary or {},
        ),
        "run_health": {
            "final_document": "rendered" if final_document_exists else "missing",
            "quality_score": quality_report.get("overall_score"),
            "quality_verdict": quality_report.get("verdict"),
            "missing_sections": missing_sections,
            "sections_with_issues": sections_with_issues,
            "revision_count": revision_count,
        },
        "model_workflow": {
            "workflow_mode": generation_report.get("workflow_mode"),
            "langfuse_trace_url": generation_report.get("langfuse_trace_url"),
            "total_llm_calls": generation_report.get("total_llm_calls"),
            "total_latency_ms": generation_report.get("total_latency_ms"),
            "total_latency": format_duration_ms(generation_report.get("total_latency_ms")),
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
            "total_tokens": total_prompt_tokens + total_response_tokens,
            "truncated_calls": generation_report.get("truncated_calls"),
            "empty_responses": generation_report.get("empty_responses"),
            "slowest_stage": _slowest_stage(stage_rows),
            "stage_rows": stage_rows,
            "rubric_summary": rubric_summary or _empty_rubric_summary(),
            "prompt_response_refs": (rubric_summary or {}).get("prompt_response_refs", []),
            "section_rubrics": _section_rubric_rows(
                quality_report,
                rubric_summary or _empty_rubric_summary(),
            ),
        },
        "audit_confidence": _audit_confidence(context, doc_id, generation_report, rubric_summary),
    }
    return overview


def fetch_langfuse_rubric_summary(context: Any, doc_id: str) -> dict[str, Any]:
    """Read rubric scores from existing Langfuse data when credentials are available."""
    generation_report = _parse_generation_report(_generation_report_path(context, doc_id))
    trace_id = generation_report.get("langfuse_trace_id")
    if not trace_id:
        return _empty_rubric_summary("trace id unavailable")

    langfuse_cfg = getattr(context, "langfuse_cfg", None)
    if langfuse_cfg is None or not getattr(langfuse_cfg, "enabled", False):
        return _empty_rubric_summary("Langfuse disabled")

    public_key = os.getenv(getattr(langfuse_cfg, "public_key_env", ""))
    secret_key = os.getenv(getattr(langfuse_cfg, "secret_key_env", ""))
    if not public_key or not secret_key:
        return _empty_rubric_summary("Langfuse credentials unavailable")

    try:
        from langfuse import Langfuse

        trace_url = generation_report.get("langfuse_trace_url")
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=getattr(langfuse_cfg, "host", None),
        )
        scores = _fetch_langfuse_scores(client, trace_id)
        observation_metadata = _fetch_langfuse_observation_metadata(client, trace_id)
    except Exception as exc:
        return _empty_rubric_summary(_clean_langfuse_error(exc))

    return _rubric_summary_from_scores(
        scores,
        observation_metadata,
        trace_url=trace_url,
    )


def format_run_health_markdown(overview: dict[str, Any]) -> str:
    run_health = overview["run_health"]
    lines = [
        "# 00 Run Health",
        "",
        f"**Decision: {overview['readiness']}**",
        "",
        overview["diagnosis"],
        "",
        "| Signal | Value |",
        "| --- | --- |",
        f"| Document ID | `{overview['doc_id']}` |",
        f"| Final document | {run_health['final_document']} |",
        (
            "| Quality score | "
            f"{_score_display(run_health.get('quality_score'))} "
            f"({run_health.get('quality_verdict') or 'n/a'}) |"
        ),
        f"| Missing sections | {_join_or_none(run_health['missing_sections'])} |",
        f"| Sections with issues | {_join_or_none(run_health['sections_with_issues'])} |",
        f"| Revision rounds | {_int_display(run_health['revision_count'])} |",
    ]
    return "\n".join(lines).rstrip() + "\n"


def format_model_workflow_markdown(overview: dict[str, Any]) -> str:
    workflow = overview["model_workflow"]
    rubric = workflow["rubric_summary"]
    slowest = workflow.get("slowest_stage") or {}
    lines = [
        "# 01 Model Workflow",
        "",
        "Rubric scale: `1 = bad`, `5 = good`.",
        (
            "`Total latency` is the sum of all LLM calls in that component. "
            "`Avg latency` is total latency divided by calls. These are model-call "
            "timings, not necessarily wall-clock runtime when work runs in parallel."
        ),
        "",
        "| Signal | Value |",
        "| --- | --- |",
        f"| Workflow mode | {workflow.get('workflow_mode') or 'n/a'} |",
        f"| Langfuse trace | {_link_or_na(workflow.get('langfuse_trace_url'), 'open trace')} |",
        f"| Total LLM calls | {_int_display(workflow.get('total_llm_calls'))} |",
        f"| Total model time for document | {workflow.get('total_latency') or 'n/a'} |",
        f"| Total tokens | {_int_display(workflow.get('total_tokens'))} |",
        f"| Truncated calls | {_int_display(workflow.get('truncated_calls'))} |",
        f"| Empty responses | {_int_display(workflow.get('empty_responses'))} |",
        (
            "| Slowest stage | "
            f"{slowest.get('stage', 'n/a')} "
            f"({slowest.get('total_latency', 'n/a')}) |"
        ),
        f"| Overall rubric | {_rubric_display(rubric.get('overall'))} |",
        f"| Lowest rubric dimension | {_metric_display(rubric.get('lowest_metric'))} |",
        f"| Rubric data status | {rubric.get('status') or 'n/a'} |",
        "",
        (
            "| Component | Calls | Total latency | Avg latency | Band | Prompt tokens | "
            "Response tokens |"
        ),
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in workflow.get("stage_rows", []):
        lines.append(
            "| "
            f"{row.get('stage') or 'unknown'} | "
            f"{_int_display(row.get('calls'))} | "
            f"{format_duration_ms(row.get('total_latency_ms'))} | "
            f"{format_duration_ms(row.get('avg_latency_ms'))} | "
            f"{latency_band(row.get('avg_latency_ms'))} | "
            f"{_int_display(row.get('prompt_tokens'))} | "
            f"{_int_display(row.get('response_tokens'))} |"
        )
    lines.extend(
        [
            "",
            "## Section Rubrics",
            "",
            _section_rubric_note(workflow),
            "",
            (
                "| Section | Quality | Rubric avg | Grounding | Completeness | "
                "Legal style | Chronology | Revision | Langfuse | Main issue |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in workflow.get("section_rubrics", []):
        lines.append(
            "| "
            f"{row['section']} | "
            f"{_score_display(row.get('quality_score'))} | "
            f"{_rubric_number_display(row.get('overall'))} | "
            f"{_rubric_number_display(row.get('grounding'))} | "
            f"{_rubric_number_display(row.get('completeness'))} | "
            f"{_rubric_number_display(row.get('legal_style'))} | "
            f"{_rubric_number_display(row.get('chronology'))} | "
            f"{_revision_display(row.get('revision'))} | "
            f"{_link_or_na(row.get('langfuse_url'), 'trace')} | "
            f"{row.get('main_issue') or 'none'} |"
        )
    if workflow.get("prompt_response_refs"):
        lines.extend(
            [
                "",
                "## Prompt/Response References",
                "",
                (
                    "These links point to the Langfuse observations that contain the "
                    "actual prompts and responses for each final section revision."
                ),
                "",
                "| Section | Text generation prompts/responses | Critic rubric prompt/response |",
                "| --- | --- | --- |",
            ]
        )
        for row in workflow.get("prompt_response_refs", []):
            lines.append(
                "| "
                f"{row.get('section') or 'unknown'} | "
                f"{row.get('text_links') or 'n/a'} | "
                f"{row.get('critic_link') or 'n/a'} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def format_audit_confidence_markdown(overview: dict[str, Any]) -> str:
    lines = [
        "# 02 Audit Confidence",
        "",
        "| Evidence | Status | Meaning |",
        "| --- | --- | --- |",
    ]
    for item in overview["audit_confidence"]:
        lines.append(
            "| "
            f"{item['evidence']} | "
            f"{item['status']} | "
            f"{item['meaning']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def format_duration_ms(milliseconds: Any) -> str:
    total_seconds = round(_int_value(milliseconds) / 1000)
    if total_seconds <= 0:
        return "0s"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m" if not seconds else f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m" if not seconds else f"{minutes}m {seconds}s"
    return f"{seconds}s"


def latency_band(milliseconds: Any) -> str:
    value = _int_value(milliseconds)
    for label, upper_bound in LATENCY_BANDS:
        if value <= upper_bound:
            return label
    return "very slow"


def _parse_generation_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "stage_rows": [],
            "workflow_mode": None,
            "total_llm_calls": 0,
            "total_latency_ms": 0,
            "empty_responses": 0,
            "truncated_calls": 0,
            "langfuse_trace_id": None,
            "langfuse_trace_url": None,
        }
    text = path.read_text(encoding="utf-8")
    return {
        "exists": True,
        "workflow_mode": _match_text(text, r"- Workflow mode:\s*(.+)"),
        "total_llm_calls": _match_int(text, r"- Total LLM calls:\s*(\d+)"),
        "total_latency_ms": _match_int(text, r"- Total LLM latency ms:\s*(\d+)"),
        "empty_responses": _match_int(text, r"- Empty LLM responses:\s*(\d+)"),
        "truncated_calls": _match_int(text, r"- Truncated LLM calls:\s*(\d+)"),
        "langfuse_trace_id": _match_text(text, r"- Langfuse trace id:\s*(.+)"),
        "langfuse_trace_url": _match_text(text, r"- Langfuse trace url:\s*(.+)"),
        "stage_rows": [_stage_from_row(row) for row in _parse_stage_table(text.splitlines())],
    }


def _parse_stage_table(lines: list[str]) -> list[dict[str, str]]:
    for index, line in enumerate(lines):
        if not line.startswith("| Stage | Calls | Total ms | Avg ms |"):
            continue
        rows: list[dict[str, str]] = []
        headers = _split_table_row(line)
        for row_line in lines[index + 2 :]:
            if not row_line.startswith("|"):
                break
            cells = _split_table_row(row_line)
            if len(cells) != len(headers):
                break
            rows.append(dict(zip(headers, cells, strict=True)))
        return rows
    return []


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _stage_from_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "stage": row.get("Stage") or "unknown",
        "calls": _int_value(row.get("Calls")),
        "total_latency_ms": _int_value(row.get("Total ms")),
        "avg_latency_ms": _int_value(row.get("Avg ms")),
        "prompt_tokens": _int_value(row.get("Prompt Tokens")),
        "response_tokens": _int_value(row.get("Response Tokens")),
        "empty_responses": _int_value(row.get("Empty")),
        "truncated_calls": _int_value(row.get("Truncated")),
        "errors": _int_value(row.get("Errors")),
    }


def _fetch_langfuse_observation_metadata(client: Any, trace_id: str) -> dict[str, dict[str, Any]]:
    try:
        return _fetch_langfuse_observation_metadata_v1(client, trace_id)
    except Exception as v1_exc:
        try:
            return _fetch_langfuse_observation_metadata_v2(client, trace_id)
        except Exception:
            raise v1_exc


def _fetch_langfuse_observation_metadata_v1(
    client: Any,
    trace_id: str,
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    page = 1
    total_pages = 1
    while page <= total_pages:
        response = client.api.legacy.observations_v1.get_many(
            trace_id=trace_id,
            limit=100,
            page=page,
        )
        for observation in getattr(response, "data", []) or []:
            _store_observation_metadata(observations, observation)
        meta = getattr(response, "meta", None)
        total_pages = _int_value(getattr(meta, "total_pages", 1)) or 1
        page += 1
    return observations


def _fetch_langfuse_observation_metadata_v2(
    client: Any,
    trace_id: str,
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    cursor = None
    for _page in range(10):
        kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "limit": 100,
        }
        if cursor:
            kwargs["cursor"] = cursor
        response = client.api.observations.get_many(**kwargs)
        for observation in getattr(response, "data", []) or []:
            _store_observation_metadata(observations, observation)
        meta = getattr(response, "meta", None)
        cursor = (
            getattr(meta, "next_cursor", None)
            or getattr(meta, "nextCursor", None)
            or getattr(meta, "next_page", None)
        )
        if not cursor:
            break
    return observations


def _store_observation_metadata(
    observations: dict[str, dict[str, Any]],
    observation: Any,
) -> None:
    observation_id = str(getattr(observation, "id", "") or "")
    if not observation_id:
        return
    metadata = _as_mapping(getattr(observation, "metadata", None))
    task_id = metadata.get("task_id") or getattr(observation, "name", None)
    observations[observation_id] = {
        "observation_id": observation_id,
        "section_name": metadata.get("section_name"),
        "revision_round": metadata.get("revision_round"),
        "stage": metadata.get("stage"),
        "task_id": task_id,
    }


def _fetch_langfuse_scores(client: Any, trace_id: str) -> list[Any]:
    scores: list[Any] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        response = client.api.scores.get_many(
            trace_id=trace_id,
            limit=100,
            page=page,
        )
        scores.extend(getattr(response, "data", []) or [])
        meta = getattr(response, "meta", None)
        total_pages = _int_value(getattr(meta, "total_pages", 1)) or 1
        page += 1
    return scores


def _clean_langfuse_error(exc: Exception) -> str:
    message = str(exc)
    if "limit" in message and "100" in message:
        return "Langfuse read failed: API pagination limit exceeded"
    status_match = re.search(r"status_code:\s*(\d+)", message)
    body_match = re.search(r"body:\s*(\{.*\})", message)
    if status_match and body_match:
        return f"Langfuse read failed: HTTP {status_match.group(1)} {body_match.group(1)}"
    if len(message) > 240:
        message = message[:237].rstrip() + "..."
    return f"Langfuse read failed: {message}"


def _rubric_summary_from_scores(
    scores: list[Any],
    observation_metadata: dict[str, dict[str, Any]] | None = None,
    trace_url: str | None = None,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    overall_values: list[float] = []
    section_calls: dict[str, dict[str, dict[str, Any]]] = {}
    for score in scores:
        name = str(getattr(score, "name", "") or "")
        if not name.startswith("rubric."):
            continue
        value = getattr(score, "value", None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        metric = name.removeprefix("rubric.")
        score_value = float(value)
        if metric == "overall":
            overall_values.append(score_value)
        else:
            metric_values.setdefault(metric, []).append(score_value)
        section_name, revision_round, call_id = _score_section_context(
            score,
            observation_metadata or {},
        )
        if section_name:
            section_bucket = section_calls.setdefault(section_name, {})
            call_bucket = section_bucket.setdefault(
                call_id,
                {
                    "revision_round": revision_round,
                    "metrics": {},
                    "langfuse_url": _langfuse_observation_url(trace_url, call_id),
                },
            )
            call_bucket["metrics"].setdefault(metric, []).append(score_value)

    if not overall_values and metric_values:
        overall_values = [
            score
            for values in metric_values.values()
            for score in values
        ]
    metric_averages = {
        metric: round(sum(values) / len(values), 2)
        for metric, values in metric_values.items()
    }
    lowest_metric = min(metric_averages.items(), key=lambda item: item[1], default=None)
    if not overall_values and not metric_averages:
        return _empty_rubric_summary("no rubric scores found")
    return {
        "overall": round(sum(overall_values) / len(overall_values), 2)
        if overall_values
        else None,
        "lowest_metric": {
            "metric": lowest_metric[0],
            "score": lowest_metric[1],
        }
        if lowest_metric
        else None,
        "sections": _section_rubric_summary_from_calls(section_calls),
        "prompt_response_refs": _prompt_response_reference_rows(
            observation_metadata or {},
            trace_url,
        ),
        "trace_url": trace_url,
        "status": "available",
    }


def _score_section_context(
    score: Any,
    observation_metadata: dict[str, dict[str, Any]],
) -> tuple[str | None, int | None, str]:
    observation_id = str(getattr(score, "observation_id", "") or "")
    metadata = observation_metadata.get(observation_id, {})
    score_metadata = _as_mapping(getattr(score, "metadata", None))
    section_name = (
        metadata.get("section_name")
        or score_metadata.get("section_name")
        or score_metadata.get("section")
    )
    revision = metadata.get("revision_round", score_metadata.get("revision_round"))
    call_id = observation_id or str(getattr(score, "id", "") or len(observation_metadata))
    return (
        str(section_name) if section_name else None,
        _optional_int(revision),
        call_id,
    )


def _section_rubric_summary_from_calls(
    section_calls: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section_name, calls in sorted(section_calls.items()):
        selected_calls = _latest_rubric_calls(calls)
        metric_values: dict[str, list[float]] = {}
        selected_revisions = [
            call.get("revision_round")
            for call in selected_calls
            if call.get("revision_round") is not None
        ]
        for call in selected_calls:
            for metric, values in call.get("metrics", {}).items():
                metric_values.setdefault(metric, []).extend(values)
        metric_averages = {
            metric: round(sum(values) / len(values), 2)
            for metric, values in metric_values.items()
            if values
        }
        overall = metric_averages.get("overall")
        if overall is None:
            dimensional_values = [
                value
                for metric, value in metric_averages.items()
                if metric != "overall"
            ]
            overall = (
                round(sum(dimensional_values) / len(dimensional_values), 2)
                if dimensional_values
                else None
            )
        sections.append(
            {
                "section": section_name,
                "overall": overall,
                "revision": max(selected_revisions) if selected_revisions else None,
                "calls": len(selected_calls),
                "langfuse_url": _first_value(selected_calls, "langfuse_url"),
                **metric_averages,
            }
        )
    return sections


def _prompt_response_reference_rows(
    observation_metadata: dict[str, dict[str, Any]],
    trace_url: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_section: dict[str, list[dict[str, Any]]] = {}
    for observation in observation_metadata.values():
        section_name = observation.get("section_name")
        if not section_name:
            continue
        by_section.setdefault(str(section_name), []).append(observation)

    for section_name, observations in sorted(by_section.items()):
        text_observations = _latest_text_observations(observations)
        critic_observations = _latest_stage_observations(observations, "critic")
        rows.append(
            {
                "section": section_name,
                "text_url": _first_observation_url(text_observations, trace_url),
                "critic_url": _first_observation_url(critic_observations, trace_url),
                "text_links": _observation_links(text_observations, trace_url),
                "critic_link": _observation_links(critic_observations[:1], trace_url),
            }
        )
    return rows


def _latest_text_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    polisher_observations = _latest_stage_observations(observations, "polisher")
    if polisher_observations:
        return polisher_observations
    return _latest_stage_observations(observations, "writer")


def _latest_stage_observations(
    observations: list[dict[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    matching = [
        observation
        for observation in observations
        if observation.get("stage") == stage
    ]
    revisions = [
        _optional_int(observation.get("revision_round"))
        for observation in matching
        if _optional_int(observation.get("revision_round")) is not None
    ]
    if revisions:
        latest_revision = max(revisions)
        matching = [
            observation
            for observation in matching
            if _optional_int(observation.get("revision_round")) == latest_revision
        ]
    return sorted(matching, key=_observation_sort_key)


def _observation_sort_key(observation: dict[str, Any]) -> tuple[int, str]:
    task_id = str(observation.get("task_id") or "")
    chunk_match = re.search(r"_chunk_(\d+)$", task_id)
    chunk = int(chunk_match.group(1)) if chunk_match else 0
    return (chunk, task_id)


def _observation_links(
    observations: list[dict[str, Any]],
    trace_url: str | None,
) -> str:
    links: list[str] = []
    for observation in observations:
        observation_id = str(observation.get("observation_id") or "")
        url = _langfuse_observation_url(trace_url, observation_id)
        if not url:
            continue
        links.append(_link_or_na(url, _observation_label(observation)))
    return ", ".join(links)


def _observation_label(observation: dict[str, Any]) -> str:
    task_id = str(observation.get("task_id") or "")
    revision = _optional_int(observation.get("revision_round"))
    chunk_match = re.search(r"_chunk_(\d+)$", task_id)
    if chunk_match:
        chunk_label = f"chunk {chunk_match.group(1)}"
        return f"r{revision} {chunk_label}" if revision is not None else chunk_label
    stage = str(observation.get("stage") or "observation")
    return f"{stage} r{revision}" if revision is not None else stage


def _first_observation_url(
    observations: list[dict[str, Any]],
    trace_url: str | None,
) -> str | None:
    for observation in observations:
        url = _langfuse_observation_url(
            trace_url,
            str(observation.get("observation_id") or ""),
        )
        if url:
            return url
    return None


def _latest_rubric_calls(calls: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(calls.values())
    revisions = [
        call.get("revision_round")
        for call in values
        if call.get("revision_round") is not None
    ]
    if not revisions:
        return values
    latest_revision = max(revisions)
    return [
        call
        for call in values
        if call.get("revision_round") == latest_revision
    ]


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value:
            return value
    return None


def _langfuse_observation_url(trace_url: str | None, observation_id: str) -> str | None:
    if not trace_url or not observation_id:
        return trace_url
    separator = "&" if "?" in trace_url else "?"
    return f"{trace_url}{separator}observation={observation_id}"


def _section_rubric_rows(
    quality_report: dict[str, Any],
    rubric_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_url = rubric_summary.get("trace_url")
    rubric_by_section = {
        str(section.get("section")): section
        for section in rubric_summary.get("sections", [])
        if section.get("section")
    }
    rows = []
    for section in quality_report.get("sections", []):
        section_name = str(section.get("section") or "unknown")
        rubric = rubric_by_section.get(section_name, {})
        issues = section.get("issues") or []
        rows.append(
            {
                "section": section_name,
                "quality_score": section.get("score"),
                "overall": rubric.get("overall"),
                "grounding": rubric.get("grounding"),
                "completeness": rubric.get("completeness"),
                "legal_style": rubric.get("legal_style"),
                "chronology": rubric.get("chronology"),
                "revision": (
                    rubric.get("revision")
                    if rubric.get("revision") is not None
                    else section.get("revision")
                ),
                "langfuse_url": rubric.get("langfuse_url") or trace_url,
                "main_issue": issues[0] if issues else "none",
            }
        )
    return rows


def _audit_confidence(
    context: Any,
    doc_id: str,
    generation_report: dict[str, Any],
    rubric_summary: dict[str, Any] | None,
) -> list[dict[str, str]]:
    memory_dir = Path(context.memory_dir) / f"case_{doc_id}"
    schema_path = Path(context.schema_dir) / f"{doc_id}.json"
    rubric_status = "available" if (rubric_summary or {}).get("overall") is not None else "missing"
    return [
        _evidence("Case memory", memory_dir / "CASE_MEMORY.md", "Allowed facts and entities"),
        _evidence("Run history", memory_dir / "RUN_HISTORY.md", "Workflow history"),
        _evidence(
            "Generation report",
            _generation_report_path(context, doc_id),
            "Model workflow evidence",
        ),
        _evidence(
            "Final document",
            _final_document_path(context, doc_id),
            "Rendered product output",
        ),
        _evidence("Schema", schema_path, "Relationship graph"),
        {
            "evidence": "Langfuse trace",
            "status": "available" if generation_report.get("langfuse_trace_id") else "missing",
            "meaning": "Prompts, responses, node metrics",
        },
        {
            "evidence": "Rubric scores",
            "status": rubric_status,
            "meaning": "Critic scoring signal",
        },
    ]


def _evidence(label: str, path: Path, meaning: str) -> dict[str, str]:
    return {
        "evidence": label,
        "status": "available" if path.exists() else "missing",
        "meaning": meaning,
    }


def _readiness(
    *,
    final_document_exists: bool,
    missing_sections: list[str],
    quality_score: float,
    revision_count: int,
    truncated_calls: int,
    rubric_summary: dict[str, Any],
) -> str:
    rubric_score = rubric_summary.get("overall")
    if not final_document_exists or missing_sections or quality_score < 50:
        return "Not Ready"
    if quality_score < 85 or revision_count > 0 or truncated_calls > 0:
        return "Needs Review"
    if isinstance(rubric_score, (int, float)) and rubric_score < 3.5:
        return "Needs Review"
    return "Ready"


def _diagnosis(
    *,
    readiness: str,
    final_document_exists: bool,
    missing_sections: list[str],
    sections_with_issues: list[str],
    revision_count: int,
    generation_report: dict[str, Any],
    quality_report: dict[str, Any],
    rubric_summary: dict[str, Any],
) -> str:
    if not final_document_exists:
        return "The final document is missing, so product completion is the primary issue."
    if missing_sections:
        return (
            "The document is incomplete because sections are missing: "
            f"{', '.join(missing_sections)}."
        )

    parts = []
    if readiness == "Ready":
        parts.append("The document rendered and the main quality signals are acceptable.")
    else:
        parts.append("The document rendered, but it should remain in review.")
    if revision_count:
        parts.append(f"The model workflow required {revision_count} revision round(s).")
    if sections_with_issues:
        parts.append(f"{len(sections_with_issues)} section(s) still have validator issues.")
    if _int_value(generation_report.get("truncated_calls")):
        parts.append(f"{generation_report['truncated_calls']} LLM call(s) were truncated.")
    if rubric_summary.get("overall") is not None:
        parts.append(f"Overall rubric is {_rubric_display(rubric_summary['overall'])}.")
    if not parts:
        parts.append(f"Quality verdict is {quality_report.get('verdict', 'n/a')}.")
    return " ".join(parts)


def _slowest_stage(stage_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not stage_rows:
        return None
    row = max(stage_rows, key=lambda item: _int_value(item.get("total_latency_ms")))
    return {
        "stage": row.get("stage"),
        "total_latency_ms": row.get("total_latency_ms"),
        "total_latency": format_duration_ms(row.get("total_latency_ms")),
    }


def _empty_rubric_summary(reason: str = "unavailable") -> dict[str, Any]:
    return {
        "overall": None,
        "lowest_metric": None,
        "sections": [],
        "prompt_response_refs": [],
        "trace_url": None,
        "status": reason,
    }


def _generation_report_path(context: Any, doc_id: str) -> Path:
    return Path(context.output_dir) / doc_id / "generation_report.md"


def _final_document_path(context: Any, doc_id: str) -> Path:
    return Path(context.output_dir) / doc_id / f"{doc_id}.txt"


def _match_text(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value == "n/a" else value


def _match_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return int(match.group(1)) if match else 0


def _join_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _score_display(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{round(float(value), 2):g}"
    return "n/a"


def _rubric_display(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{round(float(value), 2):g} / 5"
    return "n/a"


def _metric_display(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('metric', 'n/a')} ({_rubric_display(value.get('score'))})"
    return "n/a"


def _section_rubric_note(workflow: dict[str, Any]) -> str:
    rubric = workflow.get("rubric_summary") or {}
    if rubric.get("sections"):
        return (
            "Rows use the latest available critic rubric scores for each section. "
            "The Langfuse link opens the matching trace observation where available."
        )
    status = rubric.get("status") or "unavailable"
    return (
        "`n/a` means section-level critic rubric scores were not available to this "
        f"quality run. Rubric data status: `{status}`. The deterministic quality "
        "score and validator issue are still shown."
    )


def _link_or_na(url: Any, label: str) -> str:
    if isinstance(url, str) and url.strip():
        return f"[{label}]({url.strip()})"
    return "n/a"


def _rubric_number_display(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{round(float(value), 2):g}"
    return "n/a"


def _revision_display(value: Any) -> str:
    revision = _optional_int(value)
    return str(revision) if revision is not None else "-"


def _int_display(value: Any) -> str:
    return f"{_int_value(value):,}"


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned in {"", "-"}:
            return 0
        try:
            return int(cleaned)
        except ValueError:
            return 0
    return 0


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}
