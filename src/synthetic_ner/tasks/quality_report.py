"""Quality report generation from existing document artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.validators import validate_section_text
from src.synthetic_ner.utils import load_config

_REVISION_RE = re.compile(r"^r(\d+)$")
DEFAULT_QUALITY_CONFIG_PATH = "config_quality.yaml"
DEFAULT_SCORING_CONFIG = {
    "validator_issue_penalty": 12,
    "validator_issue_penalty_cap": 60,
    "revision_penalty": 6,
    "revision_penalty_cap": 18,
    "short_section_penalty": 15,
    "short_section_word_threshold": 80,
}


def build_quality_report(
    context: Any,
    doc_id: str,
    scoring_config: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic quality report for one generated document."""
    scoring = scoring_config or DEFAULT_SCORING_CONFIG
    memory_path = context.memory_dir / f"case_{doc_id}" / "CASE_MEMORY.md"
    memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

    sections = []
    for section_name, word_target in context.section_word_targets.items():
        section = _score_section(
            context=context,
            doc_id=doc_id,
            section_name=section_name,
            word_target=word_target,
            memory_text=memory_text,
            scoring_config=scoring,
        )
        sections.append(section)

    section_scores = [section["score"] for section in sections]
    overall_score = round(sum(section_scores) / len(section_scores), 2) if section_scores else 0
    return {
        "doc_id": doc_id,
        "overall_score": overall_score,
        "verdict": _verdict(overall_score),
        "paths": {
            "memory": str(memory_path),
            "partial_sections": str(context.output_dir / "_partial" / doc_id / "sections"),
            "final_output": str(context.output_dir / doc_id),
        },
        "sections": sections,
        "top_failures": _top_failures(sections),
        "scoring_config": scoring,
    }


def load_quality_scoring_config(path: Path | str) -> dict[str, int]:
    """Load deterministic quality scoring weights from a dedicated config file."""
    raw = load_config(path) or {}
    scoring = raw.get("quality_scoring")
    if not isinstance(scoring, dict):
        raise ValueError("config_quality.yaml must contain a quality_scoring mapping")
    return {
        key: _positive_int(scoring.get(key), key)
        for key in DEFAULT_SCORING_CONFIG
    }


def _score_section(
    *,
    context: Any,
    doc_id: str,
    section_name: str,
    word_target: int,
    memory_text: str,
    scoring_config: dict[str, int],
) -> dict[str, Any]:
    latest = _latest_section_revision(context, doc_id, section_name)
    if latest is None:
        return {
            "section": section_name,
            "score": 0,
            "verdict": "missing",
            "revision": None,
            "word_count": 0,
            "expected_words": word_target,
            "issues": ["Section artifact is missing."],
            "score_breakdown": {
                "base": 100,
                "issue_count": 1,
                "issue_penalty": 100,
                "revision_penalty": 0,
                "short_section_penalty": 0,
                "score": 0,
            },
            "path": None,
        }

    revision, section_path = latest
    section_text = section_path.read_text(encoding="utf-8")
    issues = validate_section_text(
        section_name=section_name,
        section_text=section_text,
        memory_text=memory_text,
        word_target=word_target,
    )
    score_breakdown = _section_score_breakdown(
        issues=issues,
        revision=revision,
        word_count=len(section_text.split()),
        scoring_config=scoring_config,
    )
    return {
        "section": section_name,
        "score": score_breakdown["score"],
        "verdict": _verdict(score_breakdown["score"]),
        "revision": revision,
        "word_count": len(section_text.split()),
        "expected_words": word_target,
        "issues": issues,
        "score_breakdown": score_breakdown,
        "path": str(section_path),
    }


def _latest_section_revision(
    context: Any,
    doc_id: str,
    section_name: str,
) -> tuple[int, Path] | None:
    section_dir = context.output_dir / "_partial" / doc_id / "sections" / section_name
    if not section_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for revision_dir in section_dir.iterdir():
        if not revision_dir.is_dir():
            continue
        match = _REVISION_RE.match(revision_dir.name)
        if not match:
            continue
        combined_path = revision_dir / "combined.txt"
        if combined_path.exists():
            candidates.append((int(match.group(1)), combined_path))
    return max(candidates, default=None, key=lambda item: item[0])


def _section_score(
    *,
    issues: list[str],
    revision: int,
    word_count: int,
    scoring_config: dict[str, int],
) -> int:
    return _section_score_breakdown(
        issues=issues,
        revision=revision,
        word_count=word_count,
        scoring_config=scoring_config,
    )["score"]


def _section_score_breakdown(
    *,
    issues: list[str],
    revision: int,
    word_count: int,
    scoring_config: dict[str, int],
) -> dict[str, int]:
    issue_penalty = min(
        scoring_config["validator_issue_penalty_cap"],
        len(issues) * scoring_config["validator_issue_penalty"],
    )
    revision_penalty = min(
        scoring_config["revision_penalty_cap"],
        revision * scoring_config["revision_penalty"],
    )
    short_section_penalty = (
        scoring_config["short_section_penalty"]
        if word_count < scoring_config["short_section_word_threshold"]
        else 0
    )
    score = 100
    score -= issue_penalty
    score -= revision_penalty
    score -= short_section_penalty
    return {
        "base": 100,
        "issue_count": len(issues),
        "issue_penalty": issue_penalty,
        "revision_penalty": revision_penalty,
        "short_section_penalty": short_section_penalty,
        "score": max(0, score),
    }


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"quality_scoring.{key} must be a non-negative integer")
    return value


def _verdict(score: float) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "acceptable"
    if score >= 50:
        return "risky"
    return "bad"


def _top_failures(sections: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for section in sections:
        for issue in section["issues"]:
            counts[issue] = counts.get(issue, 0) + 1
    return [
        issue
        for issue, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


def format_markdown_report(report: dict[str, Any]) -> str:
    include_langfuse = any(
        section.get("langfuse_url") for section in report["sections"]
    )
    lines = [
        f"# Quality Report: {report['doc_id']}",
        "",
        f"- Overall score: {report['overall_score']}",
        f"- Verdict: {report['verdict']}",
        "",
        "## Section Scores",
        "",
    ]
    if include_langfuse:
        lines.extend(
            [
                (
                    "| Section | Score | Verdict | Revision | Words | "
                    "Expected words | Issues | Langfuse |"
                ),
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Section | Score | Verdict | Revision | Words | Expected words | Issues |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
    for section in report["sections"]:
        revision = section["revision"] if section["revision"] is not None else "-"
        row = (
            "| "
            f"{section['section']} | "
            f"{section['score']} | "
            f"{section['verdict']} | "
            f"{revision} | "
            f"{section['word_count']} | "
            f"{section.get('expected_words', 'n/a')} | "
            f"{len(section['issues'])} |"
        )
        if include_langfuse:
            row += f" {_quality_link_or_na(section.get('langfuse_url'))} |"
        lines.append(row)

    lines.extend(_score_explanation_lines(report))

    lines.extend(["", "## Top Failures", ""])
    if report["top_failures"]:
        lines.extend(f"- {issue}" for issue in report["top_failures"])
    else:
        lines.append("- none")

    lines.extend(["", "## Section Issues", ""])
    for section in report["sections"]:
        lines.extend([f"### {section['section']}", ""])
        if section["issues"]:
            lines.extend(f"- {issue}" for issue in section["issues"])
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _quality_link_or_na(url: Any) -> str:
    if isinstance(url, str) and url.strip():
        return f"[prompt/response]({url.strip()})"
    return "n/a"


def _score_explanation_lines(report: dict[str, Any]) -> list[str]:
    scoring = report.get("scoring_config") or DEFAULT_SCORING_CONFIG
    lines = [
        "",
        "## Quality Score Explanation",
        "",
        (
            "The quality score is deterministic. Each section starts at `100` and "
            "loses points for validator issues, revision rounds, and very short "
            "section text. It is separate from the LLM critic rubric, which scores "
            "semantic legal quality on a `1-5` scale."
        ),
        "",
        "| Rule | Current setting |",
        "| --- | --- |",
        (
            "| Validator issues | "
            f"-{scoring['validator_issue_penalty']} each, capped at "
            f"{scoring['validator_issue_penalty_cap']} |"
        ),
        (
            "| Revision rounds | "
            f"-{scoring['revision_penalty']} each, capped at "
            f"{scoring['revision_penalty_cap']} |"
        ),
        (
            "| Short section | "
            f"-{scoring['short_section_penalty']} if under "
            f"{scoring['short_section_word_threshold']} words |"
        ),
        "| Overall score | Average of section scores |",
        "| Verdict bands | good `85+`, acceptable `70-84`, risky `50-69`, bad `<50` |",
        "",
        "| Section | Base | Issue penalty | Revision penalty | Short penalty | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for section in report.get("sections", []):
        breakdown = section.get("score_breakdown") or {}
        lines.append(
            "| "
            f"{section.get('section') or 'unknown'} | "
            f"{_int_display(breakdown.get('base'), 100)} | "
            f"-{_int_display(breakdown.get('issue_penalty'), 0)} | "
            f"-{_int_display(breakdown.get('revision_penalty'), 0)} | "
            f"-{_int_display(breakdown.get('short_section_penalty'), 0)} | "
            f"{_int_display(section.get('score'), 0)} |"
        )
    return lines


def _int_display(value: Any, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback
