"""Recover a document after an interrupted LangGraph run."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.synthetic_ner.constants import EN_LABELS, EN_SECTIONS, PROSECUTION
from src.synthetic_ner.tasks.validators import (
    build_deterministic_fallback_section,
    clean_generated_section_text,
    repair_section_text,
    validate_section_text,
)
from src.synthetic_ner.utils import write_groundtruth

DEFAULT_WORD_TARGET = 500


@dataclass(slots=True)
class RecoverySection:
    name: str
    text: str
    source: str
    confidence: str
    issues: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a document from Langfuse trace metadata and local memory."
    )
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--trace-json", type=Path)
    parser.add_argument("--memory-dir", type=Path, default=Path("memory"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--partial-dir", type=Path, default=Path("output/_partial"))
    parser.add_argument("--templates-dir", type=Path, default=Path("templates"))
    args = parser.parse_args()

    recover_document(
        doc_id=args.doc_id,
        trace_json=args.trace_json,
        memory_dir=args.memory_dir,
        output_dir=args.output_dir,
        partial_dir=args.partial_dir,
        templates_dir=args.templates_dir,
    )


def recover_document(
    *,
    doc_id: str,
    trace_json: Path | None,
    memory_dir: Path,
    output_dir: Path,
    partial_dir: Path,
    templates_dir: Path,
) -> Path:
    case_dir = memory_dir / f"case_{doc_id}"
    memory_path = case_dir / "CASE_MEMORY.md"
    run_history_path = case_dir / "RUN_HISTORY.md"
    if not memory_path.exists():
        raise SystemExit(f"Missing memory file: {memory_path}")

    memory_text = memory_path.read_text(encoding="utf-8")
    run_history = run_history_path.read_text(encoding="utf-8") if run_history_path.exists() else ""
    trace_summary = _summarize_trace(trace_json) if trace_json else {}

    metadata = _parse_document_metadata(memory_text)
    doc_type = metadata.get("document type", "indictment")
    required_sections = _parse_required_sections(memory_text)
    if not required_sections:
        required_sections = list(EN_SECTIONS.get(doc_type, EN_SECTIONS["indictment"]))

    recovered_sections = [
        _recover_section(
            section_name=section_name,
            memory_text=memory_text,
            run_history=run_history,
            partial_dir=partial_dir,
            doc_id=doc_id,
        )
        for section_name in required_sections
    ]

    rendered_text = _render_document(
        doc_type=doc_type,
        metadata=metadata,
        counts=_parse_counts(memory_text),
        sections=[section.text for section in recovered_sections],
        templates_dir=templates_dir,
    )

    doc_dir = output_dir / doc_id
    section_dir = doc_dir / "recovered_sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    for section in recovered_sections:
        (section_dir / f"{section.name}.txt").write_text(section.text + "\n", encoding="utf-8")

    recovered_path = doc_dir / f"{doc_id}.recovered.txt"
    recovered_path.write_text(rendered_text, encoding="utf-8")
    canonical_path = doc_dir / f"{doc_id}.txt"
    canonical_path.write_text(rendered_text, encoding="utf-8")
    groundtruth_path = doc_dir / "groundtruth.tsv"
    write_groundtruth(
        groundtruth_path,
        _build_groundtruth_rows_from_memory(doc_id, memory_text, metadata),
    )
    report_path = doc_dir / "recovery_report.md"
    report_path.write_text(
        _build_recovery_report(
            doc_id=doc_id,
            trace_json=trace_json,
            trace_summary=trace_summary,
            recovered_sections=recovered_sections,
            recovered_path=recovered_path,
            memory_path=memory_path,
            run_history_path=run_history_path,
            canonical_path=canonical_path,
            groundtruth_path=groundtruth_path,
            partial_dir=partial_dir,
        ),
        encoding="utf-8",
    )
    print(f"Recovered document: {recovered_path}")
    print(f"Canonical document: {canonical_path}")
    print(f"Ground truth: {groundtruth_path}")
    print(f"Recovery report: {report_path}")
    return recovered_path


def _recover_section(
    *,
    section_name: str,
    memory_text: str,
    run_history: str,
    partial_dir: Path,
    doc_id: str,
) -> RecoverySection:
    summary = _latest_section_summary(run_history, section_name)
    candidates: list[tuple[str, str, str]] = []
    partial = _latest_partial_writer_output(partial_dir, doc_id, section_name)
    if partial:
        candidates.append(("partial_writer_output", "high", partial))
    if summary:
        candidates.append(("run_history_summary", "medium", summary))
    candidates.append(
        (
            "deterministic_case_memory_reconstruction",
            "medium",
            build_deterministic_fallback_section(
                section_name=section_name,
                memory_text=memory_text,
                word_target=DEFAULT_WORD_TARGET,
            ),
        )
    )

    best_source, best_confidence, best_text, best_issues = "", "low", "", []
    for source, confidence, raw_text in candidates:
        text = clean_generated_section_text(raw_text)
        issues = validate_section_text(
            section_name=section_name,
            section_text=text,
            memory_text=memory_text,
            word_target=DEFAULT_WORD_TARGET,
        )
        if issues:
            repaired = repair_section_text(
                section_text=text,
                issues=issues,
                memory_text=memory_text,
            )
            repaired = clean_generated_section_text(repaired)
            repaired_issues = validate_section_text(
                section_name=section_name,
                section_text=repaired,
                memory_text=memory_text,
                word_target=DEFAULT_WORD_TARGET,
            )
            if len(repaired_issues) < len(issues):
                text = repaired
                issues = repaired_issues
        best_source, best_confidence, best_text, best_issues = (
            source,
            confidence,
            text,
            issues,
        )
        if not issues:
            break

    return RecoverySection(
        name=section_name,
        text=best_text,
        source=best_source,
        confidence=best_confidence,
        issues=best_issues,
    )


def _latest_partial_writer_output(
    partial_dir: Path,
    doc_id: str,
    section_name: str,
) -> str:
    section_dir = partial_dir / doc_id / "sections" / section_name
    if not section_dir.exists():
        return ""
    revision_dirs = [
        path
        for path in section_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"r\d+", path.name)
    ]
    if not revision_dirs:
        return ""
    latest = max(revision_dirs, key=lambda path: int(path.name.removeprefix("r")))
    combined_path = latest / "combined.txt"
    if not combined_path.exists():
        return ""
    return combined_path.read_text(encoding="utf-8").strip()


def _summarize_trace(trace_json: Path | None) -> dict[str, Any]:
    if trace_json is None or not trace_json.exists():
        return {}
    payload = json.loads(trace_json.read_text(encoding="utf-8"))
    observations = payload.get("observations") or payload.get("trace", {}).get("observations") or []
    generations = [obs for obs in observations if obs.get("type") == "GENERATION"]
    input_types = sorted({type(obs.get("input")).__name__ for obs in observations})
    output_types = sorted({type(obs.get("output")).__name__ for obs in observations})
    return {
        "trace_id": payload.get("trace", {}).get("id"),
        "observations": len(observations),
        "generations": len(generations),
        "spans": sum(1 for obs in observations if obs.get("type") == "SPAN"),
        "inputs_available": any(obs.get("input") is not None for obs in observations),
        "outputs_available": any(obs.get("output") is not None for obs in observations),
        "input_types": input_types,
        "output_types": output_types,
        "generation_names": [obs.get("name") for obs in generations if obs.get("name")],
    }


def _parse_document_metadata(memory_text: str) -> dict[str, str]:
    block = _extract_block(memory_text, "Document")
    metadata: dict[str, str] = {}
    for line in block.splitlines():
        match = re.match(r"-\s*([^:]+):\s*(.+)", line.strip())
        if match:
            metadata[match.group(1).strip().lower()] = match.group(2).strip()
    return metadata


def _parse_required_sections(memory_text: str) -> list[str]:
    block = _extract_block(memory_text, "Required Sections")
    return [
        line.removeprefix("-").strip()
        for line in block.splitlines()
        if line.strip().startswith("-") and line.removeprefix("-").strip()
    ]


def _parse_counts(memory_text: str) -> list[dict[str, str]]:
    block = _extract_block(memory_text, "Counts")
    counts = []
    for line in block.splitlines():
        if not line.strip().startswith("-"):
            continue
        parts = [part.strip() for part in line.removeprefix("-").split("|", 2)]
        if len(parts) == 3:
            counts.append(
                {
                    "offence": parts[0],
                    "statute": parts[1],
                    "particulars": parts[2],
                }
            )
    return counts


def _build_groundtruth_rows_from_memory(
    doc_id: str,
    memory_text: str,
    metadata: dict[str, str],
) -> list[tuple[str, str, str, str, str]]:
    defendant_names = {
        item["name"] for item in _parse_people_block(memory_text, "Defendants")
    }
    collateral = _parse_people_block(memory_text, "Collateral")
    allowed_people = _parse_allowed_people(memory_text)
    orgs = _parse_allowed_orgs(memory_text)
    charged_org_names = _infer_charged_org_names(memory_text, orgs)

    rows: list[tuple[str, str, str, str, str]] = []
    for person in _parse_people_block(memory_text, "Defendants"):
        allowed_forms = allowed_people.get(person["name"], [person["name"]])
        for form in allowed_forms:
            rows.append((doc_id, form, "PERSON", "yes", "defendant surface form"))
        if person.get("address"):
            rows.append((doc_id, person["address"], "LOCATION", "yes", "defendant address"))

    for person in collateral:
        rows.append((doc_id, person["name"], "PERSON", "yes", "collateral person"))

    for org in orgs:
        if org["name"] in charged_org_names:
            rows.append((doc_id, org["name"], "ORG", "yes", "charged org"))
            street, city_postcode = _split_org_address(org["address"])
            if street:
                rows.append((doc_id, street, "LOCATION", "yes", "org street"))
            if city_postcode:
                rows.append((doc_id, city_postcode, "LOCATION", "yes", "org city/postcode"))
        else:
            rows.append((doc_id, org["name"], "ORG", "yes", "associated org"))

    rows.append((doc_id, PROSECUTION, "NEGATIVE_CONTROL", "no", "prosecution"))
    court = metadata.get("court")
    if court:
        rows.append((doc_id, court, "NEGATIVE_CONTROL", "no", "court"))
    return rows


def _parse_people_block(memory_text: str, heading: str) -> list[dict[str, str]]:
    people = []
    for line in _extract_block(memory_text, heading).splitlines():
        if not line.strip().startswith("-"):
            continue
        parts = [part.strip() for part in line.removeprefix("-").split("|")]
        if not parts or not parts[0]:
            continue
        item = {"name": parts[0]}
        for part in parts[1:]:
            if ":" in part:
                key, value = part.split(":", 1)
                item[key.strip().lower()] = value.strip()
        people.append(item)
    return people


def _parse_allowed_people(memory_text: str) -> dict[str, list[str]]:
    block = _extract_subsection(memory_text, "Allowed Person Surface Forms")
    people: dict[str, list[str]] = {}
    for line in block.splitlines():
        if not line.strip().startswith("-"):
            continue
        first, *rest = [part.strip() for part in line.removeprefix("-").split("|")]
        forms = [first]
        for part in rest:
            if part.lower().startswith("allowed forms:"):
                raw_forms = part.split(":", 1)[1]
                forms = [form.strip() for form in raw_forms.split(";") if form.strip()]
                break
        people[first] = forms
    return people


def _parse_allowed_orgs(memory_text: str) -> list[dict[str, str]]:
    block = _extract_subsection(memory_text, "Allowed Organisations")
    orgs = []
    for line in block.splitlines():
        if not line.strip().startswith("-"):
            continue
        parts = [part.strip() for part in line.removeprefix("-").split("|")]
        if not parts or not parts[0]:
            continue
        item = {"name": parts[0], "address": ""}
        for part in parts[1:]:
            if ":" in part:
                key, value = part.split(":", 1)
                item[key.strip().lower()] = value.strip()
        orgs.append(item)
    return orgs


def _infer_charged_org_names(memory_text: str, orgs: list[dict[str, str]]) -> set[str]:
    counts_text = _extract_block(memory_text, "Counts")
    charged = {org["name"] for org in orgs if org["name"] in counts_text}
    if charged:
        return charged
    return {org["name"] for org in orgs[:5]}


def _split_org_address(address: str) -> tuple[str, str]:
    if "," not in address:
        return address.strip(), ""
    street, city = address.rsplit(",", 1)
    return street.strip(), city.strip()


def _extract_subsection(text: str, heading: str) -> str:
    match = re.search(
        rf"^### {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^### |^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body").strip() if match else ""


def _latest_section_summary(run_history: str, section_name: str) -> str:
    pattern = re.compile(
        rf"^## Section Memory:\s*{re.escape(section_name)}\s*$"
        rf"(?P<body>.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(run_history))
    if not matches:
        return ""
    body = matches[-1].group("body")
    summary_match = re.search(
        r"^Summary:\s*(?P<summary>.*?)(?=^Issues:|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    return summary_match.group("summary").strip() if summary_match else ""


def _extract_block(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body").strip() if match else ""


def _render_document(
    *,
    doc_type: str,
    metadata: dict[str, str],
    counts: list[dict[str, str]],
    sections: list[str],
    templates_dir: Path,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(f"en_{doc_type}.j2")
    return template.render(
        prosecution=PROSECUTION,
        court=metadata.get("court", ""),
        sections=EN_SECTIONS.get(doc_type, EN_SECTIONS["indictment"]),
        labels=EN_LABELS,
        case_number=metadata.get("case number", ""),
        cross_ref=metadata.get("cross reference", ""),
        filing_date=metadata.get("filing date", ""),
        persons=[],
        orgs=[],
        counts=counts,
        llm_sections=sections,
    )


def _build_recovery_report(
    *,
    doc_id: str,
    trace_json: Path | None,
    trace_summary: dict[str, Any],
    recovered_sections: list[RecoverySection],
    recovered_path: Path,
    memory_path: Path,
    run_history_path: Path,
    canonical_path: Path,
    groundtruth_path: Path,
    partial_dir: Path,
) -> str:
    complete = all(section.text and not section.issues for section in recovered_sections)
    lines = [
        f"# Recovery Report: {doc_id}",
        "",
        f"- Recovery status: {'complete' if complete else 'partial'}",
        f"- Recovered document: {recovered_path}",
        f"- Folded document: {canonical_path}",
        f"- Ground truth: {groundtruth_path}",
        f"- Partial writer output directory: {partial_dir}",
        f"- Case memory: {memory_path}",
        f"- Run history: {run_history_path if run_history_path.exists() else 'missing'}",
        f"- Trace export: {trace_json or 'not provided'}",
    ]
    if trace_summary:
        lines.extend(
            [
                f"- Trace id: {trace_summary.get('trace_id') or 'n/a'}",
                f"- Observations: {trace_summary.get('observations', 0)}",
                f"- Generations: {trace_summary.get('generations', 0)}",
                f"- Trace inputs available: {str(trace_summary.get('inputs_available')).lower()}",
                f"- Trace outputs available: {str(trace_summary.get('outputs_available')).lower()}",
            ]
        )

    lines.extend(
        [
            "",
            "## Section Recovery",
            "",
            "| Section | Source | Confidence | Issues |",
            "| --- | --- | --- | --- |",
        ]
    )
    for section in recovered_sections:
        issues = "; ".join(section.issues) if section.issues else "none"
        lines.append(
            f"| {section.name} | {section.source} | {section.confidence} | {issues} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This reconstruction is not an exact replay of the interrupted LLM output "
            "unless sections are marked as partial_writer_output. Sections marked as "
            "partial_writer_output are recovered from writer chunks persisted during "
            "generation. Sections marked as run_history_summary are based on persisted "
            "section summaries. Sections marked as deterministic_case_memory_reconstruction "
            "were rebuilt from CASE_MEMORY using only recorded case facts.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
