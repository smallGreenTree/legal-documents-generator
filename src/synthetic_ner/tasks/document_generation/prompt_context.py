"""Prompt context helpers for section-level generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS
from src.synthetic_ner.utils import load_config

_CONTRACTS_PATH = Path(__file__).resolve().parents[4] / "prompts" / "section_contracts.yaml"


def build_section_context(memory_text: str, section_name: str) -> str:
    """Build a compact evidence packet for one section prompt."""

    parts = [
        "# SECTION_CONTEXT",
        "",
        "## Section",
        f"- Name: {section_name}",
        f"- Purpose: {SECTION_DESCRIPTIONS.get(section_name, section_name)}",
        "",
        *_section_reference_parts(memory_text, section_name),
        "",
        *_section_memory_parts(memory_text, section_name),
    ]
    return "\n".join(part for part in parts if part.strip()).strip()


def build_section_contract(section_name: str) -> str:
    contracts = _load_section_contracts()
    contract = contracts.get(section_name) or contracts.get("default") or ""
    return contract.strip() if isinstance(contract, str) else ""


def _section_reference_parts(memory_text: str, section_name: str) -> list[str]:
    reference_headings = {
        "persons": ["Allowed Person Surface Forms"],
        "companies": ["Allowed Organisations"],
        "history": ["Case References and Dates"],
        "charges": [
            "Case References and Dates",
            "Allowed Person Surface Forms",
            "Allowed Organisations",
            "Allowed Amounts",
        ],
        "facts": [
            "Case References and Dates",
            "Allowed Person Surface Forms",
            "Allowed Organisations",
            "Allowed Amounts",
            "Relationship Facts",
        ],
        "evidence": [
            "Case References and Dates",
            "Allowed Organisations",
            "Allowed Amounts",
        ],
        "assessment": [
            "Case References and Dates",
            "Allowed Person Surface Forms",
            "Allowed Organisations",
            "Allowed Amounts",
            "Relationship Facts",
        ],
    }
    headings = reference_headings.get(
        section_name.lower(),
        ["Case References and Dates", "Allowed Person Surface Forms"],
    )

    parts = ["## Allowed References"]
    for heading in headings:
        parts.extend(
            ["", f"### {heading}", _extract_markdown_subsection(memory_text, heading)]
        )
    return parts


@lru_cache(maxsize=1)
def _load_section_contracts() -> dict[str, str]:
    raw = load_config(_CONTRACTS_PATH)
    if not isinstance(raw, dict):
        return {}
    contracts = raw.get("section_contracts", raw)
    if not isinstance(contracts, dict):
        return {}
    return {key: value for key, value in contracts.items() if isinstance(value, str)}


def _section_memory_parts(memory_text: str, section_name: str) -> list[str]:
    section = section_name.lower()
    if section == "persons":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Person Identity Facts",
            _person_identity_facts(memory_text),
        ]
    if section == "companies":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Organisations",
            _extract_markdown_block(memory_text, "Organisations"),
        ]
    if section == "history":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
        ]
    if section == "charges":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Defendants",
            _extract_markdown_block(memory_text, "Defendants"),
            "",
            "## Organisations",
            _extract_markdown_block(memory_text, "Organisations"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
            "",
            "## Amounts",
            _extract_markdown_block(memory_text, "Amounts"),
        ]
    if section == "facts":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
            "",
            "## Amounts",
            _extract_markdown_block(memory_text, "Amounts"),
            "",
            "## Relationship Facts",
            _extract_markdown_block(memory_text, "Relationship Graph"),
        ]
    if section == "evidence":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Evidence Categories",
            _extract_markdown_block(memory_text, "Evidence Categories"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
        ]
    if section == "assessment":
        return [
            "",
            "## Investigator Scenario Brief",
            _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
            "",
            "## Relationship Facts",
            _extract_markdown_block(memory_text, "Relationship Graph"),
        ]
    return [
        "",
        "## Investigator Scenario Brief",
        _extract_markdown_block(memory_text, "Investigator Scenario Brief"),
        "",
        "## Counts",
        _extract_markdown_block(memory_text, "Counts"),
        "",
        "## Relationship Facts",
        _extract_markdown_block(memory_text, "Relationship Graph"),
    ]


def _extract_markdown_block(memory_text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = memory_text.find(marker)
    if start == -1:
        return "- none"

    start_index = start + len(marker)
    tail = memory_text[start_index:]
    next_h2 = tail.find("\n## ")
    block = tail[:next_h2] if next_h2 != -1 else tail
    return block.strip() or "- none"


def _extract_markdown_subsection(memory_text: str, heading: str) -> str:
    marker = f"### {heading}"
    start = memory_text.find(marker)
    if start == -1:
        return "- none"

    start_index = start + len(marker)
    tail = memory_text[start_index:]
    next_heading_positions = [
        position
        for position in (tail.find("\n### "), tail.find("\n## "))
        if position != -1
    ]
    end = min(next_heading_positions) if next_heading_positions else len(tail)
    block = tail[:end]
    return block.strip() or "- none"


def _person_identity_facts(memory_text: str) -> str:
    allowed_people = _allowed_person_details(memory_text)
    lines = []
    for group, heading in (("defendant", "Defendants"), ("collateral", "Collateral")):
        for line in _iter_bullet_lines(_extract_markdown_block(memory_text, heading)):
            fields = _parse_pipe_fields(line)
            name = fields.pop("name", "")
            if not name:
                continue
            details = {
                "group": group,
                **allowed_people.get(name, {}),
                **fields,
            }
            lines.append(_format_person_identity_line(name, details))
    return "\n".join(lines) or "- none"


def _allowed_person_details(memory_text: str) -> dict[str, dict[str, str]]:
    people = {}
    block = _extract_markdown_subsection(memory_text, "Allowed Person Surface Forms")
    for line in _iter_bullet_lines(block):
        fields = _parse_pipe_fields(line)
        name = fields.pop("name", "")
        if name:
            people[name] = fields
    return people


def _iter_bullet_lines(block: str) -> list[str]:
    return [
        line[2:].strip()
        for line in block.splitlines()
        if line.strip().startswith("- ")
    ]


def _parse_pipe_fields(line: str) -> dict[str, str]:
    parts = [part.strip() for part in line.split("|") if part.strip()]
    if not parts or parts[0].lower() == "none":
        return {}

    fields = {"name": parts[0]}
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        fields[normalized_key] = value.strip()
    return fields


def _format_person_identity_line(name: str, details: dict[str, str]) -> str:
    ordered_keys = ("group", "dob", "nationality", "role", "address", "allowed_forms")
    fields = [f"{key}: {details[key]}" for key in ordered_keys if details.get(key)]
    return f"- {name} | {' | '.join(fields)}"
