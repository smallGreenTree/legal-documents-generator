"""Markdown memory extraction helpers for validation and fallback prose."""

from __future__ import annotations

import re

from src.synthetic_ner.tasks.facts import (
    CASE_REF_RE,
    DATE_RE,
    normalize_phrase,
)

DATE_TEXT_PATTERN = (
    r"\d{1,2} "
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December) "
    r"\d{4}"
)


def choose_values(values: set[str], *, limit: int) -> list[str]:
    normalized_values = {normalize_phrase(value) for value in values if normalize_phrase(value)}
    ordered = sorted(normalized_values, key=lambda item: (-len(item), item))
    return ordered[:limit]


def extract_people_from_block(memory_text: str, heading: str, *, limit: int) -> list[str]:
    block = extract_markdown_block(memory_text, heading)
    people: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        base = normalize_phrase(stripped[2:].split("|", 1)[0])
        if base and base.lower() != "none" and base not in people:
            people.append(base)
        if len(people) >= limit:
            break
    return people


def extract_organisations_from_memory(memory_text: str, *, limit: int) -> list[str]:
    block = extract_markdown_block(memory_text, "Organisations")
    organisations: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        base = normalize_phrase(stripped[2:].split("|", 1)[0])
        if base and base.lower() != "none" and base not in organisations:
            organisations.append(base)
        if len(organisations) >= limit:
            break
    return organisations


def extract_case_refs_and_dates(memory_text: str) -> tuple[list[str], list[str]]:
    refs_block = extract_markdown_sub_block(
        memory_text,
        heading="Allowed References",
        subheading="Case References and Dates",
    )
    case_ref_values = [normalize_phrase(value) for value in CASE_REF_RE.findall(refs_block)]
    date_values = [normalize_phrase(value) for value in DATE_RE.findall(refs_block)]
    return unique_preserve_order(case_ref_values), unique_preserve_order(date_values)


def extract_offences(memory_text: str, *, limit: int) -> list[str]:
    counts_block = extract_markdown_block(memory_text, "Counts")
    offences: list[str] = []
    for line in counts_block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        offence = normalize_phrase(stripped[2:].split("|", 1)[0])
        if offence and offence.lower() != "none" and offence not in offences:
            offences.append(offence)
        if len(offences) >= limit:
            break
    return offences


def extract_charged_period(memory_text: str) -> str:
    counts_block = extract_markdown_block(memory_text, "Counts")
    match = re.search(
        rf"between\s+({DATE_TEXT_PATTERN})\s+and\s+({DATE_TEXT_PATTERN})",
        counts_block,
    )
    if not match:
        return ""
    return f"{normalize_phrase(match.group(1))} and {normalize_phrase(match.group(2))}"


def extract_document_fields(memory_text: str) -> dict[str, str]:
    block = extract_markdown_block(memory_text, "Document")
    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        payload = stripped[2:]
        if ":" not in payload:
            continue
        key, raw_value = payload.split(":", 1)
        normalized_key = normalize_phrase(key).lower()
        normalized_value = normalize_phrase(raw_value)
        if normalized_key and normalized_value:
            fields[normalized_key] = normalized_value
    return fields


def extract_count_entries(memory_text: str, *, limit: int) -> list[tuple[str, str, str]]:
    block = extract_markdown_block(memory_text, "Counts")
    entries: list[tuple[str, str, str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        parts = [normalize_phrase(part) for part in stripped[2:].split("|")]
        offence = parts[0] if parts else ""
        statute = parts[1] if len(parts) > 1 else ""
        particulars = parts[2] if len(parts) > 2 else ""
        if offence:
            entries.append((offence, statute, particulars))
        if len(entries) >= limit:
            break
    return entries


def extract_relationship_facts(memory_text: str, *, limit: int) -> list[str]:
    block = extract_markdown_block(memory_text, "Relationship Graph")
    relationships: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        fact = normalize_phrase(stripped[2:])
        if fact and fact.lower() != "none" and fact not in relationships:
            relationships.append(fact)
        if len(relationships) >= limit:
            break
    return relationships


def extract_markdown_block(memory_text: str, heading: str) -> str:
    marker = f"## {heading}\n"
    start = memory_text.find(marker)
    if start == -1:
        return ""
    start_index = start + len(marker)
    tail = memory_text[start_index:]
    end = tail.find("\n## ")
    return tail[:end] if end != -1 else tail


def extract_markdown_sub_block(memory_text: str, *, heading: str, subheading: str) -> str:
    parent = extract_markdown_block(memory_text, heading)
    if not parent:
        return ""
    marker = f"### {subheading}\n"
    start = parent.find(marker)
    if start == -1:
        return ""
    start_index = start + len(marker)
    tail = parent[start_index:]
    end = tail.find("\n### ")
    return tail[:end] if end != -1 else tail


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def ensure_terminal_punctuation(value: str) -> str:
    cleaned = normalize_phrase(value)
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?;":
        return cleaned
    return f"{cleaned}."
