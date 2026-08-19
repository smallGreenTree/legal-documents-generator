"""Markdown memory extraction helpers for validation."""

from __future__ import annotations

import re

from src.synthetic_ner.tasks.document_generation.facts import (
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


def extract_charged_period(memory_text: str) -> str:
    counts_block = extract_markdown_block(memory_text, "Counts")
    match = re.search(
        rf"between\s+({DATE_TEXT_PATTERN})\s+and\s+({DATE_TEXT_PATTERN})",
        counts_block,
    )
    if not match:
        return ""
    return f"{normalize_phrase(match.group(1))} and {normalize_phrase(match.group(2))}"


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
