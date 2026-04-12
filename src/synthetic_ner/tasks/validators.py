"""Deterministic validators for generated section text."""

from __future__ import annotations

import re

from src.synthetic_ner.tasks.facts import (
    CASE_REF_RE,
    DATE_RE,
    INITIALS_RE,
    ORG_NAME_RE,
    TITLE_NAME_RE,
    VAT_RE,
    collect_allowed_facts_from_memory,
    normalize_phrase,
    normalize_title_phrase,
)

_UNKNOWN_VALUE_ISSUE_RE = re.compile(r"Section mentions unknown [^']+ '([^']+)'\.")
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def validate_section_text(
    *,
    section_name: str,
    section_text: str,
    memory_text: str,
    word_target: int,
) -> list[str]:
    text = section_text.strip()
    if not text:
        return ["Section is empty."]

    issues = []
    lowered = text.lower()

    if text == "[section not generated]":
        issues.append("Section contains placeholder text.")
    if "<think>" in lowered:
        issues.append("Section still contains hidden reasoning markup.")
    if len(text.split()) < max(60, word_target // 4):
        issues.append("Section is significantly shorter than the requested target.")

    allowed = collect_allowed_facts_from_memory(memory_text)
    known_entities = sorted(
        [
            entity
            for entity in (allowed.person_surface_forms | allowed.org_names)
            if len(entity) >= 4
        ],
        key=len,
        reverse=True,
    )
    if section_name in {"history", "charges", "facts", "findings", "background"}:
        if known_entities and not any(entity in text for entity in known_entities):
            issues.append("Section does not mention any known case entity.")

    issues.extend(_find_unknown_case_refs(text, allowed.case_refs))
    issues.extend(_find_unknown_dates(text, allowed.dates))
    issues.extend(_find_unknown_vats(text, allowed.vat_numbers))
    issues.extend(_find_unknown_orgs(text, allowed.org_names))
    issues.extend(_find_unknown_titled_people(text, allowed.titled_people))
    issues.extend(_find_unknown_initials(text, allowed.initials))

    return list(dict.fromkeys(issues))


def repair_section_text(
    *,
    section_text: str,
    issues: list[str],
    memory_text: str,
) -> str:
    text = section_text.strip()
    if not text:
        return text

    for issue in issues:
        if issue == "Section contains placeholder text.":
            text = text.replace("[section not generated]", "").strip()
            continue
        if issue == "Section still contains hidden reasoning markup.":
            text = _THINK_RE.sub("", text).strip()
            continue

        match = _UNKNOWN_VALUE_ISSUE_RE.match(issue)
        if match:
            text = _remove_unknown_value(text, match.group(1))

    allowed = collect_allowed_facts_from_memory(memory_text)
    if (
        "Section does not mention any known case entity." in issues
        and text
        and not _contains_any_entity(text, allowed.person_surface_forms | allowed.org_names)
    ):
        anchor_entity = _pick_anchor_entity(allowed.person_surface_forms | allowed.org_names)
        if anchor_entity:
            text = f"{text}\n\nThe facts above concern {anchor_entity}."

    return _normalize_repaired_text(text)


def _remove_unknown_value(text: str, value: str) -> str:
    escaped = re.escape(value)
    text = re.sub(rf"\b{escaped}\b", "", text)
    text = text.replace(value, "")
    return _normalize_repaired_text(text)


def _contains_any_entity(text: str, entities: set[str]) -> bool:
    for entity in entities:
        normalized = entity.strip()
        if len(normalized) < 4:
            continue
        if normalized in text:
            return True
    return False


def _pick_anchor_entity(entities: set[str]) -> str | None:
    candidates = sorted(
        (entity.strip() for entity in entities if entity and len(entity.strip()) >= 4),
        key=len,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _normalize_repaired_text(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"[ ]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _find_unknown_case_refs(text: str, allowed_case_refs: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_phrase(value) for value in CASE_REF_RE.findall(text)}):
        if match not in allowed_case_refs:
            issues.append(f"Section mentions unknown case reference '{match}'.")
    return issues


def _find_unknown_dates(text: str, allowed_dates: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_phrase(value) for value in DATE_RE.findall(text)}):
        if match not in allowed_dates:
            issues.append(f"Section mentions unknown date '{match}'.")
    return issues


def _find_unknown_vats(text: str, allowed_vats: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_phrase(value) for value in VAT_RE.findall(text)}):
        if match not in allowed_vats:
            issues.append(f"Section mentions unknown VAT/reference number '{match}'.")
    return issues


def _find_unknown_orgs(text: str, allowed_orgs: set[str]) -> list[str]:
    issues = []
    allowed_casefold = {value.casefold() for value in allowed_orgs}
    for match in sorted({normalize_phrase(value) for value in ORG_NAME_RE.findall(text)}):
        if match in allowed_orgs:
            continue
        match_casefold = match.casefold()
        if any(match_casefold in allowed for allowed in allowed_casefold):
            continue
        if any(allowed.endswith(match_casefold) for allowed in allowed_casefold):
            continue
        if any(allowed.startswith(match_casefold) for allowed in allowed_casefold):
            continue
        if match not in allowed_orgs:
            issues.append(f"Section mentions unknown organisation '{match}'.")
    return issues


def _find_unknown_titled_people(text: str, allowed_people: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_title_phrase(value) for value in TITLE_NAME_RE.findall(text)}):
        if match not in allowed_people:
            issues.append(f"Section mentions unknown titled person '{match}'.")
    return issues


def _find_unknown_initials(text: str, allowed_initials: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_phrase(value) for value in INITIALS_RE.findall(text)}):
        if match not in allowed_initials:
            issues.append(f"Section mentions unknown initials '{match}'.")
    return issues
