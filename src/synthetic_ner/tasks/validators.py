"""Deterministic validators for generated section text."""

from __future__ import annotations

from src.synthetic_ner.tasks.facts import (
    CASE_REF_RE,
    DATE_RE,
    INITIALS_RE,
    ORG_NAME_RE,
    TITLE_NAME_RE,
    VAT_RE,
    collect_allowed_facts,
    normalize_phrase,
    normalize_title_phrase,
)


def validate_section_text(
    *,
    section_name: str,
    section_text: str,
    document,
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

    known_entities = [
        person["name_plain"] for person in document.defendants
    ] + [
        person["name"] for person in document.collateral
    ] + [
        org["name"] for org in (document.charged_orgs + document.associated_orgs)
    ]
    if section_name in {"history", "charges", "facts", "findings", "background"}:
        if known_entities and not any(entity in text for entity in known_entities):
            issues.append("Section does not mention any known case entity.")

    allowed = collect_allowed_facts(document)
    issues.extend(_find_unknown_case_refs(text, allowed.case_refs))
    issues.extend(_find_unknown_dates(text, allowed.dates))
    issues.extend(_find_unknown_vats(text, allowed.vat_numbers))
    issues.extend(_find_unknown_orgs(text, allowed.org_names))
    issues.extend(_find_unknown_titled_people(text, allowed.titled_people))
    issues.extend(_find_unknown_initials(text, allowed.initials))

    return list(dict.fromkeys(issues))


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
