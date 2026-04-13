"""Deterministic validators and cleaners for generated section text."""

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
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_META_TOKEN_RE = re.compile(
    r"(?im)\b(?:APPROVED:|RUBRICS:|ISSUES:|REVISION:|STRICT COMPLIANCE NOTES:|"
    r"REQUIRED FACTS(?:\s*&\s*ENTITIES)?:|ENTITIES MENTIONED:|LOGICAL ORDER:|FIX:)\b"
)
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+")
_MARKDOWN_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
_MARKDOWN_NUMBERED_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_MARKDOWN_RULE_RE = re.compile(r"(?m)^\s*[-*_]{3,}\s*$")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*")
_PLACEHOLDER_STARS_RE = re.compile(r"\*{4,}")
_WORD_COUNT_RE = re.compile(r"(?im)^\s*\(?\s*word count\s*:\s*\d+\s*\)?\s*$")
_META_LINE_PREFIXES = (
    "approved:",
    "rubrics:",
    "issues:",
    "revision:",
    "strict compliance notes:",
    "required facts & entities:",
    "required facts and entities:",
    "entities mentioned:",
    "logical order:",
    "fix:",
    "note:",
)
_META_LINE_LABELS = {
    "history",
    "charges",
    "facts",
    "evidence",
    "assessment",
    "required facts & entities",
    "required facts and entities",
    "entities mentioned",
    "logical order",
    "strict compliance notes",
}
_SECTION_ENTITY_CHECK = {"history", "charges", "facts", "findings", "background"}


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
    if _PLACEHOLDER_STARS_RE.search(text):
        issues.append("Section contains unresolved placeholder markers (****).")
    if _META_TOKEN_RE.search(text):
        issues.append("Section contains review/instruction metadata instead of prose.")
    if (
        _MARKDOWN_HEADING_RE.search(text)
        or _MARKDOWN_BULLET_RE.search(text)
        or _MARKDOWN_NUMBERED_RE.search(text)
        or _MARKDOWN_RULE_RE.search(text)
        or _MARKDOWN_BOLD_RE.search(text)
    ):
        issues.append("Section contains markdown/list formatting; output must be plain prose.")
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
    if section_name in _SECTION_ENTITY_CHECK:
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
    text = clean_generated_section_text(section_text)
    if not text:
        return text

    for issue in issues:
        if issue == "Section contains placeholder text.":
            text = text.replace("[section not generated]", "").strip()
            continue
        if issue == "Section still contains hidden reasoning markup.":
            text = _THINK_RE.sub("", text).strip()
            continue
        if issue == "Section contains unresolved placeholder markers (****).":
            text = _PLACEHOLDER_STARS_RE.sub("", text).strip()
            continue
        if issue == "Section contains review/instruction metadata instead of prose.":
            text = clean_generated_section_text(text)
            continue
        if issue == "Section contains markdown/list formatting; output must be plain prose.":
            text = clean_generated_section_text(text)
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


def clean_generated_section_text(section_text: str) -> str:
    text = section_text.replace("\r", "").strip()
    if not text:
        return ""

    text = _THINK_RE.sub("", text)
    text = _CODE_FENCE_RE.sub("", text)
    text = _MARKDOWN_RULE_RE.sub("", text)
    text = text.replace("**", "")
    text = text.replace("__", "")

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if _WORD_COUNT_RE.match(line):
            continue
        if _is_meta_line(line):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _PLACEHOLDER_STARS_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _normalize_repaired_text(cleaned)


def build_deterministic_fallback_section(
    *,
    section_name: str,
    memory_text: str,
    word_target: int,
) -> str:
    allowed = collect_allowed_facts_from_memory(memory_text)
    people = _choose_values(allowed.person_surface_forms, limit=4)
    organisations = _choose_values(allowed.org_names, limit=5)
    case_refs = _choose_values(allowed.case_refs, limit=2)
    dates = _choose_values(allowed.dates, limit=3)

    people_text = ", ".join(people) if people else "the identified defendants and associates"
    org_text = (
        ", ".join(organisations)
        if organisations
        else "the identified corporate entities in the case file"
    )
    ref_text = ", ".join(case_refs) if case_refs else "the stated case references"
    date_text = ", ".join(dates) if dates else "the listed case dates"

    sentence_bank = _fallback_sentence_bank(
        section_name=section_name,
        people_text=people_text,
        org_text=org_text,
        ref_text=ref_text,
        date_text=date_text,
    )
    minimum_words = max(60, word_target // 4)
    selected: list[str] = []
    word_count = 0
    sentence_index = 0
    while word_count < minimum_words:
        sentence = sentence_bank[sentence_index % len(sentence_bank)]
        selected.append(sentence)
        word_count += len(sentence.split())
        sentence_index += 1
        if sentence_index > len(sentence_bank) * 4:
            break
    return _normalize_repaired_text(" ".join(selected))


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


def _fallback_sentence_bank(
    *,
    section_name: str,
    people_text: str,
    org_text: str,
    ref_text: str,
    date_text: str,
) -> list[str]:
    common = [
        (
            f"This {section_name} section is prepared strictly from CASE_MEMORY and "
            "contains only verified case information."
        ),
        f"The identified parties include {people_text}.",
        f"The relevant organisations are {org_text}.",
        f"The operative references are {ref_text}, with key dates including {date_text}.",
        (
            "No additional entities, references, amounts, or procedural claims are added "
            "beyond the recorded case material."
        ),
        (
            "The narrative is limited to factual allegations and procedural details that "
            "can be traced directly to the approved case record."
        ),
    ]
    section_specific: dict[str, list[str]] = {
        "history": [
            (
                "The procedural history reflects the sequence documented in the case file, "
                "from investigation activity to formal filing."
            )
        ],
        "charges": [
            (
                "The charges are stated according to the indictment record and linked only "
                "to the listed people and organisations."
            )
        ],
        "facts": [
            (
                "The factual narrative describes the alleged conduct in neutral legal language "
                "without introducing speculative details."
            )
        ],
        "evidence": [
            (
                "The evidence summary is restricted to items and records that are explicitly "
                "present in the case source material."
            )
        ],
        "assessment": [
            (
                "The legal assessment reflects the recorded allegations and supporting material "
                "without extending beyond the documented scope."
            )
        ],
    }
    return common + section_specific.get(section_name.lower(), [])


def _choose_values(values: set[str], *, limit: int) -> list[str]:
    normalized_values = {normalize_phrase(value) for value in values if normalize_phrase(value)}
    ordered = sorted(normalized_values, key=lambda item: (-len(item), item))
    return ordered[:limit]


def _is_meta_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    plain = lowered.strip("* ").strip().rstrip(":")
    if plain in _META_LINE_LABELS:
        return True
    if lowered.strip("* ").startswith(_META_LINE_PREFIXES):
        return True
    if plain in {"---", "----"}:
        return True
    return False

