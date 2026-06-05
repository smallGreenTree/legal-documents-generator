"""Deterministic validators and cleaners for generated section text."""

from __future__ import annotations

import re
from collections.abc import Mapping

from src.synthetic_ner.tasks.document_generation.facts import (
    AMOUNT_RE,
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
from src.synthetic_ner.tasks.document_generation.validation_contracts import validate_facts_contract
from src.synthetic_ner.tasks.document_generation.validation_repetition import (
    has_repeated_long_sentences,
    has_repeated_sentence_fragments,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
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
_INCOMPLETE_RANGE_RE = re.compile(r"(?i)\bbetween\s+and\b")
_DANGLING_BETWEEN_RE = re.compile(r"(?i)\bbetween(?:\s+[A-Za-z0-9,.-]+){0,5}\s*(?:[.,;:]|$)")
_BROKEN_TIMELINE_RE = re.compile(
    r"(?i)\b(?:commenced|started|began)\s+on\s+and\s+(?:continued|lasted)\s+until\b"
)
_TRUNCATED_END_RE = re.compile(
    r"(?i)\b(?:and|or|to|of|with|through|including|by|for|from|in|on|at|between)\.?$"
)
_VAT_LABEL_RE = re.compile(r"(?i)\bVAT(?:\s+Registration\s+No\.)?\s*:\s*([A-Z0-9]+)\b")
_META_SUMMARY_OPENING_RE = re.compile(
    r"(?i)\bthis\s+(?:history|charges|facts|evidence|assessment)\s+section\s+is\s+"
    r"(?:drawn|prepared)\s+strictly\s+from\s+case_memory\b"
)
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
    "persons",
    "companies",
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
_SECTION_ENTITY_CHECK = {
    "persons",
    "companies",
    "history",
    "charges",
    "facts",
    "findings",
    "background",
}
DEFAULT_VALIDATORS = {
    "empty_section": True,
    "placeholder_text": True,
    "hidden_reasoning_markup": True,
    "placeholder_markers": True,
    "review_metadata": True,
    "meta_summary_style": True,
    "markdown_formatting": True,
    "incomplete_date_range": True,
    "dangling_between_phrase": True,
    "unresolved_timeline_placeholder": True,
    "partial_vat_identifier": True,
    "repeated_long_sentences": True,
    "repeated_sentence_fragments": True,
    "truncated_sentence": True,
    "minimum_length": True,
    "required_person_facts": True,
    "required_company_facts": True,
    "known_entity_presence": True,
    "unknown_case_references": True,
    "unknown_dates": True,
    "unknown_amounts": True,
    "unknown_vat_numbers": True,
    "unknown_organisations": True,
    "unknown_titled_people": True,
    "unknown_initials": True,
    "facts_contract": True,
}


def validate_section_text(
    *,
    section_name: str,
    section_text: str,
    memory_text: str,
    word_target: int,
    min_completion_ratio: float = 0.7,
    enabled_validators: Mapping[str, bool] | None = None,
) -> list[str]:
    validators = _normalise_validator_config(enabled_validators)
    text = section_text.strip()
    if not text:
        return ["Section is empty."] if validators["empty_section"] else []

    issues = _basic_section_issues(
        text,
        word_target,
        min_completion_ratio,
        validators,
    )
    allowed = collect_allowed_facts_from_memory(memory_text)
    issues.extend(_required_section_fact_issues(section_name, text, memory_text, validators))
    issues.extend(_entity_presence_issues(section_name, text, allowed, validators))
    issues.extend(_unknown_value_issues(text, allowed, validators))
    if section_name.lower() == "facts" and validators["facts_contract"]:
        issues.extend(
            validate_facts_contract(
                text,
                memory_text,
                has_meta_summary_style=_contains_meta_summary_style(text),
            )
        )

    return list(dict.fromkeys(issues))


def _required_section_fact_issues(
    section_name: str,
    text: str,
    memory_text: str,
    validators: Mapping[str, bool],
) -> list[str]:
    if section_name == "persons" and validators["required_person_facts"]:
        return _required_person_fact_issues(text, memory_text)
    if section_name == "companies" and validators["required_company_facts"]:
        return _required_company_fact_issues(text, memory_text)
    return []


def _required_person_fact_issues(text: str, memory_text: str) -> list[str]:
    people = _memory_people(memory_text)
    issues = []
    for person in people:
        for field in ("name", "dob", "nationality", "role", "address"):
            value = person.get(field)
            if value and value not in text:
                issues.append(
                    f"Section is missing required person {field} '{value}'."
                )
    return issues


def _required_company_fact_issues(text: str, memory_text: str) -> list[str]:
    companies = _memory_companies(memory_text)
    issues = []
    for company in companies:
        for field in ("name", "vat", "address"):
            value = company.get(field)
            if value and value not in text:
                issues.append(
                    f"Section is missing required company {field} '{value}'."
                )
    return issues


def _basic_section_issues(
    text: str,
    word_target: int,
    min_completion_ratio: float,
    validators: Mapping[str, bool],
) -> list[str]:
    checks = (
        (
            "placeholder_text",
            text == "[section not generated]",
            "Section contains placeholder text.",
        ),
        (
            "hidden_reasoning_markup",
            "<think>" in text.lower(),
            "Section still contains hidden reasoning markup.",
        ),
        (
            "placeholder_markers",
            bool(_PLACEHOLDER_STARS_RE.search(text)),
            "Section contains unresolved placeholder markers (****).",
        ),
        (
            "review_metadata",
            bool(_META_TOKEN_RE.search(text)),
            "Section contains review/instruction metadata instead of prose.",
        ),
        (
            "meta_summary_style",
            _contains_meta_summary_style(text),
            "Section contains template/meta summary wording instead of legal prose.",
        ),
        (
            "markdown_formatting",
            _contains_markdown_formatting(text),
            "Section contains markdown/list formatting; output must be plain prose.",
        ),
        (
            "incomplete_date_range",
            bool(_INCOMPLETE_RANGE_RE.search(text)),
            "Section contains an incomplete date range ('between ... and ...').",
        ),
        (
            "dangling_between_phrase",
            _has_dangling_between(text),
            "Section contains a dangling 'between' phrase.",
        ),
        (
            "unresolved_timeline_placeholder",
            bool(_BROKEN_TIMELINE_RE.search(text)),
            "Section contains unresolved timeline placeholders.",
        ),
        (
            "partial_vat_identifier",
            _has_partial_vat_identifier(text),
            "Section contains a truncated VAT/reference identifier.",
        ),
        (
            "repeated_long_sentences",
            has_repeated_long_sentences(text),
            "Section contains repeated long sentences/paragraphs.",
        ),
        (
            "repeated_sentence_fragments",
            has_repeated_sentence_fragments(text),
            "Section contains repeated sentence fragments.",
        ),
        (
            "truncated_sentence",
            len(text) > 120 and bool(_TRUNCATED_END_RE.search(text.strip())),
            "Section appears truncated or ends mid-sentence.",
        ),
        (
            "minimum_length",
            len(text.split()) < _minimum_section_words(word_target, min_completion_ratio),
            "Section is significantly shorter than the requested target.",
        ),
    )
    return [issue for key, failed, issue in checks if validators[key] and failed]


def _minimum_section_words(word_target: int, min_completion_ratio: float) -> int:
    return max(60, int(word_target * min_completion_ratio))


def _contains_markdown_formatting(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (
            _MARKDOWN_HEADING_RE,
            _MARKDOWN_BULLET_RE,
            _MARKDOWN_NUMBERED_RE,
            _MARKDOWN_RULE_RE,
            _MARKDOWN_BOLD_RE,
        )
    )


def _memory_people(memory_text: str) -> list[dict[str, str]]:
    people_by_name: dict[str, dict[str, str]] = {}
    for heading in ("Defendants", "Collateral"):
        for line in _memory_bullet_lines(memory_text, heading):
            _add_memory_person(people_by_name, line)

    for line in _memory_bullet_lines(memory_text, "Allowed Person Surface Forms"):
        _add_memory_person_dob(people_by_name, line)
    return list(people_by_name.values())


def _add_memory_person(
    people_by_name: dict[str, dict[str, str]],
    line: str,
) -> None:
    parts = [part.strip() for part in line.split("|")]
    if not parts or not parts[0] or parts[0].lower() == "none":
        return
    person = people_by_name.setdefault(parts[0], {"name": parts[0]})
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = [item.strip() for item in part.split(":", 1)]
        if key in {"role", "nationality", "address"} and value:
            person[key] = value


def _add_memory_person_dob(
    people_by_name: dict[str, dict[str, str]],
    line: str,
) -> None:
    parts = [part.strip() for part in line.split("|")]
    if not parts or parts[0].lower() == "none":
        return
    person = people_by_name.setdefault(parts[0], {"name": parts[0]})
    for part in parts[1:]:
        if part.lower().startswith("dob:"):
            dob = part.split(":", 1)[1].strip()
            if dob:
                person["dob"] = dob


def _memory_companies(memory_text: str) -> list[dict[str, str]]:
    companies = []
    for line in _memory_bullet_lines(memory_text, "Organisations"):
        parts = [part.strip() for part in line.split("|")]
        if not parts or not parts[0] or parts[0].lower() == "none":
            continue
        company = {"name": parts[0]}
        for part in parts[1:]:
            if ":" not in part:
                continue
            key, value = [item.strip() for item in part.split(":", 1)]
            if key.upper() == "VAT" and value:
                company["vat"] = value
            elif key == "address" and value:
                company["address"] = value
        companies.append(company)
    return companies


def _memory_bullet_lines(memory_text: str, heading: str) -> list[str]:
    block = _memory_section(memory_text, heading)
    return [
        line.strip()[2:].strip()
        for line in block.splitlines()
        if line.strip().startswith("- ")
    ]


def _memory_section(memory_text: str, heading: str) -> str:
    for marker in (f"## {heading}", f"### {heading}"):
        start = memory_text.find(marker)
        if start == -1:
            continue
        start_index = start + len(marker)
        tail = memory_text[start_index:]
        next_h2 = tail.find("\n## ")
        next_h3 = tail.find("\n### ")
        candidates = [index for index in (next_h2, next_h3) if index != -1]
        end_index = min(candidates) if candidates else len(tail)
        return tail[:end_index]
    return ""


def _entity_presence_issues(
    section_name: str,
    text: str,
    allowed,
    validators: Mapping[str, bool],
) -> list[str]:
    if not validators["known_entity_presence"] or section_name not in _SECTION_ENTITY_CHECK:
        return []
    known_entities = [
        entity
        for entity in (allowed.person_surface_forms | allowed.org_names)
        if len(entity) >= 4
    ]
    if known_entities and not any(entity in text for entity in known_entities):
        return ["Section does not mention any known case entity."]
    return []


def _unknown_value_issues(text: str, allowed, validators: Mapping[str, bool]) -> list[str]:
    return [
        *(
            _find_unknown_case_refs(text, allowed.case_refs)
            if validators["unknown_case_references"]
            else []
        ),
        *(_find_unknown_dates(text, allowed.dates) if validators["unknown_dates"] else []),
        *(_find_unknown_amounts(text, allowed.amounts) if validators["unknown_amounts"] else []),
        *(
            _find_unknown_vats(text, allowed.vat_numbers)
            if validators["unknown_vat_numbers"]
            else []
        ),
        *(
            _find_unknown_orgs(text, allowed.org_names)
            if validators["unknown_organisations"]
            else []
        ),
        *(
            _find_unknown_titled_people(text, allowed.titled_people)
            if validators["unknown_titled_people"]
            else []
        ),
        *(_find_unknown_initials(text, allowed.initials) if validators["unknown_initials"] else []),
    ]


def _normalise_validator_config(
    enabled_validators: Mapping[str, bool] | None,
) -> dict[str, bool]:
    validators = dict(DEFAULT_VALIDATORS)
    if enabled_validators:
        validators.update(
            {
                key: bool(value)
                for key, value in enabled_validators.items()
                if key in validators
            }
        )
    return validators


def clean_generated_section_text(section_text: str) -> str:
    text = section_text.replace("\r", "").strip()
    if not text:
        return ""

    text = _THINK_RE.sub("", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            continue
        if not line:
            cleaned_lines.append("")
            continue
        if _WORD_COUNT_RE.match(line):
            continue
        if _is_meta_line(line):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _normalize_cleaned_text(cleaned)


def _normalize_cleaned_text(text: str) -> str:
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


def _find_unknown_amounts(text: str, allowed_amounts: set[str]) -> list[str]:
    issues = []
    for match in sorted({normalize_phrase(value) for value in AMOUNT_RE.findall(text)}):
        if match not in allowed_amounts:
            issues.append(f"Section mentions unknown amount '{match}'.")
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


def _has_dangling_between(text: str) -> bool:
    for match in _DANGLING_BETWEEN_RE.finditer(text):
        snippet = match.group(0).strip()
        if re.search(r"(?i)\bbetween\b.+\band\b", snippet):
            continue
        if "during the charged period" in snippet.lower():
            continue
        return True
    return False


def _has_partial_vat_identifier(text: str) -> bool:
    for match in _VAT_LABEL_RE.finditer(text):
        raw_value = match.group(1).strip().upper()
        if not VAT_RE.fullmatch(raw_value):
            return True
    return False


def _contains_meta_summary_style(text: str) -> bool:
    return bool(_META_SUMMARY_OPENING_RE.search(text))
