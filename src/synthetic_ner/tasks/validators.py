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
from src.synthetic_ner.tasks.validation_contracts import validate_facts_contract
from src.synthetic_ner.tasks.validation_fallback import (
    build_deterministic_fallback_section as build_deterministic_fallback_section,
)
from src.synthetic_ner.tasks.validation_repetition import (
    dedupe_repeated_content,
    has_repeated_long_sentences,
    has_repeated_sentence_fragments,
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
_META_SUMMARY_SENTENCE_RE = re.compile(
    r"(?is)\b(?:"
    r"this\s+(?:history|charges|facts|evidence|assessment)\s+section\s+is\s+"
    r"(?:drawn|prepared)\s+strictly\s+from\s+case_memory[^.!?]*[.!?]?"
    r"|the\s+identified\s+parties\s+include[^.!?]*[.!?]?"
    r"|the\s+principal\s+organisations\s+relevant\s+to\s+this\s+section\s+are[^.!?]*[.!?]?"
    r"|the\s+operative\s+references\s+are[^.!?]*[.!?]?"
    r"|no\s+additional\s+entities,\s+references,\s+amounts,\s+or\s+procedural\s+claims\s+"
    r"are\s+added\s+beyond\s+the\s+recorded\s+case\s+material[^.!?]*[.!?]?"
    r")"
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

    issues = _basic_section_issues(text, word_target)
    allowed = collect_allowed_facts_from_memory(memory_text)
    issues.extend(_entity_presence_issues(section_name, text, allowed))
    issues.extend(_unknown_value_issues(text, allowed))
    if section_name.lower() == "facts":
        issues.extend(
            validate_facts_contract(
                text,
                memory_text,
                has_meta_summary_style=_contains_meta_summary_style(text),
            )
        )

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
        text = _repair_issue(text, issue)

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


def _basic_section_issues(text: str, word_target: int) -> list[str]:
    checks = (
        (text == "[section not generated]", "Section contains placeholder text."),
        ("<think>" in text.lower(), "Section still contains hidden reasoning markup."),
        (
            bool(_PLACEHOLDER_STARS_RE.search(text)),
            "Section contains unresolved placeholder markers (****).",
        ),
        (
            bool(_META_TOKEN_RE.search(text)),
            "Section contains review/instruction metadata instead of prose.",
        ),
        (
            _contains_meta_summary_style(text),
            "Section contains template/meta summary wording instead of legal prose.",
        ),
        (
            _contains_markdown_formatting(text),
            "Section contains markdown/list formatting; output must be plain prose.",
        ),
        (
            bool(_INCOMPLETE_RANGE_RE.search(text)),
            "Section contains an incomplete date range ('between ... and ...').",
        ),
        (_has_dangling_between(text), "Section contains a dangling 'between' phrase."),
        (
            bool(_BROKEN_TIMELINE_RE.search(text)),
            "Section contains unresolved timeline placeholders.",
        ),
        (
            _has_partial_vat_identifier(text),
            "Section contains a truncated VAT/reference identifier.",
        ),
        (
            has_repeated_long_sentences(text),
            "Section contains repeated long sentences/paragraphs.",
        ),
        (
            has_repeated_sentence_fragments(text),
            "Section contains repeated sentence fragments.",
        ),
        (
            len(text) > 120 and bool(_TRUNCATED_END_RE.search(text.strip())),
            "Section appears truncated or ends mid-sentence.",
        ),
        (
            len(text.split()) < max(60, word_target // 4),
            "Section is significantly shorter than the requested target.",
        ),
    )
    return [issue for failed, issue in checks if failed]


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


def _entity_presence_issues(section_name: str, text: str, allowed) -> list[str]:
    if section_name not in _SECTION_ENTITY_CHECK:
        return []
    known_entities = [
        entity
        for entity in (allowed.person_surface_forms | allowed.org_names)
        if len(entity) >= 4
    ]
    if known_entities and not any(entity in text for entity in known_entities):
        return ["Section does not mention any known case entity."]
    return []


def _unknown_value_issues(text: str, allowed) -> list[str]:
    return [
        *_find_unknown_case_refs(text, allowed.case_refs),
        *_find_unknown_dates(text, allowed.dates),
        *_find_unknown_vats(text, allowed.vat_numbers),
        *_find_unknown_orgs(text, allowed.org_names),
        *_find_unknown_titled_people(text, allowed.titled_people),
        *_find_unknown_initials(text, allowed.initials),
    ]


def _repair_issue(text: str, issue: str) -> str:
    repairer = _EXACT_ISSUE_REPAIRERS.get(issue)
    if repairer is not None:
        return repairer(text)

    match = _UNKNOWN_VALUE_ISSUE_RE.match(issue)
    if match:
        return _remove_unknown_value(text, match.group(1))
    return text


def _finish_truncated_sentence(text: str) -> str:
    if text and text[-1] not in ".!?;":
        return text.rstrip(",;: ") + "."
    return text


_EXACT_ISSUE_REPAIRERS = {
    "Section contains placeholder text.": lambda text: text.replace(
        "[section not generated]",
        "",
    ).strip(),
    "Section still contains hidden reasoning markup.": lambda text: _THINK_RE.sub(
        "",
        text,
    ).strip(),
    "Section contains unresolved placeholder markers (****).": lambda text: (
        _PLACEHOLDER_STARS_RE.sub("", text).strip()
    ),
    "Section contains review/instruction metadata instead of prose.": (
        lambda text: clean_generated_section_text(text)
    ),
    "Section contains template/meta summary wording instead of legal prose.": (
        lambda text: _remove_meta_summary_sentences(text)
    ),
    "Section contains markdown/list formatting; output must be plain prose.": (
        lambda text: clean_generated_section_text(text)
    ),
    "Section contains an incomplete date range ('between ... and ...').": (
        lambda text: _INCOMPLETE_RANGE_RE.sub(
            "during the charged period",
            text,
        ).strip()
    ),
    "Section contains a dangling 'between' phrase.": (
        lambda text: _normalize_between_phrases(text)
    ),
    "Section contains unresolved timeline placeholders.": lambda text: (
        _BROKEN_TIMELINE_RE.sub("occurred during the charged period", text)
    ),
    "Section contains a truncated VAT/reference identifier.": (
        lambda text: _drop_partial_vat_fragments(text)
    ),
    "Section contains repeated long sentences/paragraphs.": dedupe_repeated_content,
    "Section contains repeated sentence fragments.": dedupe_repeated_content,
    "Section appears truncated or ends mid-sentence.": _finish_truncated_sentence,
}


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
    cleaned = _BROKEN_TIMELINE_RE.sub("occurred during the charged period", cleaned)
    cleaned = _remove_meta_summary_sentences(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = dedupe_repeated_content(cleaned)
    cleaned = _normalize_between_phrases(cleaned)
    cleaned = _drop_partial_vat_fragments(cleaned)
    if cleaned and cleaned[-1] not in ".!?;":
        cleaned = cleaned.rstrip(",;: ") + "."
    return _normalize_repaired_text(cleaned)


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


def _normalize_between_phrases(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        snippet = match.group(0)
        if re.search(r"(?i)\bbetween\b.+\band\b", snippet):
            return snippet
        punctuation = ""
        if snippet and snippet[-1] in ".,;:":
            punctuation = snippet[-1]
        return f"during the charged period{punctuation or ''}"

    return _DANGLING_BETWEEN_RE.sub(replace, text)


def _has_partial_vat_identifier(text: str) -> bool:
    for match in _VAT_LABEL_RE.finditer(text):
        raw_value = match.group(1).strip().upper()
        if not VAT_RE.fullmatch(raw_value):
            return True
    return False


def _drop_partial_vat_fragments(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_value = match.group(1).strip().upper()
        if VAT_RE.fullmatch(raw_value):
            return match.group(0)
        return ""

    cleaned = _VAT_LABEL_RE.sub(replace, text)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned


def _contains_meta_summary_style(text: str) -> bool:
    return bool(_META_SUMMARY_OPENING_RE.search(text))


def _remove_meta_summary_sentences(text: str) -> str:
    cleaned = _META_SUMMARY_SENTENCE_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

