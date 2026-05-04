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
_INCOMPLETE_RANGE_RE = re.compile(r"(?i)\bbetween\s+and\b")
_DANGLING_BETWEEN_RE = re.compile(r"(?i)\bbetween(?:\s+[A-Za-z0-9,.-]+){0,5}\s*(?:[.,;:]|$)")
_BROKEN_TIMELINE_RE = re.compile(
    r"(?i)\b(?:commenced|started|began)\s+on\s+and\s+(?:continued|lasted)\s+until\b"
)
_TRUNCATED_END_RE = re.compile(
    r"(?i)\b(?:and|or|to|of|with|through|including|by|for|from|in|on|at|between)\.?$"
)
_VAT_LABEL_RE = re.compile(r"(?i)\bVAT(?:\s+Registration\s+No\.)?\s*:\s*([A-Z0-9]+)\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
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
    if _contains_meta_summary_style(text):
        issues.append("Section contains template/meta summary wording instead of legal prose.")
    if (
        _MARKDOWN_HEADING_RE.search(text)
        or _MARKDOWN_BULLET_RE.search(text)
        or _MARKDOWN_NUMBERED_RE.search(text)
        or _MARKDOWN_RULE_RE.search(text)
        or _MARKDOWN_BOLD_RE.search(text)
    ):
        issues.append("Section contains markdown/list formatting; output must be plain prose.")
    if _INCOMPLETE_RANGE_RE.search(text):
        issues.append("Section contains an incomplete date range ('between ... and ...').")
    if _has_dangling_between(text):
        issues.append("Section contains a dangling 'between' phrase.")
    if _BROKEN_TIMELINE_RE.search(text):
        issues.append("Section contains unresolved timeline placeholders.")
    if _has_partial_vat_identifier(text):
        issues.append("Section contains a truncated VAT/reference identifier.")
    if _has_repeated_long_sentences(text):
        issues.append("Section contains repeated long sentences/paragraphs.")
    if _has_repeated_sentence_fragments(text):
        issues.append("Section contains repeated sentence fragments.")
    if len(text) > 120 and _TRUNCATED_END_RE.search(text.strip()):
        issues.append("Section appears truncated or ends mid-sentence.")
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
    if section_name.lower() == "facts":
        issues.extend(_validate_facts_contract(text, memory_text))

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
        if issue == "Section contains template/meta summary wording instead of legal prose.":
            text = _remove_meta_summary_sentences(text)
            continue
        if issue == "Section contains markdown/list formatting; output must be plain prose.":
            text = clean_generated_section_text(text)
            continue
        if issue == "Section contains an incomplete date range ('between ... and ...').":
            text = _INCOMPLETE_RANGE_RE.sub("during the charged period", text).strip()
            continue
        if issue == "Section contains a dangling 'between' phrase.":
            text = _normalize_between_phrases(text)
            continue
        if issue == "Section contains unresolved timeline placeholders.":
            text = _BROKEN_TIMELINE_RE.sub("occurred during the charged period", text)
            continue
        if issue == "Section contains a truncated VAT/reference identifier.":
            text = _drop_partial_vat_fragments(text)
            continue
        if issue == "Section contains repeated long sentences/paragraphs.":
            text = _dedupe_repeated_content(text)
            continue
        if issue == "Section contains repeated sentence fragments.":
            text = _dedupe_repeated_content(text)
            continue
        if issue == "Section appears truncated or ends mid-sentence.":
            if text and text[-1] not in ".!?;":
                text = text.rstrip(",;: ") + "."
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
    cleaned = _BROKEN_TIMELINE_RE.sub("occurred during the charged period", cleaned)
    cleaned = _remove_meta_summary_sentences(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _dedupe_repeated_content(cleaned)
    cleaned = _normalize_between_phrases(cleaned)
    cleaned = _drop_partial_vat_fragments(cleaned)
    if cleaned and cleaned[-1] not in ".!?;":
        cleaned = cleaned.rstrip(",;: ") + "."
    return _normalize_repaired_text(cleaned)


def build_deterministic_fallback_section(
    *,
    section_name: str,
    memory_text: str,
    word_target: int,
) -> str:
    allowed = collect_allowed_facts_from_memory(memory_text)
    defendants = _extract_people_from_block(memory_text, "Defendants", limit=3)
    organisations = _extract_organisations_from_memory(memory_text, limit=5)
    case_refs, key_dates = _extract_case_refs_and_dates(memory_text)
    offences = _extract_offences(memory_text, limit=2)
    charged_period = _extract_charged_period(memory_text)
    document_fields = _extract_document_fields(memory_text)
    count_entries = _extract_count_entries(memory_text, limit=2)
    relationship_facts = _extract_relationship_facts(memory_text, limit=6)

    if not defendants:
        defendants = _choose_values(allowed.person_surface_forms, limit=3)
    if not organisations:
        organisations = _choose_values(allowed.org_names, limit=5)
    if not case_refs:
        case_refs = _choose_values(allowed.case_refs, limit=2)
    if not key_dates:
        key_dates = _choose_values(allowed.dates, limit=2)

    people_text = ", ".join(defendants) if defendants else "the identified defendants"
    org_text = (
        ", ".join(organisations)
        if organisations
        else "the principal organisations listed in the case file"
    )
    ref_text = ", ".join(case_refs) if case_refs else "the stated case references"
    date_text = ", ".join(key_dates) if key_dates else "the listed case dates"
    offence_text = ", ".join(offences) if offences else "the charged offences"
    period_text = charged_period or "the charged period recorded in the indictment"

    sentence_bank = _fallback_sentence_bank(
        section_name=section_name,
        people_text=people_text,
        org_text=org_text,
        ref_text=ref_text,
        date_text=date_text,
        offence_text=offence_text,
        period_text=period_text,
        filing_date=document_fields.get("filing date", ""),
        court=document_fields.get("court", ""),
        count_entries=count_entries,
        relationship_facts=relationship_facts,
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
    offence_text: str,
    period_text: str,
    filing_date: str,
    court: str,
    count_entries: list[tuple[str, str, str]],
    relationship_facts: list[str],
) -> list[str]:
    court_clause = f" before {court}" if court else ""
    common = [
        f"The defendants identified in this matter are {people_text}.",
        f"The organisations materially connected to the allegations are {org_text}.",
        f"The operative case references are {ref_text}.",
        f"The charged period reflected in the case record is {period_text}.",
    ]
    if date_text:
        common.append(f"The recorded dates include {date_text}.")
    if filing_date:
        common.append(f"The indictment was filed on {filing_date}{court_clause}.")
    for offence, statute, particulars in count_entries:
        if offence and statute:
            common.append(f"One count alleges {offence} contrary to {statute}.")
        if particulars:
            common.append(
                f"The recorded particulars state that {_ensure_terminal_punctuation(particulars)}"
            )
    if relationship_facts:
        common.append(f"CASE_MEMORY records that {_ensure_terminal_punctuation(relationship_facts[0])}")
    if len(relationship_facts) > 1:
        common.append(
            f"It further records that {_ensure_terminal_punctuation(relationship_facts[1])}"
        )
    section_specific: dict[str, list[str]] = {
        "persons": [
            (
                f"The persons section identifies {people_text} using only the "
                "biographical and address details recorded in CASE_MEMORY."
            ),
            "No additional personal history or allegation is added in this identifying section.",
        ],
        "companies": [
            (
                f"The companies section identifies {org_text} using only the "
                "registered address and VAT details recorded in CASE_MEMORY."
            ),
            (
                "No additional corporate history or transaction detail is added "
                "in this identifying section."
            ),
        ],
        "history": [
            (
                "The procedural sequence runs from alleged conduct in the charged period "
                "to the formal filing of the indictment."
            ),
            "The chronology remains limited to people, organisations, references, and dates explicitly present in CASE_MEMORY.",
        ],
        "charges": [
            (
                f"The charges correspond to {offence_text} and are tied to the named defendants and organisations."
            ),
            "Each allegation follows the recorded count particulars without adding new accusations.",
        ],
        "facts": [
            (
                f"The factual narrative describes the alleged scheme during {period_text} and identifies the entities used in the recorded transactions."
            ),
            "The description remains neutral and includes only relationships documented in CASE_MEMORY.",
        ],
        "evidence": [
            (
                "The evidence summary is limited to documented links, recorded references, and named entities from CASE_MEMORY."
            ),
            "No evidential detail is introduced beyond the case record.",
        ],
        "assessment": [
            (
                f"The legal assessment maps the documented conduct to {offence_text} using only recorded facts."
            ),
            "Any legal conclusion is confined to what the existing case material supports.",
        ],
    }
    specific = section_specific.get(section_name.lower(), [])
    return (specific + common) if specific else common


def _choose_values(values: set[str], *, limit: int) -> list[str]:
    normalized_values = {normalize_phrase(value) for value in values if normalize_phrase(value)}
    ordered = sorted(normalized_values, key=lambda item: (-len(item), item))
    return ordered[:limit]


def _extract_people_from_block(memory_text: str, heading: str, *, limit: int) -> list[str]:
    block = _extract_markdown_block(memory_text, heading)
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


def _extract_organisations_from_memory(memory_text: str, *, limit: int) -> list[str]:
    block = _extract_markdown_block(memory_text, "Organisations")
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


def _extract_case_refs_and_dates(memory_text: str) -> tuple[list[str], list[str]]:
    refs_block = _extract_markdown_sub_block(
        memory_text,
        heading="Allowed References",
        subheading="Case References and Dates",
    )
    case_ref_values = [normalize_phrase(value) for value in CASE_REF_RE.findall(refs_block)]
    date_values = [normalize_phrase(value) for value in DATE_RE.findall(refs_block)]
    return _unique_preserve_order(case_ref_values), _unique_preserve_order(date_values)


def _extract_offences(memory_text: str, *, limit: int) -> list[str]:
    counts_block = _extract_markdown_block(memory_text, "Counts")
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


def _extract_charged_period(memory_text: str) -> str:
    counts_block = _extract_markdown_block(memory_text, "Counts")
    match = re.search(
        r"between\s+(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\s+and\s+(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})",
        counts_block,
    )
    if not match:
        return ""
    return f"{normalize_phrase(match.group(1))} and {normalize_phrase(match.group(2))}"


def _extract_document_fields(memory_text: str) -> dict[str, str]:
    block = _extract_markdown_block(memory_text, "Document")
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


def _extract_count_entries(memory_text: str, *, limit: int) -> list[tuple[str, str, str]]:
    block = _extract_markdown_block(memory_text, "Counts")
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


def _extract_relationship_facts(memory_text: str, *, limit: int) -> list[str]:
    block = _extract_markdown_block(memory_text, "Relationship Graph")
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


def _extract_markdown_block(memory_text: str, heading: str) -> str:
    marker = f"## {heading}\n"
    start = memory_text.find(marker)
    if start == -1:
        return ""
    start_index = start + len(marker)
    tail = memory_text[start_index:]
    end = tail.find("\n## ")
    return tail[:end] if end != -1 else tail


def _extract_markdown_sub_block(memory_text: str, *, heading: str, subheading: str) -> str:
    parent = _extract_markdown_block(memory_text, heading)
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


def _unique_preserve_order(values: list[str]) -> list[str]:
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


def _has_repeated_long_sentences(text: str) -> bool:
    normalized_sentences = []
    for sentence in _SENTENCE_SPLIT_RE.split(" ".join(text.split())):
        normalized = sentence.strip().lower()
        if len(normalized) < 80:
            continue
        normalized = re.sub(r"\s+", " ", normalized)
        normalized_sentences.append(normalized)
    if len(normalized_sentences) < 2:
        return False
    seen: set[str] = set()
    for sentence in normalized_sentences:
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def _dedupe_repeated_content(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text

    deduped_lines: list[str] = []
    seen_sentence_keys: set[str] = set()
    for line in lines:
        normalized_line = " ".join(line.split()).strip()
        if not normalized_line:
            if deduped_lines and deduped_lines[-1] == "":
                continue
            deduped_lines.append("")
            continue

        key = normalized_line.casefold()
        if len(key) >= 80 and key in seen_sentence_keys:
            continue

        if (
            deduped_lines
            and deduped_lines[-1]
            and deduped_lines[-1].casefold() == key
        ):
            continue

        deduped_lines.append(line)
        if len(key) >= 80:
            seen_sentence_keys.add(key)

    cleaned = "\n".join(deduped_lines).strip()
    if not cleaned:
        return cleaned

    sentence_parts: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(" ".join(cleaned.split())):
        part = sentence.strip()
        if not part:
            continue
        key = re.sub(r"\s+", " ", part).casefold()
        if len(key) >= 90 and key in seen_sentences:
            continue
        sentence_parts.append(part)
        if len(key) >= 90:
            seen_sentences.add(key)
    return " ".join(sentence_parts).strip()


def _has_repeated_sentence_fragments(text: str) -> bool:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return False
    fragment_counts: dict[str, int] = {}
    for sentence in _SENTENCE_SPLIT_RE.split(normalized_text):
        tokens = [token.lower() for token in _TOKEN_RE.findall(sentence)]
        if len(tokens) < 8:
            continue
        fragment_key = " ".join(tokens[:10])
        fragment_counts[fragment_key] = fragment_counts.get(fragment_key, 0) + 1
        if fragment_counts[fragment_key] >= 2:
            return True
    return False


def _contains_meta_summary_style(text: str) -> bool:
    return bool(_META_SUMMARY_OPENING_RE.search(text))


def _remove_meta_summary_sentences(text: str) -> str:
    cleaned = _META_SUMMARY_SENTENCE_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _validate_facts_contract(text: str, memory_text: str) -> list[str]:
    issues: list[str] = []
    defendants = _extract_people_from_block(memory_text, "Defendants", limit=3)
    organisations = _extract_organisations_from_memory(memory_text, limit=5)
    charged_period = _extract_charged_period(memory_text)
    _, key_dates = _extract_case_refs_and_dates(memory_text)

    if defendants:
        defendant_mentions = sum(1 for defendant in defendants if defendant in text)
        if defendant_mentions < min(2, len(defendants)):
            issues.append("Facts section omits key defendants listed in CASE_MEMORY.")
    if organisations:
        organisation_mentions = sum(1 for organisation in organisations if organisation in text)
        if organisation_mentions < min(3, len(organisations)):
            issues.append("Facts section omits key organisations listed in CASE_MEMORY.")
    if charged_period:
        start_date, _, end_date = charged_period.partition(" and ")
        if start_date and end_date and (start_date not in text or end_date not in text):
            issues.append("Facts section omits one or both charged-period dates from CASE_MEMORY.")
    elif key_dates and not any(date in text for date in key_dates[:2]):
        issues.append("Facts section omits key case dates from CASE_MEMORY.")
    if _contains_meta_summary_style(text):
        issues.append("Facts section uses template wording instead of factual narrative.")
    return issues


def _ensure_terminal_punctuation(value: str) -> str:
    cleaned = normalize_phrase(value)
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?;":
        return cleaned
    return f"{cleaned}."
