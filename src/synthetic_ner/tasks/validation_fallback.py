"""Deterministic fallback section generation."""

from __future__ import annotations

from src.synthetic_ner.tasks.facts import (
    collect_allowed_facts_from_memory,
)
from src.synthetic_ner.tasks.validation_memory import (
    choose_values,
    ensure_terminal_punctuation,
    extract_case_refs_and_dates,
    extract_charged_period,
    extract_count_entries,
    extract_document_fields,
    extract_offences,
    extract_organisations_from_memory,
    extract_people_from_block,
    extract_relationship_facts,
)


def build_deterministic_fallback_section(
    *,
    section_name: str,
    memory_text: str,
    word_target: int,
) -> str:
    allowed = collect_allowed_facts_from_memory(memory_text)
    defendants = extract_people_from_block(memory_text, "Defendants", limit=3)
    organisations = extract_organisations_from_memory(memory_text, limit=5)
    case_refs, key_dates = extract_case_refs_and_dates(memory_text)
    offences = extract_offences(memory_text, limit=2)
    charged_period = extract_charged_period(memory_text)
    document_fields = extract_document_fields(memory_text)
    count_entries = extract_count_entries(memory_text, limit=2)
    relationship_facts = extract_relationship_facts(memory_text, limit=6)

    if not defendants:
        defendants = choose_values(allowed.person_surface_forms, limit=3)
    if not organisations:
        organisations = choose_values(allowed.org_names, limit=5)
    if not case_refs:
        case_refs = choose_values(allowed.case_refs, limit=2)
    if not key_dates:
        key_dates = choose_values(allowed.dates, limit=2)

    sentence_bank = _fallback_sentence_bank(
        section_name=section_name,
        people_text=", ".join(defendants) if defendants else "the identified defendants",
        org_text=_join_or_default(
            organisations,
            "the principal organisations listed in the case file",
        ),
        ref_text=", ".join(case_refs) if case_refs else "the stated case references",
        date_text=", ".join(key_dates) if key_dates else "the listed case dates",
        offence_text=", ".join(offences) if offences else "the charged offences",
        period_text=charged_period or "the charged period recorded in the indictment",
        filing_date=document_fields.get("filing date", ""),
        court=document_fields.get("court", ""),
        count_entries=count_entries,
        relationship_facts=relationship_facts,
    )
    return _select_fallback_sentences(sentence_bank, word_target)


def _join_or_default(values: list[str], fallback: str) -> str:
    return ", ".join(values) if values else fallback


def _select_fallback_sentences(sentence_bank: list[str], word_target: int) -> str:
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
    common = _common_fallback_sentences(
        people_text=people_text,
        org_text=org_text,
        ref_text=ref_text,
        date_text=date_text,
        period_text=period_text,
        filing_date=filing_date,
        court=court,
        count_entries=count_entries,
        relationship_facts=relationship_facts,
    )
    specific = _SECTION_SENTENCES.get(section_name.lower(), lambda **_: [])(
        people_text=people_text,
        org_text=org_text,
        offence_text=offence_text,
        period_text=period_text,
    )
    return (specific + common) if specific else common


def _common_fallback_sentences(
    *,
    people_text: str,
    org_text: str,
    ref_text: str,
    date_text: str,
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
    common.extend(_count_sentences(count_entries))
    common.extend(_relationship_sentences(relationship_facts))
    return common


def _count_sentences(count_entries: list[tuple[str, str, str]]) -> list[str]:
    sentences = []
    for offence, statute, particulars in count_entries:
        if offence and statute:
            sentences.append(f"One count alleges {offence} contrary to {statute}.")
        if particulars:
            sentences.append(
                "The recorded particulars state that "
                f"{ensure_terminal_punctuation(particulars)}"
            )
    return sentences


def _relationship_sentences(relationship_facts: list[str]) -> list[str]:
    sentences = []
    if relationship_facts:
        sentences.append(
            "CASE_MEMORY records that "
            f"{ensure_terminal_punctuation(relationship_facts[0])}"
        )
    if len(relationship_facts) > 1:
        sentences.append(
            f"It further records that {ensure_terminal_punctuation(relationship_facts[1])}"
        )
    return sentences


def _person_sentences(**context) -> list[str]:
    return [
        (
            f"The persons section identifies {context['people_text']} using only the "
            "biographical and address details recorded in CASE_MEMORY."
        ),
        "No additional personal history or allegation is added in this identifying section.",
    ]


def _company_sentences(**context) -> list[str]:
    return [
        (
            f"The companies section identifies {context['org_text']} using only the "
            "registered address and VAT details recorded in CASE_MEMORY."
        ),
        (
            "No additional corporate history or transaction detail is added "
            "in this identifying section."
        ),
    ]


def _history_sentences(**context) -> list[str]:
    return [
        (
            "The procedural sequence runs from alleged conduct in the charged period "
            "to the formal filing of the indictment."
        ),
        (
            "The chronology remains limited to people, organisations, references, "
            "and dates explicitly present in CASE_MEMORY."
        ),
    ]


def _charge_sentences(**context) -> list[str]:
    return [
        (
            f"The charges correspond to {context['offence_text']} and are tied to "
            "the named defendants and organisations."
        ),
        (
            "Each allegation follows the recorded count particulars without adding "
            "new accusations."
        ),
    ]


def _facts_sentences(**context) -> list[str]:
    return [
        (
            "The factual narrative describes the alleged scheme during "
            f"{context['period_text']} and identifies the entities used in the "
            "recorded transactions."
        ),
        (
            "The description remains neutral and includes only relationships "
            "documented in CASE_MEMORY."
        ),
    ]


def _evidence_sentences(**context) -> list[str]:
    del context
    return [
        (
            "The evidence summary is limited to documented links, recorded references, "
            "and named entities from CASE_MEMORY."
        ),
        "No evidential detail is introduced beyond the case record.",
    ]


def _assessment_sentences(**context) -> list[str]:
    return [
        (
            "The legal assessment maps the documented conduct to "
            f"{context['offence_text']} using only recorded facts."
        ),
        "Any legal conclusion is confined to what the existing case material supports.",
    ]


_SECTION_SENTENCES = {
    "persons": _person_sentences,
    "companies": _company_sentences,
    "history": _history_sentences,
    "charges": _charge_sentences,
    "facts": _facts_sentences,
    "evidence": _evidence_sentences,
    "assessment": _assessment_sentences,
}


def _normalize_repaired_text(text: str) -> str:
    return " ".join(text.split()).strip()
