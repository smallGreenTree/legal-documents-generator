"""Section-specific validation contracts."""

from __future__ import annotations

from src.synthetic_ner.tasks.validation_memory import (
    extract_case_refs_and_dates,
    extract_charged_period,
    extract_organisations_from_memory,
    extract_people_from_block,
)


def validate_facts_contract(
    text: str,
    memory_text: str,
    *,
    has_meta_summary_style: bool,
) -> list[str]:
    issues: list[str] = []
    issues.extend(_defendant_issues(text, memory_text))
    issues.extend(_organisation_issues(text, memory_text))
    issues.extend(_date_issues(text, memory_text))
    if has_meta_summary_style:
        issues.append("Facts section uses template wording instead of factual narrative.")
    return issues


def _defendant_issues(text: str, memory_text: str) -> list[str]:
    defendants = extract_people_from_block(memory_text, "Defendants", limit=3)
    if not defendants:
        return []
    defendant_mentions = sum(1 for defendant in defendants if defendant in text)
    if defendant_mentions < min(2, len(defendants)):
        return ["Facts section omits key defendants listed in CASE_MEMORY."]
    return []


def _organisation_issues(text: str, memory_text: str) -> list[str]:
    organisations = extract_organisations_from_memory(memory_text, limit=5)
    if not organisations:
        return []
    organisation_mentions = sum(1 for organisation in organisations if organisation in text)
    if organisation_mentions < min(3, len(organisations)):
        return ["Facts section omits key organisations listed in CASE_MEMORY."]
    return []


def _date_issues(text: str, memory_text: str) -> list[str]:
    charged_period = extract_charged_period(memory_text)
    _, key_dates = extract_case_refs_and_dates(memory_text)
    if charged_period:
        return _charged_period_issues(text, charged_period)
    if key_dates and not any(date in text for date in key_dates[:2]):
        return ["Facts section omits key case dates from CASE_MEMORY."]
    return []


def _charged_period_issues(text: str, charged_period: str) -> list[str]:
    start_date, _, end_date = charged_period.partition(" and ")
    if start_date and end_date and (start_date not in text or end_date not in text):
        return ["Facts section omits one or both charged-period dates from CASE_MEMORY."]
    return []
