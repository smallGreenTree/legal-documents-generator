"""Canonical fact helpers for prompts and deterministic validation."""

from __future__ import annotations

from src.synthetic_ner.tasks.document_generation.constants import (
    CASE_REF_RE,
    DATE_RE,
    INITIALS_RE,
    ORG_NAME_RE,
    TITLE_NAME_RE,
    VAT_RE,
)
from src.synthetic_ner.text.entities import AMOUNT_RE, normalize_phrase
from src.synthetic_ner.types.document_generation import AllowedFacts


def build_allowed_facts_section(document) -> str:
    metadata = document.metadata
    offence_period = metadata.get("offence_period")
    case_refs = [
        f"- Case number: {metadata['case_number']}",
        f"- Legal reference: {metadata.get('legal_reference', 'none')}",
        f"- Cross reference: {metadata['cross_ref']}",
        f"- Filing date: {metadata['filing_date']}",
    ]
    if offence_period:
        case_refs.append(f"- Offence period: {offence_period[0]} to {offence_period[1]}")

    people_lines = []
    for person in document.defendants + document.collateral:
        forms = unique_phrases(
            [
                person["name"],
                person["initials"],
                person["title_surname"],
                person["short_name"],
                *person["surface_forms_list"],
            ]
        )
        people_lines.append(
            f"- {person['name']} | allowed forms: {'; '.join(forms)} | "
            f"dob: {person['dob']} | nationality: {person['nationality']}"
        )

    organisation_lines = [
        (
            f"- {org['name']} | role: {org.get('role') or 'organisation'} | "
            f"VAT: {org['vat']} | address: {org['address']}"
        )
        for org in (document.charged_orgs + document.associated_orgs)
    ]
    amount_lines = []
    total_loss = document.amounts.get("total_loss")
    if total_loss:
        amount_lines.append(f"- Total alleged loss: {total_loss}")
    invoice_value = document.amounts.get("inflated_invoice_value")
    if invoice_value:
        amount_lines.append(f"- Inflated invoice value: {invoice_value}")
    for transfer in document.amounts.get("transfers", []):
        if isinstance(transfer, dict) and transfer.get("amount"):
            amount_lines.append(
                f"- Transfer amount: {transfer['amount']} | "
                f"from: {transfer.get('from')} | to: {transfer.get('to')}"
            )

    blocks = [
        "## Allowed References",
        (
            "- If a name, organisation, VAT number, case reference, or date is not "
            "listed here, do not use it."
        ),
        "",
        "### Case References and Dates",
        *case_refs,
        "",
        "### Allowed Person Surface Forms",
        *(people_lines or ["- none"]),
        "",
        "### Allowed Organisations",
        *(organisation_lines or ["- none"]),
        "",
        "### Allowed Amounts",
        *(amount_lines or ["- none"]),
    ]
    return "\n".join(blocks)


def collect_allowed_facts_from_memory(memory_text: str) -> AllowedFacts:
    refs_block = _extract_markdown_section(memory_text, "### Case References and Dates")
    people_block = _extract_markdown_section(memory_text, "### Allowed Person Surface Forms")
    orgs_block = _extract_markdown_section(memory_text, "### Allowed Organisations")
    amounts_block = _extract_markdown_section(memory_text, "### Allowed Amounts")
    counts_block = _extract_markdown_section(memory_text, "## Counts")

    case_refs, dates = _parse_case_refs_and_dates(refs_block)
    person_surface_forms, titled_people, initials, people_dates = _parse_people_block(people_block)
    dates.update(people_dates)
    dates.update({normalize_phrase(match) for match in DATE_RE.findall(counts_block)})
    case_refs.update({normalize_phrase(match) for match in CASE_REF_RE.findall(counts_block)})
    org_names, vat_numbers = _parse_orgs_block(orgs_block)
    amounts = {normalize_phrase(match) for match in AMOUNT_RE.findall(amounts_block)}
    amounts.update({normalize_phrase(match) for match in AMOUNT_RE.findall(counts_block)})

    return AllowedFacts(
        person_surface_forms=person_surface_forms,
        titled_people=titled_people,
        initials=initials,
        org_names=org_names,
        vat_numbers=vat_numbers,
        amounts=amounts,
        case_refs=case_refs,
        dates=dates,
    )


def _parse_case_refs_and_dates(block: str) -> tuple[set[str], set[str]]:
    case_refs = {normalize_phrase(match) for match in CASE_REF_RE.findall(block)}
    dates = {normalize_phrase(match) for match in DATE_RE.findall(block)}
    return case_refs, dates


def _parse_people_block(block: str) -> tuple[set[str], set[str], set[str], set[str]]:
    person_surface_forms: set[str] = set()
    dates: set[str] = set()

    for line in _iter_bullet_lines(block):
        parts = [part.strip() for part in line.split("|")]
        if not parts:
            continue
        _add_person_parts(parts, person_surface_forms, dates)

    person_blob = "\n".join(sorted(person_surface_forms))
    titled_people = {normalize_title_phrase(match) for match in TITLE_NAME_RE.findall(person_blob)}
    initials = {normalize_phrase(match) for match in INITIALS_RE.findall(person_blob)}
    return person_surface_forms, titled_people, initials, dates


def _add_person_parts(parts: list[str], person_surface_forms: set[str], dates: set[str]) -> None:
    base_name = normalize_phrase(parts[0])
    if base_name and base_name.lower() != "none":
        person_surface_forms.add(base_name)

    for part in parts[1:]:
        lowered = part.lower()
        if lowered.startswith("allowed forms:"):
            _add_allowed_forms(part, person_surface_forms)
        elif lowered.startswith("dob:"):
            dob = normalize_phrase(part.split(":", 1)[1])
            if dob:
                dates.add(dob)


def _add_allowed_forms(part: str, person_surface_forms: set[str]) -> None:
    raw_forms = part.split(":", 1)[1]
    for form in raw_forms.split(";"):
        normalized = normalize_phrase(form)
        if normalized:
            person_surface_forms.add(normalized)


def _parse_orgs_block(block: str) -> tuple[set[str], set[str]]:
    org_names: set[str] = set()
    vat_numbers: set[str] = set()

    for line in _iter_bullet_lines(block):
        parts = [part.strip() for part in line.split("|")]
        if not parts:
            continue
        _add_org_parts(parts, org_names, vat_numbers)

    if not org_names:
        org_names = {normalize_phrase(match) for match in ORG_NAME_RE.findall(block)}
    if not vat_numbers:
        vat_numbers = {normalize_phrase(match) for match in VAT_RE.findall(block)}
    return org_names, vat_numbers


def _add_org_parts(parts: list[str], org_names: set[str], vat_numbers: set[str]) -> None:
    org_name = normalize_phrase(parts[0])
    if org_name and org_name.lower() != "none":
        org_names.add(org_name)

    for part in parts[1:]:
        lowered = part.lower()
        if lowered.startswith("vat:"):
            vat_number = normalize_phrase(part.split(":", 1)[1])
            if vat_number:
                vat_numbers.add(vat_number)


def normalize_title_phrase(value: str) -> str:
    normalized = normalize_phrase(value)
    return (
        normalized.replace("Mr. ", "Mr ")
        .replace("Mrs. ", "Mrs ")
        .replace("Ms. ", "Ms ")
        .replace("Dr. ", "Dr ")
        .replace("Prof. ", "Prof ")
    )


def unique_phrases(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if not value:
            continue
        normalized = normalize_phrase(value)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _extract_markdown_section(memory_text: str, heading: str) -> str:
    start = memory_text.find(heading)
    if start == -1:
        return ""

    start_index = start + len(heading)
    tail = memory_text[start_index:]
    next_h2 = tail.find("\n## ")
    next_h3 = tail.find("\n### ")

    candidates = [idx for idx in (next_h2, next_h3) if idx != -1]
    end_index = min(candidates) if candidates else len(tail)
    return tail[:end_index]


def _iter_bullet_lines(block: str) -> list[str]:
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
    return lines
