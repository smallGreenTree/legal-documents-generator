"""Canonical fact helpers for prompts and deterministic validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.synthetic_ner.constants import COMPANY_SUFFIXES

MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
DATE_RE = re.compile(rf"\b\d{{1,2}} (?:{MONTH_PATTERN}) \d{{4}}\b")
VAT_RE = re.compile(r"\b[A-Z]{2}(?=[A-Z0-9]{8,14}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{8,14}\b")
CASE_REF_RE = re.compile(r"\b(?:CPS/\d{4}/\d{4}|C/\d{4}/\d{1,4}|T\d{9,10})\b")
INITIALS_RE = re.compile(r"\b(?:[A-Z]\.){2,4}\b")
TITLE_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lord)\.? [A-Z][A-Za-z'-]+\b")

_ORG_SUFFIX_PATTERN = "|".join(
    sorted((re.escape(suffix) for suffix in COMPANY_SUFFIXES), key=len, reverse=True)
)
ORG_NAME_RE = re.compile(
    rf"\b[A-Z0-9][A-Z0-9&'/-]*(?: [A-Z0-9][A-Z0-9&'/-]*)* "
    rf"(?:{_ORG_SUFFIX_PATTERN})\b"
)

@dataclass(slots=True)
class AllowedFacts:
    person_surface_forms: set[str]
    titled_people: set[str]
    initials: set[str]
    org_names: set[str]
    vat_numbers: set[str]
    case_refs: set[str]
    dates: set[str]


def build_allowed_facts_section(document, schema: dict) -> str:
    metadata = document.metadata
    offence_period = metadata.get("offence_period")
    case_refs = [
        f"- Case number: {metadata['case_number']}",
        f"- Cross reference: {metadata['cross_ref']}",
        f"- Filing date: {metadata['filing_date']}",
    ]
    if offence_period:
        case_refs.append(f"- Offence period: {offence_period[0]} to {offence_period[1]}")

    people_lines = []
    for person in document.defendants + document.collateral:
        forms = unique_phrases(
            [
                person["name_plain"],
                person["name"],
                person["initials"],
                person["title_surname"],
                person["short_name"],
                *person["surface_forms_list"],
            ]
        )
        people_lines.append(
            f"- {person['name_plain']} | allowed forms: {'; '.join(forms)} | "
            f"dob: {person['dob']} | nationality: {person['nationality']}"
        )

    organisation_lines = [
        f"- {org['name']} | VAT: {org['vat']} | address: {org['address']}"
        for org in (document.charged_orgs + document.associated_orgs)
    ]
    edge_lines = [f"- {edge['label']}" for edge in schema.get("edges", [])]

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
        "### Relationship Facts",
        *(edge_lines or ["- none"]),
    ]
    return "\n".join(blocks)


def collect_allowed_facts(document) -> AllowedFacts:
    metadata = document.metadata
    people = document.defendants + document.collateral
    person_surface_forms = {
        normalize_phrase(form)
        for person in people
        for form in (
            [
                person.get("name"),
                person.get("name_plain"),
                person.get("title_surname"),
                person.get("short_name"),
                person.get("initials"),
            ]
            + list(person.get("surface_forms_list", []))
        )
        if form
    }
    titled_people = {
        normalize_title_phrase(person["title_surname"])
        for person in people
        if person.get("title_surname")
    }
    initials = {
        normalize_phrase(person["initials"])
        for person in people
        if person.get("initials")
    }
    org_names = {
        normalize_phrase(org["name"])
        for org in (document.charged_orgs + document.associated_orgs)
        if org.get("name")
    }
    vat_numbers = {
        normalize_phrase(org["vat"])
        for org in (document.charged_orgs + document.associated_orgs)
        if org.get("vat")
    }

    case_refs = {
        normalize_phrase(metadata["case_number"]),
        normalize_phrase(metadata["cross_ref"]),
    }
    dates = {
        normalize_phrase(metadata["filing_date"]),
        *(
            {
                normalize_phrase(metadata["offence_period"][0]),
                normalize_phrase(metadata["offence_period"][1]),
            }
            if metadata.get("offence_period")
            else set()
        ),
        *{
            normalize_phrase(person["dob"])
            for person in people
            if person.get("dob")
        },
        *{
            normalize_phrase(date_text)
            for count in document.counts_list
            for date_text in DATE_RE.findall(count.get("particulars", ""))
        },
    }

    return AllowedFacts(
        person_surface_forms=person_surface_forms,
        titled_people=titled_people,
        initials=initials,
        org_names=org_names,
        vat_numbers=vat_numbers,
        case_refs=case_refs,
        dates=dates,
    )


def collect_allowed_facts_from_memory(memory_text: str) -> AllowedFacts:
    seed_memory = _extract_seed_memory(memory_text)
    refs_block = _extract_markdown_section(seed_memory, "### Case References and Dates")
    people_block = _extract_markdown_section(seed_memory, "### Allowed Person Surface Forms")
    orgs_block = _extract_markdown_section(seed_memory, "### Allowed Organisations")
    counts_block = _extract_markdown_section(seed_memory, "## Counts")

    case_refs, dates = _parse_case_refs_and_dates(refs_block)
    person_surface_forms, titled_people, initials, people_dates = _parse_people_block(
        people_block
    )
    dates.update(people_dates)
    dates.update({normalize_phrase(match) for match in DATE_RE.findall(counts_block)})
    case_refs.update({normalize_phrase(match) for match in CASE_REF_RE.findall(counts_block)})
    org_names, vat_numbers = _parse_orgs_block(orgs_block)

    return AllowedFacts(
        person_surface_forms=person_surface_forms,
        titled_people=titled_people,
        initials=initials,
        org_names=org_names,
        vat_numbers=vat_numbers,
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
    titled_people = {
        normalize_title_phrase(match)
        for match in TITLE_NAME_RE.findall(person_blob)
    }
    initials = {
        normalize_phrase(match)
        for match in INITIALS_RE.findall(person_blob)
    }
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
        org_names = {
            normalize_phrase(match)
            for match in ORG_NAME_RE.findall(block)
        }
    if not vat_numbers:
        vat_numbers = {
            normalize_phrase(match)
            for match in VAT_RE.findall(block)
        }
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


def normalize_phrase(value: str) -> str:
    return " ".join(str(value).strip().split()).strip(".,;:()[]{}")


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


def _extract_seed_memory(memory_text: str) -> str:
    marker = "\n## Document Plan\n"
    if marker in memory_text:
        return memory_text.split(marker, 1)[0]
    return memory_text


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
