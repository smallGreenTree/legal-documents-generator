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
