"""Ground-truth candidates derived from generated case inputs."""

from __future__ import annotations

from typing import Any

from src.synthetic_ner.core.constants import PROSECUTION
from src.synthetic_ner.text.entities import AMOUNT_RE, strip_surface_punctuation


def build_entity_references(
    document: Any,
    *,
    address_surface_forms: int = 3,
) -> list[dict[str, Any]]:
    """Build the entity catalogue from values generated before document writing."""
    references: dict[tuple[str, str], dict[str, Any]] = {}
    _add_people_groups(references, document, address_surface_forms)
    _add_organisation_groups(references, document, address_surface_forms)
    _add_metadata_references(references, document.metadata)
    _add_count_amount_references(references, document.counts_list)
    _add_amount_references(references, document.amounts)
    _add_negative_control_references(references, document.metadata)
    return list(references.values())


def select_present_entity_references(
    document_text: str,
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only generated entity surfaces that occur in the rendered text."""
    return [
        reference
        for reference in references
        if isinstance(reference.get("entity_text"), str)
        and reference["entity_text"] in document_text
    ]


def _add_people_groups(
    references: dict[tuple[str, str], dict[str, Any]],
    document: Any,
    address_surface_forms: int,
) -> None:
    for group_name, people in (
        ("defendants", document.defendants),
        ("collateral", document.collateral),
    ):
        for index, person in enumerate(people):
            _add_person_references(
                references,
                person,
                prefix=f"case.{group_name}[{index}]",
                group_name=group_name,
                address_surface_forms=address_surface_forms,
            )


def _add_person_references(
    references: dict[tuple[str, str], dict[str, Any]],
    person: dict[str, Any],
    *,
    prefix: str,
    group_name: str,
    address_surface_forms: int,
) -> None:
    name = person.get("name")
    initials = person.get("initials")
    title_surname = person.get("title_surname")
    short_name = person.get("short_name")
    surname = str(name or "").split()[-1] if name else ""
    stripped_initials = strip_surface_punctuation(initials)
    _add_reference(references, name, "PERSON", f"{prefix}.name", f"{group_name} person")
    _add_reference(
        references,
        initials,
        "INITIAL",
        f"{prefix}.initials",
        f"initials for {name}",
    )
    if stripped_initials and stripped_initials != initials:
        _add_reference(
            references,
            stripped_initials,
            "INITIAL",
            f"{prefix}.initials.normalized",
            f"normalized initials for {name}",
        )
    if surname and surname != name:
        _add_reference(
            references,
            surname,
            "PERSON",
            f"{prefix}.name.surname",
            f"surname for {name}",
        )
    if title_surname and title_surname != surname:
        _add_reference(
            references,
            title_surname,
            "TITLE",
            f"{prefix}.title_surname",
            f"title surface for {name}",
        )
    known_surfaces = {name, initials, stripped_initials, title_surname, short_name, surname}
    if short_name not in {name, initials, stripped_initials, title_surname, surname}:
        _add_reference(
            references,
            short_name,
            "PERSON",
            f"{prefix}.short_name",
            f"short name for {name}",
        )
    for surface_index, surface in enumerate(person.get("surface_forms_list") or []):
        if surface in known_surfaces:
            continue
        _add_reference(
            references,
            surface,
            "PERSON",
            f"{prefix}.surface_forms_list[{surface_index}]",
            f"configured person surface for {name}",
        )
    _add_reference(
        references,
        person.get("dob"),
        "DATE",
        f"{prefix}.dob",
        f"date of birth for {name}",
    )
    _add_address_references(
        references,
        person,
        prefix=prefix,
        notes=f"address for {name}",
        address_surface_forms=address_surface_forms,
    )


def _add_organisation_groups(
    references: dict[tuple[str, str], dict[str, Any]],
    document: Any,
    address_surface_forms: int,
) -> None:
    for group_name, organisations in (
        ("charged_orgs", document.charged_orgs),
        ("associated_orgs", document.associated_orgs),
    ):
        for index, organisation in enumerate(organisations):
            prefix = f"case.{group_name}[{index}]"
            name = organisation.get("name")
            _add_reference(
                references,
                name,
                "ORG",
                f"{prefix}.name",
                f"{group_name} organisation",
            )
            _add_address_references(
                references,
                organisation,
                prefix=prefix,
                notes=f"address for {name}",
                address_surface_forms=address_surface_forms,
            )
            _add_reference(
                references,
                organisation.get("vat"),
                "VAT",
                f"{prefix}.vat",
                f"VAT number for {name}",
            )


def _add_metadata_references(
    references: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    for field_name in ("case_number", "cross_ref", "legal_reference"):
        _add_reference(
            references,
            metadata.get(field_name),
            "CASE_REFERENCE",
            f"case.metadata.{field_name}",
            field_name.replace("_", " "),
        )
    _add_reference(
        references,
        metadata.get("filing_date"),
        "DATE",
        "case.metadata.filing_date",
        "filing date",
    )
    offence_period = metadata.get("offence_period")
    if offence_period:
        _add_reference(
            references,
            offence_period[0],
            "DATE",
            "case.metadata.offence_period.start",
            "offence period start",
        )
        _add_reference(
            references,
            offence_period[1],
            "DATE",
            "case.metadata.offence_period.end",
            "offence period end",
        )


def _add_count_amount_references(
    references: dict[tuple[str, str], dict[str, Any]],
    counts_list: list[dict[str, Any]],
) -> None:
    for count_index, count in enumerate(counts_list):
        for amount_index, amount in enumerate(_extract_amounts(count.get("particulars", ""))):
            _add_reference(
                references,
                amount,
                "AMOUNT",
                f"case.counts[{count_index}].particulars.amounts[{amount_index}]",
                "amount in count particulars",
            )


def _add_amount_references(
    references: dict[tuple[str, str], dict[str, Any]],
    amounts: dict[str, Any],
) -> None:
    for field_name, notes in (
        ("total_loss", "total alleged loss"),
        ("inflated_invoice_value", "inflated invoice value"),
    ):
        _add_reference(
            references,
            amounts.get(field_name),
            "AMOUNT",
            f"case.amounts.{field_name}",
            notes,
        )
    for index, transfer in enumerate(amounts.get("transfers", [])):
        if not isinstance(transfer, dict):
            continue
        _add_reference(
            references,
            transfer.get("amount"),
            "AMOUNT",
            f"case.amounts.transfers[{index}].amount",
            "transfer amount",
        )


def _add_negative_control_references(
    references: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    _add_reference(
        references,
        PROSECUTION,
        "NEGATIVE_CONTROL",
        "template.prosecution",
        "prosecution negative control",
    )
    _add_reference(
        references,
        metadata.get("court"),
        "NEGATIVE_CONTROL",
        "case.metadata.court",
        "court negative control",
    )


def _add_address_references(
    references: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
    *,
    prefix: str,
    notes: str,
    address_surface_forms: int,
) -> None:
    fields = ("address", "street", "city_postcode")
    for field_name in fields[:address_surface_forms]:
        _add_reference(
            references,
            record.get(field_name),
            "ADDRESS",
            f"{prefix}.{field_name}",
            notes,
        )


def _add_reference(
    references: dict[tuple[str, str], dict[str, Any]],
    value: Any,
    label: str,
    source_field: str,
    notes: str,
) -> None:
    if value is None:
        return
    entity_text = str(value).strip()
    if not entity_text:
        return
    key = (entity_text, label)
    existing = references.get(key)
    if existing is None:
        references[key] = {
            "entity_text": entity_text,
            "label": label,
            "source_fields": [source_field],
            "notes": [notes],
        }
        return
    if source_field not in existing["source_fields"]:
        existing["source_fields"].append(source_field)
    if notes not in existing["notes"]:
        existing["notes"].append(notes)


def _extract_amounts(value: str) -> list[str]:
    amounts: list[str] = []
    for match in AMOUNT_RE.findall(value):
        cleaned = strip_surface_punctuation(match)
        if cleaned and cleaned not in amounts:
            amounts.append(cleaned)
    return amounts
