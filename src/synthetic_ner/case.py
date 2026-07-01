"""Case data generation and normalization."""

import random
from datetime import date, timedelta
from typing import Any

from src.synthetic_ner.case_entities import (
    build_clean_person_name,
    build_company_base_name,
    build_orgs_from_count,
    build_orgs_from_specs,
    build_people_from_specs,
    build_person_name_variants,
    clean_company_token,
    clean_person_part,
    make_org,
    make_person,
    make_vat,
    normalize_org_record,
    normalize_person_record,
)
from src.synthetic_ner.types.app_config import (
    CaseConfig,
    CountConfig,
    PersonVariantGenerationConfig,
)
from src.synthetic_ner.utils import is_auto

__all__ = [
    "build_clean_person_name",
    "build_company_base_name",
    "build_orgs_from_count",
    "build_orgs_from_specs",
    "build_people_from_specs",
    "build_person_name_variants",
    "clean_company_token",
    "clean_person_part",
    "make_org",
    "make_person",
    "make_vat",
    "normalize_org_record",
    "normalize_person_record",
]


def make_case_number(doc_type: str) -> str:
    year = random.randint(2024, 2026)
    number = random.randint(100, 9999)
    if doc_type == "indictment":
        return f"CPS/{year}/{number:04d}"
    return f"T{year}{number:05d}"


def make_cross_ref() -> str:
    return f"C/{random.randint(2024, 2026)}/{random.randint(100, 9999)}"


def make_legal_reference() -> str:
    return f"{random.randint(1, 9_999_999):07d}/{random.randint(1, 999):03d}"


def make_filing_date() -> str:
    filing_date = date.today() - timedelta(days=random.randint(30, 400))
    return filing_date.strftime("%-d %B %Y")


def make_offence_period() -> tuple[str, str]:
    end_date = date.today() - timedelta(days=random.randint(60, 300))
    start_date = end_date - timedelta(days=random.randint(90, 540))
    fmt = "%-d %B %Y"
    return start_date.strftime(fmt), end_date.strftime(fmt)


def resolve_case_entities(
    case_cfg: CaseConfig,
    nat_locales: dict[str, str],
    vat_prefixes: dict[str, str],
    person_variants: PersonVariantGenerationConfig,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    cast_cfg = case_cfg.cast

    explicit_defendants = case_cfg.defendants
    if isinstance(explicit_defendants, list):
        defendants = [
            normalize_person_record(person, True, f"case.defendants[{index}]")
            for index, person in enumerate(explicit_defendants)
        ]
    else:
        defendants = build_people_from_specs(
            cast_cfg.defendants,
            nat_locales,
            True,
            person_variants,
        )

    explicit_collateral = case_cfg.collateral
    if isinstance(explicit_collateral, list):
        collateral = [
            normalize_person_record(person, False, f"case.collateral[{index}]")
            for index, person in enumerate(explicit_collateral)
        ]
    else:
        collateral = build_people_from_specs(
            cast_cfg.collateral,
            nat_locales,
            False,
            person_variants,
        )

    defendant_nationalities = [spec.nationality for spec in cast_cfg.defendants]
    if not defendant_nationalities and (
        cast_cfg.charged_orgs > 0 or cast_cfg.associated_orgs > 0
    ):
        raise ValueError(
            "case.cast.defendants must not be empty when company auto-generation is enabled"
        )

    explicit_charged_orgs = case_cfg.charged_orgs
    if isinstance(explicit_charged_orgs, list):
        charged_orgs = [
            normalize_org_record(org, f"case.charged_orgs[{index}]")
            for index, org in enumerate(explicit_charged_orgs)
        ]
    else:
        charged_orgs = _build_auto_orgs(
            specs=cast_cfg.organisation_specs,
            group="charged",
            count=cast_cfg.charged_orgs,
            defendant_nationalities=defendant_nationalities,
            nat_locales=nat_locales,
            vat_prefixes=vat_prefixes,
        )

    explicit_associated_orgs = case_cfg.associated_orgs
    if isinstance(explicit_associated_orgs, list):
        associated_orgs = [
            normalize_org_record(org, f"case.associated_orgs[{index}]")
            for index, org in enumerate(explicit_associated_orgs)
        ]
    else:
        associated_orgs = _build_auto_orgs(
            specs=cast_cfg.organisation_specs,
            group="associated",
            count=cast_cfg.associated_orgs,
            defendant_nationalities=defendant_nationalities,
            nat_locales=nat_locales,
            vat_prefixes=vat_prefixes,
        )

    return defendants, collateral, charged_orgs, associated_orgs


def _build_auto_orgs(
    *,
    specs: list,
    group: str,
    count: int,
    defendant_nationalities: list[str],
    nat_locales: dict[str, str],
    vat_prefixes: dict[str, str],
) -> list[dict]:
    if specs:
        return build_orgs_from_specs(
            specs,
            group,
            count,
            defendant_nationalities,
            nat_locales,
            vat_prefixes,
        )
    return build_orgs_from_count(
        count,
        defendant_nationalities,
        nat_locales,
        vat_prefixes,
    )


def resolve_case_metadata(case_cfg: CaseConfig, doc_type: str) -> dict:
    metadata = case_cfg.metadata
    offence_period_cfg = metadata.offence_period
    if is_auto(offence_period_cfg.start) and is_auto(offence_period_cfg.end):
        offence_period = None
    else:
        if (
            not is_auto(offence_period_cfg.start)
            and not is_auto(offence_period_cfg.end)
            and offence_period_cfg.start
            and offence_period_cfg.end
        ):
            offence_period = (offence_period_cfg.start, offence_period_cfg.end)
        else:
            raise ValueError(
                "case.metadata.offence_period must define non-empty start and end strings"
            )

    return {
        "court": metadata.court,
        "case_number": (
            metadata.case_number
            if not is_auto(metadata.case_number)
            else make_case_number(doc_type)
        ),
        "legal_reference": make_legal_reference(),
        "cross_ref": (
            metadata.cross_ref
            if not is_auto(metadata.cross_ref)
            else make_cross_ref()
        ),
        "filing_date": (
            metadata.filing_date
            if not is_auto(metadata.filing_date)
            else make_filing_date()
        ),
        "offence_period": offence_period,
    }


def resolve_scenario_brief(
    case_cfg: CaseConfig,
    metadata: dict,
    defendants: list,
    charged_orgs: list,
    amounts: dict,
    offence_period: tuple[str, str] | None,
) -> dict[str, Any]:
    if not case_cfg.scenario_brief:
        return {}
    return _format_scenario_value(
        case_cfg.scenario_brief,
        _scenario_template_context(
            metadata,
            defendants,
            charged_orgs,
            amounts,
            offence_period,
        ),
    )


def normalize_counts(counts_cfg: list[CountConfig]) -> list[dict]:
    return [
        {
            "offence": count.offence,
            "statute": count.statute,
            "particulars": count.particulars,
        }
        for count in counts_cfg
    ]


def make_money_amount(min_value: int = 25_000, max_value: int = 950_000) -> str:
    return f"£{random.randint(min_value, max_value):,}"


def build_amounts(charged_orgs: list, associated_orgs: list) -> dict:
    transfer_records = []
    transfer_total = 0
    if charged_orgs and associated_orgs:
        for charged_index, charged_org in enumerate(charged_orgs):
            associated_org = associated_orgs[charged_index % len(associated_orgs)]
            value = random.randint(50_000, 450_000)
            transfer_total += value
            transfer_records.append(
                {
                    "from": charged_org["name"],
                    "to": associated_org["name"],
                    "amount": f"£{value:,}",
                }
            )

    invoice_value = random.randint(25_000, 175_000)
    total_loss = transfer_total or random.randint(75_000, 750_000)
    return {
        "total_loss": f"£{total_loss:,}",
        "inflated_invoice_value": f"£{invoice_value:,}",
        "transfers": transfer_records,
    }


def build_counts(
    fraud_statutes: dict[str, list[CountConfig]],
    fraud_type: str,
    defendants: list,
    orgs: list,
    amounts: dict | None = None,
    offence_period: tuple[str, str] | None = None,
    metadata: dict | None = None,
) -> list[dict]:
    statutes = fraud_statutes.get(fraud_type)
    if not statutes:
        return []

    start_date, end_date = offence_period or make_offence_period()
    amount_values = amounts or {}
    metadata_values = metadata or {}
    total_loss = amount_values.get("total_loss", make_money_amount())
    inflated_invoice_value = amount_values.get(
        "inflated_invoice_value",
        make_money_amount(25_000, 175_000),
    )
    template_context = {
        **_scenario_template_context(
            metadata_values,
            defendants,
            orgs,
            amount_values,
            (start_date, end_date),
        ),
        "total_loss": total_loss,
        "inflated_invoice_value": inflated_invoice_value,
    }

    counts = []
    for statute in statutes:
        particulars = _format_template_string(
            statute.particulars,
            template_context,
        ).strip()
        counts.append(
            {
                "offence": _format_template_string(statute.offence, template_context),
                "statute": _format_template_string(statute.statute, template_context),
                "particulars": particulars,
            }
        )
    return counts


def resolve_counts(
    fraud_statutes: dict[str, list[CountConfig]],
    case_cfg: CaseConfig,
    doc_type: str,
    fraud_type: str,
    defendants: list,
    charged_orgs: list,
    amounts: dict | None,
    offence_period: tuple[str, str] | None,
    metadata: dict | None = None,
) -> list[dict]:
    if doc_type != "indictment":
        return []

    counts_cfg = case_cfg.counts
    if isinstance(counts_cfg, list):
        return normalize_counts(counts_cfg)

    return build_counts(
        fraud_statutes,
        fraud_type,
        defendants,
        charged_orgs,
        amounts=amounts,
        offence_period=offence_period,
        metadata=metadata,
    )


def _scenario_template_context(
    metadata: dict,
    defendants: list,
    orgs: list,
    amounts: dict,
    offence_period: tuple[str, str] | None,
) -> dict[str, str]:
    start_date, end_date = offence_period or ("", "")
    first_defendant = defendants[0] if defendants else {}
    first_org = orgs[0] if orgs else {}
    defendants_str = " and ".join(person.get("name", "") for person in defendants)
    defendants_upper = " and ".join(
        person.get("name", "").upper() for person in defendants
    )
    companies_str = " and ".join(org.get("name", "") for org in orgs)

    return {
        "legal_reference": str(metadata.get("legal_reference", "")),
        "case_number": str(metadata.get("case_number", "")),
        "cross_ref": str(metadata.get("cross_ref", "")),
        "court": str(metadata.get("court", "")),
        "filing_date": str(metadata.get("filing_date", "")),
        "start_date": start_date,
        "end_date": end_date,
        "defendants": defendants_upper,
        "defendant_names": defendants_str,
        "companies": companies_str,
        "organisations": companies_str,
        "first_defendant": str(first_defendant.get("name", "")),
        "first_defendant_upper": str(first_defendant.get("name", "")).upper(),
        "first_defendant_role": str(first_defendant.get("role", "")),
        "first_company": str(first_org.get("name", "")),
        "first_company_role": str(first_org.get("role", "")),
        "first_company_vat": str(first_org.get("vat", "")),
        "first_company_address": str(first_org.get("address", "")),
        "total_loss": str(amounts.get("total_loss", "")),
        "inflated_invoice_value": str(amounts.get("inflated_invoice_value", "")),
    }


def _format_scenario_value(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _format_template_string(value, context)
    if isinstance(value, list):
        return [_format_scenario_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _format_scenario_value(item, context)
            for key, item in value.items()
        }
    return value


def _format_template_string(value: str, context: dict[str, str]) -> str:
    return value.format_map(_SafeTemplateDict(context))


class _SafeTemplateDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_prose_overrides(
    case_cfg: CaseConfig,
    section_order: list[str],
) -> dict[str, str]:
    prose_cfg = case_cfg.prose
    if not prose_cfg:
        return {}

    extra = [name for name in prose_cfg if name not in section_order]
    if extra:
        raise ValueError(
            "case.prose has unknown sections for configured section_words: "
            + ", ".join(extra)
        )

    resolved = {}
    for section_name in section_order:
        if section_name not in prose_cfg:
            raise ValueError(
                f"case.prose is missing a value for section '{section_name}'"
            )
        value = prose_cfg[section_name]
        if is_auto(value):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"case.prose.{section_name} must be a non-empty string or 'auto'"
            )
        resolved[section_name] = value.strip()
    return resolved
