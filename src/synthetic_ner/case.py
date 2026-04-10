"""Case data generation and normalization."""

import random
import re
from datetime import date, timedelta

from faker import Faker
from src.synthetic_ner.config import resolve_section_order
from src.synthetic_ner.constants import (
    COMPANY_SUFFIXES,
    COURTS,
    NATIONALITY_ADJECTIVES,
    PERSON_ROLES,
)
from src.synthetic_ner.utils import is_auto, make_initials, split_address, strip_titles


def make_vat(prefix: str) -> str:
    def digits(length: int) -> str:
        return "".join(str(random.randint(0, 9)) for _ in range(length))

    formats = {
        "GB": lambda: f"GB{digits(9)}",
        "DE": lambda: f"DE{digits(9)}",
        "FR": lambda: (
            f"FR{random.choice('ABCDEFGHJKLMNPQRSTUVWXY')}{random.randint(0, 9)}{digits(9)}"
        ),
        "ES": lambda: (
            f"ES{random.choice('ABCDEFGHJNPQRSUVW')}{digits(7)}"
            f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
        ),
        "IT": lambda: f"IT{digits(11)}",
        "NL": lambda: f"NL{digits(9)}B{digits(2)}",
        "BE": lambda: f"BE0{digits(9)}",
        "PL": lambda: f"PL{digits(10)}",
    }
    return formats.get(prefix, lambda: f"{prefix}{digits(9)}")()


def make_case_number(doc_type: str) -> str:
    year = random.randint(2024, 2026)
    number = random.randint(100, 9999)
    if doc_type == "indictment":
        return f"CPS/{year}/{number:04d}"
    return f"T{year}{number:05d}"


def make_cross_ref() -> str:
    return f"C/{random.randint(2024, 2026)}/{random.randint(100, 9999)}"


def make_filing_date() -> str:
    filing_date = date.today() - timedelta(days=random.randint(30, 400))
    return filing_date.strftime("%-d %B %Y")


def make_offence_period() -> tuple[str, str]:
    end_date = date.today() - timedelta(days=random.randint(60, 300))
    start_date = end_date - timedelta(days=random.randint(90, 540))
    fmt = "%-d %B %Y"
    return start_date.strftime(fmt), end_date.strftime(fmt)


def clean_person_part(value: str) -> str:
    cleaned = strip_titles(str(value or ""))
    cleaned = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    return cleaned


def build_clean_person_name(fake: Faker) -> str:
    first_name = clean_person_part(fake.first_name())
    last_name = clean_person_part(fake.last_name())
    if first_name and last_name:
        return f"{first_name} {last_name}"

    fallback = clean_person_part(fake.name())
    parts = fallback.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"
    if fallback:
        return fallback
    return f"{clean_person_part(fake.first_name())} {clean_person_part(fake.last_name())}".strip()


def clean_company_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9& -]", " ", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -&")
    return cleaned.upper()


def build_company_base_name(fake: Faker) -> str:
    tokens = []
    target_parts = 2 if random.random() < 0.45 else 1

    for _ in range(6):
        token = clean_company_token(fake.last_name())
        if token and token not in tokens:
            tokens.append(token)
        if len(tokens) >= target_parts:
            break

    if not tokens:
        fallback = clean_company_token(fake.company())
        tokens = [part for part in fallback.split() if part][:target_parts]

    return " ".join(tokens) if tokens else "ARDEN"


def make_person(
    nat: str,
    title: str,
    surface_forms: int,
    nat_locales: dict,
    is_defendant: bool = True,
) -> dict:
    locale = nat_locales.get(nat, "en_GB")
    fake = Faker(locale)

    raw_name = build_clean_person_name(fake)
    parts = raw_name.split()
    if len(parts) < 2:
        fallback_last_name = clean_person_part(fake.last_name()) or raw_name
        parts = [raw_name, fallback_last_name]
        raw_name = " ".join(parts)

    first_name = parts[0]
    last_name = parts[-1]

    display_name = raw_name.upper() if is_defendant else raw_name
    initials = ".".join(part[0].upper() for part in parts if part) + "."
    title_surname = f"{title} {last_name}" if title else last_name
    short_name = first_name

    forms = [display_name]
    if surface_forms >= 2:
        forms.append(initials)
    if surface_forms >= 3:
        forms.append(title_surname)
    if surface_forms >= 4:
        forms.append(short_name)

    date_of_birth = fake.date_of_birth(minimum_age=30, maximum_age=65).strftime("%-d %B %Y")
    birthplace = fake.city()
    street = fake.street_address()
    city_postcode = f"{fake.city()} {fake.postcode()}"

    return {
        "name": display_name,
        "name_plain": raw_name,
        "initials": initials,
        "title_surname": title_surname,
        "short_name": short_name,
        "surface_forms_list": forms,
        "dob": date_of_birth,
        "birthplace": birthplace,
        "nationality": NATIONALITY_ADJECTIVES.get(nat, nat),
        "role": random.choice(PERSON_ROLES),
        "street": street,
        "city_postcode": city_postcode,
        "address": f"{street}, {city_postcode}",
        "is_defendant": is_defendant,
    }


def make_org(nat: str, nat_locales: dict, vat_prefixes: dict) -> dict:
    locale = nat_locales.get(nat, "en_GB")
    fake = Faker(locale)
    suffix = random.choice(COMPANY_SUFFIXES)
    base_name = build_company_base_name(fake)
    street = fake.street_address()
    city_postcode = f"{fake.city()} {fake.postcode()}"
    vat_prefix = vat_prefixes.get(nat, nat)
    return {
        "name": f"{base_name} {suffix}",
        "street": street,
        "city_postcode": city_postcode,
        "address": f"{street}, {city_postcode}",
        "vat": make_vat(vat_prefix),
        "nationality": nat,
    }


def normalize_person_record(person: dict, is_defendant: bool, context: str) -> dict:
    if not isinstance(person, dict):
        raise ValueError(f"{context} must be a mapping")

    name = person.get("name")
    dob = person.get("dob")
    birthplace = person.get("birthplace")
    nationality = person.get("nationality")
    role = person.get("role")

    street = person.get("street")
    city_postcode = person.get("city_postcode")
    address = person.get("address")

    if is_auto(address) and street and city_postcode:
        address = f"{street}, {city_postcode}"
    elif isinstance(address, str) and (not street or not city_postcode):
        street, city_postcode = split_address(address)

    missing = [
        key
        for key, value in (
            ("name", name),
            ("dob", dob),
            ("birthplace", birthplace),
            ("nationality", nationality),
            ("role", role),
            ("address", address),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")

    raw_name = person.get("name_plain") or name
    parts = raw_name.split()
    if len(parts) < 2:
        parts = [raw_name, raw_name]

    first_name = parts[0]
    last_name = parts[-1]
    title = person.get("title", "")

    surface_forms_list = person.get("surface_forms_list") or [name]
    if not isinstance(surface_forms_list, list) or not surface_forms_list:
        raise ValueError(f"{context}.surface_forms_list must be a non-empty list")

    return {
        "name": name,
        "name_plain": raw_name,
        "initials": person.get("initials") or make_initials(raw_name),
        "title_surname": person.get("title_surname") or (
            f"{title} {last_name}".strip() if title else last_name
        ),
        "short_name": person.get("short_name") or first_name,
        "surface_forms_list": surface_forms_list,
        "dob": dob,
        "birthplace": birthplace,
        "nationality": NATIONALITY_ADJECTIVES.get(nationality, nationality),
        "role": role,
        "street": street or split_address(address)[0],
        "city_postcode": city_postcode or split_address(address)[1],
        "address": address,
        "is_defendant": is_defendant,
    }


def normalize_org_record(org: dict, context: str) -> dict:
    if not isinstance(org, dict):
        raise ValueError(f"{context} must be a mapping")

    name = org.get("name")
    vat = org.get("vat")
    street = org.get("street")
    city_postcode = org.get("city_postcode")
    address = org.get("address")

    if is_auto(address) and street and city_postcode:
        address = f"{street}, {city_postcode}"
    elif isinstance(address, str) and (not street or not city_postcode):
        street, city_postcode = split_address(address)

    missing = [
        key
        for key, value in (
            ("name", name),
            ("street", street),
            ("city_postcode", city_postcode),
            ("vat", vat),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")

    return {
        "name": name,
        "street": street,
        "city_postcode": city_postcode,
        "address": address or f"{street}, {city_postcode}",
        "vat": vat,
        "nationality": org.get("nationality", ""),
    }


def build_people_from_specs(specs: list, nat_locales: dict, is_defendant: bool) -> list[dict]:
    return [
        make_person(
            nat=spec["nationality"],
            title=spec.get("title", ""),
            surface_forms=spec.get("surface_forms", 1),
            nat_locales=nat_locales,
            is_defendant=is_defendant,
        )
        for spec in specs
    ]


def build_orgs_from_count(
    count: int,
    def_nats: list,
    nat_locales: dict,
    vat_prefixes: dict,
) -> list[dict]:
    return [
        make_org(
            nat=random.choice(def_nats),
            nat_locales=nat_locales,
            vat_prefixes=vat_prefixes,
        )
        for _ in range(count)
    ]


def resolve_case_entities(
    profile: dict,
    case_cfg: dict,
    nat_locales: dict,
    vat_prefixes: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    cast_cfg = case_cfg.get("cast")
    if not isinstance(cast_cfg, dict):
        cast_cfg = profile.get("cast", {})

    explicit_defendants = case_cfg.get("defendants")
    if isinstance(explicit_defendants, list):
        defendants = [
            normalize_person_record(person, True, f"case.defendants[{index}]")
            for index, person in enumerate(explicit_defendants)
        ]
    else:
        defendants = build_people_from_specs(
            cast_cfg.get("defendants", []),
            nat_locales,
            True,
        )

    explicit_collateral = case_cfg.get("collateral")
    if isinstance(explicit_collateral, list):
        collateral = [
            normalize_person_record(person, False, f"case.collateral[{index}]")
            for index, person in enumerate(explicit_collateral)
        ]
    else:
        collateral = build_people_from_specs(
            cast_cfg.get("collateral", []),
            nat_locales,
            False,
        )

    defendant_nationalities = [
        spec["nationality"]
        for spec in cast_cfg.get("defendants", [])
        if isinstance(spec, dict) and spec.get("nationality")
    ] or ["GB"]

    explicit_charged_orgs = case_cfg.get("charged_orgs")
    if isinstance(explicit_charged_orgs, list):
        charged_orgs = [
            normalize_org_record(org, f"case.charged_orgs[{index}]")
            for index, org in enumerate(explicit_charged_orgs)
        ]
    else:
        charged_orgs = build_orgs_from_count(
            cast_cfg.get("charged_orgs", 3),
            defendant_nationalities,
            nat_locales,
            vat_prefixes,
        )

    explicit_associated_orgs = case_cfg.get("associated_orgs")
    if isinstance(explicit_associated_orgs, list):
        associated_orgs = [
            normalize_org_record(org, f"case.associated_orgs[{index}]")
            for index, org in enumerate(explicit_associated_orgs)
        ]
    else:
        associated_orgs = build_orgs_from_count(
            cast_cfg.get("associated_orgs", 0),
            defendant_nationalities,
            nat_locales,
            vat_prefixes,
        )

    return defendants, collateral, charged_orgs, associated_orgs


def resolve_case_metadata(case_cfg: dict, doc_type: str) -> dict:
    metadata = case_cfg.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    offence_period_cfg = metadata.get("offence_period")
    if is_auto(offence_period_cfg):
        offence_period = None
    else:
        if not isinstance(offence_period_cfg, dict):
            raise ValueError("case.metadata.offence_period must be a mapping")
        start_date = offence_period_cfg.get("start")
        end_date = offence_period_cfg.get("end")
        if is_auto(start_date) and is_auto(end_date):
            offence_period = None
        elif (
            isinstance(start_date, str)
            and isinstance(end_date, str)
            and start_date
            and end_date
        ):
            offence_period = (start_date, end_date)
        else:
            raise ValueError(
                "case.metadata.offence_period must define non-empty start and end strings"
            )

    return {
        "court": (
            metadata.get("court")
            if not is_auto(metadata.get("court"))
            else random.choice(COURTS)
        ),
        "case_number": (
            metadata.get("case_number")
            if not is_auto(metadata.get("case_number"))
            else make_case_number(doc_type)
        ),
        "cross_ref": (
            metadata.get("cross_ref")
            if not is_auto(metadata.get("cross_ref"))
            else make_cross_ref()
        ),
        "filing_date": (
            metadata.get("filing_date")
            if not is_auto(metadata.get("filing_date"))
            else make_filing_date()
        ),
        "offence_period": offence_period,
    }


def normalize_counts(counts_cfg: list) -> list[dict]:
    counts = []
    for index, count in enumerate(counts_cfg):
        if not isinstance(count, dict):
            raise ValueError(f"case.counts[{index}] must be a mapping")
        missing = [
            key
            for key in ("offence", "statute", "particulars")
            if not count.get(key)
        ]
        if missing:
            raise ValueError(
                f"case.counts[{index}] is missing required fields: {', '.join(missing)}"
            )
        counts.append({
            "offence": count["offence"],
            "statute": count["statute"],
            "particulars": count["particulars"],
        })
    return counts


def build_counts(
    cfg: dict,
    fraud_type: str,
    defendants: list,
    orgs: list,
    offence_period: tuple[str, str] | None = None,
) -> list[dict]:
    statutes = cfg.get("fraud_statutes", {}).get(fraud_type)
    if not statutes:
        return []

    start_date, end_date = offence_period or make_offence_period()
    defendants_str = " and ".join(person["name"] for person in defendants)
    companies_str = " and ".join(org["name"] for org in orgs)

    counts = []
    for statute in statutes:
        particulars = (
            statute["particulars"]
            .replace("{defendants}", defendants_str)
            .replace("{companies}", companies_str)
            .replace("{start_date}", start_date)
            .replace("{end_date}", end_date)
            .strip()
        )
        counts.append({
            "offence": statute["offence"],
            "statute": statute["statute"],
            "particulars": particulars,
        })
    return counts


def resolve_counts(
    cfg: dict,
    case_cfg: dict,
    doc_type: str,
    fraud_type: str,
    defendants: list,
    charged_orgs: list,
    offence_period: tuple[str, str] | None,
) -> list[dict]:
    if doc_type != "indictment":
        return []

    counts_cfg = case_cfg.get("counts")
    if isinstance(counts_cfg, list):
        return normalize_counts(counts_cfg)

    return build_counts(
        cfg,
        fraud_type,
        defendants,
        charged_orgs,
        offence_period=offence_period,
    )


def resolve_prose_overrides(
    case_cfg: dict,
    generation_cfg,
    doc_type: str,
) -> dict[str, str]:
    prose_cfg = case_cfg.get("prose")
    if is_auto(prose_cfg):
        return {}
    if not isinstance(prose_cfg, dict):
        raise ValueError("case.prose must be a mapping")

    section_order = resolve_section_order(generation_cfg, doc_type)
    extra = [name for name in prose_cfg if name not in section_order]
    if extra:
        raise ValueError(
            f"case.prose has unknown sections for {doc_type}: {', '.join(extra)}"
        )

    resolved = {}
    for section_name in section_order:
        value = prose_cfg.get(section_name, "auto")
        if is_auto(value):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"case.prose.{section_name} must be a non-empty string or 'auto'"
            )
        resolved[section_name] = value.strip()
    return resolved
