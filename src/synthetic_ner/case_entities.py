"""Person and organisation generation helpers for synthetic cases."""

import random
import re

from faker import Faker
from src.synthetic_ner.constants import COMPANY_SUFFIXES, NATIONALITY_ADJECTIVES, PERSON_ROLES
from src.synthetic_ner.types.app_config import (
    PersonSpecConfig,
    PersonVariantGenerationConfig,
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
    nickname_variants: int,
    misspelling_variants: int,
    nat_locales: dict[str, str],
    is_defendant: bool = True,
) -> dict:
    locale = _resolve_locale(nat, nat_locales)
    fake = Faker(locale)

    raw_name = build_clean_person_name(fake)
    parts = raw_name.split()
    if len(parts) < 2:
        fallback_last_name = clean_person_part(fake.last_name()) or raw_name
        parts = [raw_name, fallback_last_name]
        raw_name = " ".join(parts)

    first_name = parts[0]
    last_name = parts[-1]

    initials = ".".join(part[0].upper() for part in parts if part) + "."
    title_surname = f"{title} {last_name}" if title else last_name
    short_name = first_name

    forms = [raw_name]
    if surface_forms >= 2:
        forms.append(initials)
    if surface_forms >= 3:
        forms.append(title_surname)
    if surface_forms >= 4:
        forms.append(short_name)
    forms.extend(
        build_person_name_variants(
            first_name=first_name,
            last_name=last_name,
            existing_forms=forms,
            nickname_variants=nickname_variants,
            misspelling_variants=misspelling_variants,
        )
    )

    date_of_birth = fake.date_of_birth(minimum_age=30, maximum_age=65).strftime("%-d %B %Y")
    birthplace = fake.city()
    street = fake.street_address()
    city_postcode = f"{fake.city()} {fake.postcode()}"

    return {
        "name": raw_name,
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


def build_person_name_variants(
    *,
    first_name: str,
    last_name: str,
    existing_forms: list[str],
    nickname_variants: int,
    misspelling_variants: int,
) -> list[str]:
    variants: list[str] = []

    for nickname in _nickname_candidates(first_name):
        _append_unique_variant(
            variants,
            existing_forms,
            f"{nickname} {last_name}",
            limit=nickname_variants,
        )
        if len(variants) >= nickname_variants:
            break

    misspellings = _misspelling_variants(first_name, last_name)
    added_misspellings = 0
    for misspelling in misspellings:
        if _append_unique_variant(
            variants,
            existing_forms,
            misspelling,
            limit=nickname_variants + misspelling_variants,
        ):
            added_misspellings += 1
        if added_misspellings >= misspelling_variants:
            break

    return variants


def make_org(
    nat: str,
    nat_locales: dict[str, str],
    vat_prefixes: dict[str, str],
) -> dict:
    locale = _resolve_locale(nat, nat_locales)
    fake = Faker(locale)
    suffix = random.choice(COMPANY_SUFFIXES)
    base_name = build_company_base_name(fake)
    street = fake.street_address()
    city_postcode = f"{fake.city()} {fake.postcode()}"
    vat_prefix = _resolve_vat_prefix(nat, vat_prefixes)
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

    required = _required_person_fields(person)
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")

    name = required["name"]
    first_name, last_name = _first_and_last_name(name)
    address = required["address"]
    street, city_postcode = _resolved_address_parts(
        address=address,
        street=person.get("street"),
        city_postcode=person.get("city_postcode"),
    )
    surface_forms_list = person.get("surface_forms_list") or [name]
    if not isinstance(surface_forms_list, list) or not surface_forms_list:
        raise ValueError(f"{context}.surface_forms_list must be a non-empty list")

    title = person.get("title", "")
    nationality = required["nationality"]
    return {
        "name": name,
        "initials": person.get("initials") or make_initials(name),
        "title_surname": person.get("title_surname") or (
            f"{title} {last_name}".strip() if title else last_name
        ),
        "short_name": person.get("short_name") or first_name,
        "surface_forms_list": surface_forms_list,
        "dob": required["dob"],
        "birthplace": required["birthplace"],
        "nationality": NATIONALITY_ADJECTIVES.get(nationality, nationality),
        "role": required["role"],
        "street": street,
        "city_postcode": city_postcode,
        "address": address,
        "is_defendant": is_defendant,
    }


def normalize_org_record(org: dict, context: str) -> dict:
    if not isinstance(org, dict):
        raise ValueError(f"{context} must be a mapping")

    street, city_postcode, address = _resolve_address_fields(org)
    required = {
        "name": org.get("name"),
        "street": street,
        "city_postcode": city_postcode,
        "vat": org.get("vat"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")

    return {
        "name": required["name"],
        "street": street,
        "city_postcode": city_postcode,
        "address": address or f"{street}, {city_postcode}",
        "vat": required["vat"],
        "nationality": org.get("nationality", ""),
    }


def build_people_from_specs(
    specs: list[PersonSpecConfig],
    nat_locales: dict[str, str],
    is_defendant: bool,
    person_variants: PersonVariantGenerationConfig,
) -> list[dict]:
    return [
        make_person(
            nat=spec.nationality,
            title=spec.title,
            surface_forms=spec.surface_forms,
            nickname_variants=(
                person_variants.nickname_variants if spec.variants.nickname else 0
            ),
            misspelling_variants=(
                person_variants.misspelling_variants
                if spec.variants.misspelling
                else 0
            ),
            nat_locales=nat_locales,
            is_defendant=is_defendant,
        )
        for spec in specs
    ]


def build_orgs_from_count(
    count: int,
    def_nats: list[str],
    nat_locales: dict[str, str],
    vat_prefixes: dict[str, str],
) -> list[dict]:
    return [
        make_org(
            nat=random.choice(def_nats),
            nat_locales=nat_locales,
            vat_prefixes=vat_prefixes,
        )
        for _ in range(count)
    ]


def _misspelling_variants(first_name: str, last_name: str) -> list[str]:
    misspellings: list[str] = []
    for variant_last_name in _misspelled_name_candidates(last_name):
        misspellings.append(f"{first_name} {variant_last_name}")
        for nickname in _nickname_candidates(first_name)[:1]:
            misspellings.append(f"{nickname} {variant_last_name}")
    return misspellings


def _nickname_candidates(first_name: str) -> list[str]:
    cleaned = first_name.strip()
    if len(cleaned) <= 4:
        return []
    candidates = []
    for size in (4, 5, 3):
        if len(cleaned) > size:
            candidates.append(cleaned[:size])
    if cleaned.endswith(("as", "os", "us")) and len(cleaned) > 5:
        candidates.append(cleaned[:-2])
    if cleaned.endswith("olas") and len(cleaned) > 6:
        candidates.append(cleaned[:-4])
    return _unique_strings(candidates)


def _misspelled_name_candidates(name: str) -> list[str]:
    candidates = []
    replacements = (
        ("z", "s"),
        ("s", "z"),
        ("c", "k"),
        ("k", "c"),
        ("i", "y"),
        ("y", "i"),
        ("ph", "f"),
        ("f", "ph"),
        ("ck", "k"),
    )
    lowered = name.lower()
    for source, replacement in replacements:
        index = lowered.find(source)
        if index == -1:
            continue
        cased_replacement = _match_replacement_case(name[index : index + len(source)], replacement)
        candidates.append(name[:index] + cased_replacement + name[index + len(source) :])

    for index, char in enumerate(name):
        if index > 0 and char.lower() in "bcdfghjklmnpqrstvwxyz":
            candidates.append(name[:index] + char + name[index:])
            break

    return _unique_strings(candidates)


def _append_unique_variant(
    variants: list[str],
    existing_forms: list[str],
    candidate: str,
    *,
    limit: int,
) -> bool:
    if len(variants) >= limit:
        return False
    normalized_candidate = candidate.casefold()
    known_forms = {form.casefold() for form in [*existing_forms, *variants]}
    if not candidate.strip() or normalized_candidate in known_forms:
        return False
    variants.append(candidate)
    return True


def _match_replacement_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _unique_strings(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _resolve_locale(nationality: str, nat_locales: dict[str, str]) -> str:
    try:
        return nat_locales[nationality]
    except KeyError as exc:
        raise ValueError(
            f"Missing nationality_locales entry for '{nationality}'"
        ) from exc


def _resolve_vat_prefix(nationality: str, vat_prefixes: dict[str, str]) -> str:
    try:
        return vat_prefixes[nationality]
    except KeyError as exc:
        raise ValueError(
            f"Missing vat_prefixes entry for '{nationality}'"
        ) from exc


def _required_person_fields(person: dict) -> dict[str, str]:
    street, city_postcode, address = _resolve_address_fields(person)
    person["street"] = street
    person["city_postcode"] = city_postcode
    return {
        "name": person.get("name"),
        "dob": person.get("dob"),
        "birthplace": person.get("birthplace"),
        "nationality": person.get("nationality"),
        "role": person.get("role"),
        "address": address,
    }


def _resolve_address_fields(record: dict) -> tuple[str | None, str | None, str | None]:
    street = record.get("street")
    city_postcode = record.get("city_postcode")
    address = record.get("address")
    if is_auto(address) and street and city_postcode:
        address = f"{street}, {city_postcode}"
    elif isinstance(address, str) and (not street or not city_postcode):
        street, city_postcode = split_address(address)
    return street, city_postcode, address


def _resolved_address_parts(
    *,
    address: str,
    street: str | None,
    city_postcode: str | None,
) -> tuple[str, str]:
    if street and city_postcode:
        return street, city_postcode
    return split_address(address)


def _first_and_last_name(name: str) -> tuple[str, str]:
    parts = name.split()
    if len(parts) < 2:
        parts = [name, name]
    return parts[0], parts[-1]
