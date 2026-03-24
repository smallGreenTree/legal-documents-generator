#!/usr/bin/env python3
"""
Synthetic legal document generator for NER validation.

Usage:
    poetry run python generator.py [--no-llm] [--documents N] [--doc-type TYPE] [--fraud-type TYPE]

All other options are driven by config.yaml → profile + case:
"""

import csv
import json
import math
import random
import re
from datetime import date, timedelta
from pathlib import Path

import requests
import yaml
from faker import Faker
from jinja2 import Environment, FileSystemLoader


WORDS_PER_PAGE = 500
CHUNK_SIZE = 700          # max words per Ollama call
MAX_CONTEXT_CHARS = 600   # tail of previous chunk passed as context


# ── Section weights (must sum to 1.0) ─────────────────────────────────────────

SECTION_WEIGHTS = {
    "indictment": {
        "history":    0.15,
        "charges":    0.10,
        "facts":      0.50,
        "evidence":   0.10,
        "assessment": 0.15,
    },
    "court_decision": {
        "background":  0.20,
        "findings":    0.50,
        "conclusions": 0.15,
        "sentence":    0.15,
    },
}

SECTION_DESCRIPTIONS = {
    "history":     "Procedural history: how the investigation started, search warrants, key dates, documents seized.",
    "charges":     "Charges: precise allegations against each defendant and each company.",
    "facts":       "Statement of facts: detailed narrative with specific dates, GBP/EUR amounts, addresses, invoice references, document codes.",
    "evidence":    "Evidence: a numbered list of exhibits (search records, bank statements, emails, invoices, witness statements).",
    "assessment":  "Legal assessment and motion: provisional legal characterisation of the conduct and a closing paragraph requesting the court to open proceedings.",
    "background":  "Background: how the matter came before the court, investigation history, procedural steps.",
    "findings":    "Findings of fact: what the court finds proved, with specific dates, amounts, addresses and document references.",
    "conclusions": "Legal conclusions: how the court characterises the conduct and which statutory provisions apply.",
    "sentence":    "Sentence and order: custodial term or order for each defendant, plus any confiscation or disqualification orders.",
}


# ── English-language constants (removed from config per design rule) ───────────

PROSECUTION = "Serious Fraud Office"

COURTS = [
    "Crown Court at Manchester",
    "Crown Court at Birmingham",
    "Crown Court at Leeds",
    "Crown Court at Southwark",
    "Crown Court at Bristol",
    "Crown Court at Newcastle",
    "Crown Court at Liverpool",
    "Crown Court at Sheffield",
]

PERSON_ROLES = [
    "company director",
    "managing director",
    "chief financial officer",
    "chief executive officer",
    "financial controller",
    "procurement officer",
    "operations manager",
    "compliance officer",
    "accountant",
    "consultant",
]

COMPANY_SUFFIXES = [
    "LTD",
    "LIMITED",
    "HOLDINGS LTD",
    "GROUP LTD",
    "INTERNATIONAL LTD",
    "CONSULTING LTD",
    "SERVICES LTD",
    "SOLUTIONS LTD",
]

NATIONALITY_ADJECTIVES = {
    "GB": "British",  "DE": "German",   "FR": "French",
    "IT": "Italian",  "NL": "Dutch",    "CZ": "Czech",
    "PL": "Polish",   "ES": "Spanish",  "PT": "Portuguese",
    "BE": "Belgian",  "AT": "Austrian", "SE": "Swedish",
    "DK": "Danish",   "FI": "Finnish",  "HU": "Hungarian",
    "RO": "Romanian", "BG": "Bulgarian","GR": "Greek",
    "HR": "Croatian", "SK": "Slovak",   "SI": "Slovenian",
}

EN_LABELS = {
    "file":     "File No.",
    "crossref": "Cross-Ref.",
    "date":     "Date",
    "born":     "born",
    "in":       "in",
    "address":  "residing at",
    "vat":      "VAT Registration No.",
}

EN_SECTIONS = {
    "indictment": {
        "title":              "INDICTMENT",
        "section_persons":    "SECTION I — PERSONS",
        "section_companies":  "SECTION II — COMPANIES",
        "section_history":    "SECTION III — PROCEDURAL HISTORY",
        "section_charges":    "SECTION IV — CHARGES",
        "section_facts":      "SECTION V — STATEMENT OF FACTS",
        "section_evidence":   "SECTION VI — EVIDENCE",
        "section_assessment": "SECTION VII — LEGAL ASSESSMENT",
    },
    "court_decision": {
        "title":              "JUDGMENT",
        "section_persons":    "SECTION I — PARTIES",
        "section_companies":  "SECTION II — COMPANIES",
        "section_history":    "SECTION III — BACKGROUND",
        "section_charges":    "SECTION IV — GROUNDS",
        "section_facts":      "SECTION V — FINDINGS OF FACT",
        "section_evidence":   "SECTION VI — EVIDENCE REVIEWED",
        "section_assessment": "SECTION VII — CONCLUSIONS AND ORDER",
    },
}


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Ground truth ───────────────────────────────────────────────────────────────

GROUNDTRUTH_HEADER = ["doc_id", "entity_text", "label", "should_propose", "notes"]


def write_groundtruth(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(GROUNDTRUTH_HEADER)
        for row in rows:
            w.writerow(row)


# ── Case schema ───────────────────────────────────────────────────────────────
#
# A case schema is a small knowledge graph saved as a JSON file under schemas/.
# It records who controlled what, who instructed whom, and how money flowed
# between companies.
#
# The generator auto-creates one; you can edit the JSON and re-run with
# --from-schema <path> to make the LLM narrate that specific story.
#
# Edge types
#   person→org  : controlled | directed | used_as_vehicle | laundered_through
#   person→person: instructed | conspired_with | bribed
#   org→org     : invoiced | subcontracted_to | received_funds_from

PERSON_ORG_EDGES   = ["controlled", "directed", "used_as_vehicle", "laundered_through"]
PERSON_PERS_EDGES  = ["instructed", "conspired_with", "bribed"]
ORG_ORG_EDGES      = ["invoiced", "subcontracted_to", "received_funds_from"]


def is_auto(value) -> bool:
    return value is None or value == "auto"


def make_schema_nodes(
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
) -> tuple[list[dict], list[dict]]:
    all_orgs = charged_orgs + associated_orgs
    persons = defendants + collateral

    nodes_p = [
        {"id": f"p{i}", "name": p["name_plain"], "display": p["name"],
         "type": "defendant" if p["is_defendant"] else "collateral"}
        for i, p in enumerate(persons)
    ]
    nodes_o = [
        {"id": f"o{i}", "name": o["name"],
         "type": "charged" if i < len(charged_orgs) else "associated"}
        for i, o in enumerate(all_orgs)
    ]
    return nodes_p, nodes_o


def make_case_schema(doc_id: str, fraud_type: str,
                     defendants: list, collateral: list,
                     charged_orgs: list, associated_orgs: list) -> dict:
    """Auto-generate a plausible relationship graph from the cast."""
    all_orgs = charged_orgs + associated_orgs
    nodes_p, nodes_o = make_schema_nodes(
        defendants,
        collateral,
        charged_orgs,
        associated_orgs,
    )

    edges = []

    # Each defendant controls at least one org
    for pi, p in enumerate(defendants):
        oi = pi % len(all_orgs)
        edges.append({
            "from":  f"p{pi}",
            "to":    f"o{oi}",
            "type":  "controlled",
            "label": f"{p['name_plain']} controlled {all_orgs[oi]['name']}",
        })

    # Main defendant instructs collateral witnesses
    if defendants and collateral:
        main_def_idx = 0
        for ci, _ in enumerate(collateral):
            pi_coll = len(defendants) + ci
            edges.append({
                "from":  f"p{main_def_idx}",
                "to":    f"p{pi_coll}",
                "type":  "instructed",
                "label": (
                    f"{defendants[main_def_idx]['name_plain']} instructed "
                    f"{collateral[ci]['name_plain']}"
                ),
            })

    # Defendants conspire with each other
    for i in range(len(defendants) - 1):
        edges.append({
            "from":  f"p{i}",
            "to":    f"p{i+1}",
            "type":  "conspired_with",
            "label": (
                f"{defendants[i]['name_plain']} conspired with "
                f"{defendants[i+1]['name_plain']}"
            ),
        })

    # Money flow between orgs (charged → associated)
    if len(charged_orgs) > 0 and len(associated_orgs) > 0:
        for ci, co in enumerate(charged_orgs):
            ai = ci % len(associated_orgs)
            ao = associated_orgs[ai]
            edges.append({
                "from":  f"o{ci}",
                "to":    f"o{len(charged_orgs) + ai}",
                "type":  "received_funds_from",
                "label": f"{ao['name']} received funds from {co['name']}",
            })

    return {
        "doc_id":     doc_id,
        "fraud_type": fraud_type,
        "persons":    nodes_p,
        "orgs":       nodes_o,
        "edges":      edges,
    }


def schema_to_context(schema: dict) -> str:
    """
    Convert a case schema into a plain-English narrative context string
    that is injected into every LLM section prompt.
    """
    lines = ["Established facts for this case (use these relationships throughout):"]
    for e in schema["edges"]:
        lines.append(f"  - {e['label'].rstrip('.')}.")
    return "\n".join(lines)


def write_case_schema(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")


def load_case_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Entity generators ──────────────────────────────────────────────────────────

def make_vat(prefix: str) -> str:
    r = lambda n: "".join(str(random.randint(0, 9)) for _ in range(n))
    formats = {
        "GB": lambda: f"GB{r(9)}",
        "DE": lambda: f"DE{r(9)}",
        "FR": lambda: f"FR{random.choice('ABCDEFGHJKLMNPQRSTUVWXY')}{random.randint(0,9)}{r(9)}",
        "ES": lambda: f"ES{random.choice('ABCDEFGHJNPQRSUVW')}{r(7)}{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}",
        "IT": lambda: f"IT{r(11)}",
        "NL": lambda: f"NL{r(9)}B{r(2)}",
        "BE": lambda: f"BE0{r(9)}",
        "PL": lambda: f"PL{r(10)}",
    }
    return formats.get(prefix, lambda: f"{prefix}{r(9)}")()


def make_case_number(doc_type: str) -> str:
    year = random.randint(2024, 2026)
    num = random.randint(100, 9999)
    return f"CPS/{year}/{num:04d}" if doc_type == "indictment" else f"T{year}{num:05d}"


def make_cross_ref() -> str:
    return f"C/{random.randint(2024, 2026)}/{random.randint(100, 9999)}"


def make_filing_date() -> str:
    d = date.today() - timedelta(days=random.randint(30, 400))
    return d.strftime("%-d %B %Y")


def make_offence_period() -> tuple[str, str]:
    end   = date.today() - timedelta(days=random.randint(60, 300))
    start = end - timedelta(days=random.randint(90, 540))
    fmt   = "%-d %B %Y"
    return start.strftime(fmt), end.strftime(fmt)


def _strip_titles(name: str) -> str:
    for t in ["Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Sir", "Lord"]:
        name = name.replace(t, "").strip()
    return name


def split_address(address: str) -> tuple[str, str]:
    if not isinstance(address, str):
        return "", ""
    if ", " in address:
        return tuple(address.rsplit(", ", 1))
    if "," in address:
        return tuple(part.strip() for part in address.rsplit(",", 1))
    return address, ""


def make_initials(name: str) -> str:
    parts = [p for p in _strip_titles(name).split() if p]
    if not parts:
        return ""
    return ".".join(p[0].upper() for p in parts) + "."


def make_person(nat: str, title: str, surface_forms: int,
                nat_locales: dict, is_defendant: bool = True) -> dict:
    """
    Build a person dict with all surface forms.

    surface_forms:
      1 = full name only
      2 = full name + initials
      3 = full name + title+surname + initials
      4 = full name + title+surname + short name + initials
    """
    locale = nat_locales.get(nat, "en_GB")
    fake = Faker(locale)

    raw = _strip_titles(fake.name())
    parts = raw.split()
    # ensure at least first + last
    if len(parts) < 2:
        parts = [raw, fake.last_name()]

    first = parts[0]
    last  = parts[-1]

    display_name  = raw.upper() if is_defendant else raw
    initials      = ".".join(p[0].upper() for p in parts if p) + "."
    title_surname = f"{title} {last}" if title else last
    short_name    = first

    forms = [display_name]
    if surface_forms >= 2:
        forms.append(initials)
    if surface_forms >= 3:
        forms.append(title_surname)
    if surface_forms >= 4:
        forms.append(short_name)

    dob       = fake.date_of_birth(minimum_age=30, maximum_age=65).strftime("%-d %B %Y")
    birthplace = fake.city()
    street     = fake.street_address()
    city_pc    = f"{fake.city()} {fake.postcode()}"

    return {
        "name":             display_name,
        "name_plain":       raw,
        "initials":         initials,
        "title_surname":    title_surname,
        "short_name":       short_name,
        "surface_forms_list": forms,
        "dob":              dob,
        "birthplace":       birthplace,
        "nationality":      NATIONALITY_ADJECTIVES.get(nat, nat),
        "role":             random.choice(PERSON_ROLES),
        "street":           street,
        "city_postcode":    city_pc,
        "address":          f"{street}, {city_pc}",
        "is_defendant":     is_defendant,
    }


def make_org(nat: str, nat_locales: dict, vat_prefixes: dict) -> dict:
    locale = nat_locales.get(nat, "en_GB")
    fake = Faker(locale)
    suffix  = random.choice(COMPANY_SUFFIXES)
    base    = re.sub(r"[,\.\-']", "", fake.company()).upper()
    street  = fake.street_address()
    city_pc = f"{fake.city()} {fake.postcode()}"
    vat_pfx = vat_prefixes.get(nat, nat)
    return {
        "name":          f"{base} {suffix}",
        "street":        street,
        "city_postcode": city_pc,
        "address":       f"{street}, {city_pc}",
        "vat":           make_vat(vat_pfx),
        "nationality":   nat,
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
        key for key, value in (
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

    first = parts[0]
    last = parts[-1]
    title = person.get("title", "")

    surface_forms_list = person.get("surface_forms_list") or [name]
    if not isinstance(surface_forms_list, list) or not surface_forms_list:
        raise ValueError(f"{context}.surface_forms_list must be a non-empty list")

    return {
        "name":               name,
        "name_plain":         raw_name,
        "initials":           person.get("initials") or make_initials(raw_name),
        "title_surname":      person.get("title_surname") or (f"{title} {last}".strip() if title else last),
        "short_name":         person.get("short_name") or first,
        "surface_forms_list": surface_forms_list,
        "dob":                dob,
        "birthplace":         birthplace,
        "nationality":        NATIONALITY_ADJECTIVES.get(nationality, nationality),
        "role":               role,
        "street":             street or split_address(address)[0],
        "city_postcode":      city_postcode or split_address(address)[1],
        "address":            address,
        "is_defendant":       is_defendant,
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
        key for key, value in (
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
        "name":          name,
        "street":        street,
        "city_postcode": city_postcode,
        "address":       address or f"{street}, {city_postcode}",
        "vat":           vat,
        "nationality":   org.get("nationality", ""),
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


def build_orgs_from_count(count: int, def_nats: list, nat_locales: dict, vat_prefixes: dict) -> list[dict]:
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
            normalize_person_record(person, True, f"case.defendants[{idx}]")
            for idx, person in enumerate(explicit_defendants)
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
            normalize_person_record(person, False, f"case.collateral[{idx}]")
            for idx, person in enumerate(explicit_collateral)
        ]
    else:
        collateral = build_people_from_specs(
            cast_cfg.get("collateral", []),
            nat_locales,
            False,
        )

    def_nats = [
        spec["nationality"]
        for spec in cast_cfg.get("defendants", [])
        if isinstance(spec, dict) and spec.get("nationality")
    ] or ["GB"]

    explicit_charged_orgs = case_cfg.get("charged_orgs")
    if isinstance(explicit_charged_orgs, list):
        charged_orgs = [
            normalize_org_record(org, f"case.charged_orgs[{idx}]")
            for idx, org in enumerate(explicit_charged_orgs)
        ]
    else:
        charged_orgs = build_orgs_from_count(
            cast_cfg.get("charged_orgs", 3),
            def_nats,
            nat_locales,
            vat_prefixes,
        )

    explicit_associated_orgs = case_cfg.get("associated_orgs")
    if isinstance(explicit_associated_orgs, list):
        associated_orgs = [
            normalize_org_record(org, f"case.associated_orgs[{idx}]")
            for idx, org in enumerate(explicit_associated_orgs)
        ]
    else:
        associated_orgs = build_orgs_from_count(
            cast_cfg.get("associated_orgs", 0),
            def_nats,
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
        start = offence_period_cfg.get("start")
        end = offence_period_cfg.get("end")
        if is_auto(start) and is_auto(end):
            offence_period = None
        elif isinstance(start, str) and isinstance(end, str) and start and end:
            offence_period = (start, end)
        else:
            raise ValueError(
                "case.metadata.offence_period must define non-empty start and end strings"
            )

    return {
        "court": metadata.get("court") if not is_auto(metadata.get("court")) else random.choice(COURTS),
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
    for idx, count in enumerate(counts_cfg):
        if not isinstance(count, dict):
            raise ValueError(f"case.counts[{idx}] must be a mapping")
        missing = [
            key for key in ("offence", "statute", "particulars")
            if not count.get(key)
        ]
        if missing:
            raise ValueError(
                f"case.counts[{idx}] is missing required fields: {', '.join(missing)}"
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
    defendants_str = " and ".join(p["name"] for p in defendants)
    companies_str  = " and ".join(o["name"] for o in orgs)

    counts = []
    for s in statutes:
        particulars = (
            s["particulars"]
            .replace("{defendants}",  defendants_str)
            .replace("{companies}",   companies_str)
            .replace("{start_date}",  start_date)
            .replace("{end_date}",    end_date)
            .strip()
        )
        counts.append({
            "offence":     s["offence"],
            "statute":     s["statute"],
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


def normalize_schema(
    schema_cfg: dict,
    doc_id: str,
    fraud_type: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
) -> dict:
    if not isinstance(schema_cfg, dict):
        raise ValueError("case.schema must be a mapping")

    derived_persons, derived_orgs = make_schema_nodes(
        defendants,
        collateral,
        charged_orgs,
        associated_orgs,
    )
    edges = schema_cfg.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("case.schema.edges must be a list")

    return {
        "doc_id":     doc_id,
        "fraud_type": fraud_type,
        "persons":    schema_cfg.get("persons", derived_persons),
        "orgs":       schema_cfg.get("orgs", derived_orgs),
        "edges":      edges,
    }


def resolve_prose_overrides(case_cfg: dict, doc_type: str) -> dict[str, str]:
    prose_cfg = case_cfg.get("prose")
    if is_auto(prose_cfg):
        return {}
    if not isinstance(prose_cfg, dict):
        raise ValueError("case.prose must be a mapping")

    section_order = list(SECTION_WEIGHTS[doc_type].keys())
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


# ── Ollama ─────────────────────────────────────────────────────────────────────

def call_ollama(ollama_cfg: dict, prompt: str) -> str | None:
    try:
        resp = requests.post(
            f"{ollama_cfg['base_url']}/api/generate",
            json={
                "model":  ollama_cfg["model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=ollama_cfg.get("timeout", 180),
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as exc:
        print(f"  [Ollama error] {exc}")
        return None


def generate_section(
    ollama_cfg: dict,
    section_name: str,
    doc_type: str,
    fraud_type: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
    case_number: str,
    word_target: int,
    entity_mentions: int,
    schema_context: str = "",
) -> str:
    all_persons = defendants + collateral
    all_orgs    = charged_orgs + associated_orgs

    persons_str = ", ".join(
        f"{p['name_plain']} ({p['initials']})" for p in all_persons
    )
    orgs_str = ", ".join(o["name"] for o in all_orgs)
    fraud_label = fraud_type.replace("_", " ")
    desc = SECTION_DESCRIPTIONS.get(section_name, section_name)

    chunks = []
    words_so_far = 0

    while words_so_far < word_target:
        remaining  = word_target - words_so_far
        this_chunk = min(CHUNK_SIZE, remaining)
        is_first   = (words_so_far == 0)

        context_fragment = ""
        if chunks:
            last = chunks[-1]
            context_fragment = (
                f"\nContinue directly from where the previous passage ended. "
                f"Last sentences: ...{last[-MAX_CONTEXT_CHARS:]}"
            )

        mention_instruction = (
            f"In this passage, mention each of the following persons at least "
            f"{entity_mentions} time(s): {persons_str}. "
            f"Mention each of the following companies at least {entity_mentions} time(s): {orgs_str}. "
            f"Refer to defendants both by full name and by initials in different sentences."
        )

        schema_block = f"\n\n{schema_context}" if schema_context else ""

        if is_first:
            prompt = (
                f"You are drafting part of a formal English {doc_type.replace('_', ' ')} "
                f"concerning {fraud_label} fraud in England and Wales.\n"
                f"Case reference: {case_number}\n"
                f"{schema_block}\n\n"
                f"Write the following section in plain, formal legal prose.\n"
                f"Section: {desc}\n\n"
                f"{mention_instruction}\n\n"
                f"Write approximately {this_chunk} words. "
                f"Do NOT include a section heading. Do NOT use markdown. "
                f"Do NOT add any preamble or thinking text."
            )
        else:
            prompt = (
                f"You are continuing a formal English {doc_type.replace('_', ' ')} "
                f"concerning {fraud_label} fraud. Case: {case_number}."
                f"{schema_block}{context_fragment}\n\n"
                f"{mention_instruction}\n\n"
                f"Write the next approximately {this_chunk} words of the same section "
                f"({desc}). Plain formal legal prose, no headings, no markdown, "
                f"no preamble."
            )

        print(f"    chunk ~{this_chunk}w ({words_so_far}/{word_target} done)…",
              end=" ", flush=True)
        raw = call_ollama(ollama_cfg, prompt)
        if raw is None:
            print("failed.")
            break

        # strip any <think>…</think> blocks that qwen3 emits
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        chunks.append(raw)
        words_so_far += len(raw.split())
        print(f"got {len(raw.split())}w")

    return "\n\n".join(chunks) if chunks else "[section not generated]"


# ── Counter ────────────────────────────────────────────────────────────────────

def doc_id_prefix(doc_type: str, fraud_type: str) -> str:
    return f"en_{doc_type}_{fraud_type}_"


def make_doc_id(doc_type: str, fraud_type: str, counter: int) -> str:
    return f"{doc_id_prefix(doc_type, fraud_type)}{counter:03d}"


def counter_from_doc_id(doc_id: str, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    if not isinstance(doc_id, str) or not doc_id.startswith(prefix):
        raise ValueError(
            f"Schema doc_id must start with '{prefix}', got {doc_id!r}"
        )

    suffix = doc_id[len(prefix):]
    if not suffix.isdigit():
        raise ValueError(f"Schema doc_id must end with digits, got {doc_id!r}")
    return int(suffix)


def next_counter(output_dir: Path, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    nums = [
        int(d.name.replace(prefix, ""))
        for d in output_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and d.name.replace(prefix, "").isdigit()
    ] if output_dir.exists() else []
    return (max(nums) + 1) if nums else 1


def build_section_word_targets(profile: dict, doc_type: str) -> dict[str, int]:
    section_order = list(SECTION_WEIGHTS[doc_type].keys())
    configured = profile.get("section_words")

    if configured is not None:
        missing = [name for name in section_order if name not in configured]
        extra = [name for name in configured if name not in section_order]
        invalid = [
            name for name in section_order
            if name in configured
            and (not isinstance(configured[name], int) or configured[name] <= 0)
        ]

        problems = []
        if missing:
            problems.append(f"missing keys: {', '.join(missing)}")
        if extra:
            problems.append(f"unknown keys: {', '.join(extra)}")
        if invalid:
            problems.append(f"non-positive integer values: {', '.join(invalid)}")
        if problems:
            raise ValueError(
                "Invalid profile.section_words for "
                f"{doc_type}: {'; '.join(problems)}"
            )

        return {name: configured[name] for name in section_order}

    pages = profile.get("pages")
    if not isinstance(pages, int) or pages <= 0:
        raise ValueError(
            "Profile must define either valid section_words or a positive integer pages value"
        )

    total_words = pages * WORDS_PER_PAGE
    prose_words = max(300, total_words - 200)
    weights = SECTION_WEIGHTS[doc_type]
    return {
        name: max(100, math.floor(prose_words * weights[name]))
        for name in section_order
    }


def resolve_documents_to_generate(profile: dict, cli_documents: int | None) -> int:
    if cli_documents is not None:
        if cli_documents <= 0:
            raise ValueError("--documents must be a positive integer")
        return cli_documents

    configured = profile.get("documents", profile.get("count"))
    if not isinstance(configured, int) or configured <= 0:
        raise ValueError(
            "Profile must define a positive integer documents value"
        )
    return configured


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Ollama; insert placeholders (for testing)")
    parser.add_argument("--documents", "--count", dest="documents", type=int, default=None,
                        help="Override profile.documents")
    parser.add_argument("--doc-type", choices=sorted(SECTION_WEIGHTS.keys()), metavar="TYPE",
                        help="Override profile.doc_type")
    parser.add_argument("--fraud-type", metavar="TYPE",
                        help="Override profile.fraud_type")
    parser.add_argument("--from-schema", metavar="PATH",
                        help="Load an existing case_schema.json instead of "
                             "auto-generating one (cast is still read from config)")
    args = parser.parse_args()

    cfg        = load_config()
    profile    = cfg["profile"]
    case_cfg   = cfg.get("case") or {}
    if not isinstance(case_cfg, dict):
        raise SystemExit("Top-level case section must be a mapping")
    ollama_cfg = cfg["ollama"]
    nat_locales  = cfg["nationality_locales"]
    vat_prefixes = cfg["vat_prefixes"]
    schema_dir = Path(cfg.get("schema_dir", "schemas"))

    doc_type   = args.doc_type if args.doc_type is not None else profile["doc_type"]
    fraud_type = args.fraud_type if args.fraud_type is not None else profile["fraud_type"]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(exist_ok=True)

    env = Environment(
        loader=FileSystemLoader("templates"),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    sections    = EN_SECTIONS[doc_type]
    labels      = EN_LABELS
    try:
        section_word_targets = build_section_word_targets(profile, doc_type)
        documents = resolve_documents_to_generate(profile, args.documents)
        prose_overrides = resolve_prose_overrides(case_cfg, doc_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for doc_idx in range(documents):
        total_prose_words = sum(section_word_targets.values())
        if profile.get("section_words") is not None:
            size_label = f"{total_prose_words}w prose"
        else:
            size_label = f"{profile['pages']}p target (~{total_prose_words}w prose)"
        print(
            f"\n[{doc_idx+1}/{documents}] Generating {doc_type} / {fraud_type} / {size_label} …"
        )

        try:
            defendants, collateral, charged_orgs, associated_orgs = resolve_case_entities(
                profile,
                case_cfg,
                nat_locales,
                vat_prefixes,
            )

            metadata = resolve_case_metadata(case_cfg, doc_type)
            court = metadata["court"]
            case_number = metadata["case_number"]
            cross_ref = metadata["cross_ref"]
            filing_date = metadata["filing_date"]

            counts_list = resolve_counts(
                cfg,
                case_cfg,
                doc_type,
                fraud_type,
                defendants,
                charged_orgs,
                metadata["offence_period"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        if args.from_schema:
            loaded_schema = load_case_schema(Path(args.from_schema))
            try:
                source_counter = counter_from_doc_id(
                    loaded_schema.get("doc_id"),
                    doc_type,
                    fraud_type,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

            doc_id = make_doc_id(doc_type, fraud_type, source_counter + 1)
            try:
                schema = normalize_schema(
                    loaded_schema,
                    doc_id,
                    fraud_type,
                    defendants,
                    collateral,
                    charged_orgs,
                    associated_orgs,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"  Schema  : loaded from {args.from_schema} → {doc_id}")
        else:
            # ── Build case schema (in memory — folder not created yet) ──
            # doc_id is needed for the schema but the folder is only written after
            # generation succeeds, so a crashed run leaves no empty folder.
            counter = next_counter(output_dir, doc_type, fraud_type)
            doc_id  = make_doc_id(doc_type, fraud_type, counter)
            try:
                if is_auto(case_cfg.get("schema")):
                    schema = make_case_schema(
                        doc_id, fraud_type,
                        defendants, collateral,
                        charged_orgs, associated_orgs,
                    )
                    print(f"  Schema  : {len(schema['edges'])} edges (auto)")
                else:
                    schema = normalize_schema(
                        case_cfg["schema"],
                        doc_id,
                        fraud_type,
                        defendants,
                        collateral,
                        charged_orgs,
                        associated_orgs,
                    )
                    print(f"  Schema  : {len(schema['edges'])} edges (from config)")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

        schema_context = schema_to_context(schema)

        # ── Generate prose sections ──
        llm_sections = []
        entity_density = 1.5  # default; could be added to profile later
        for section_name, section_words in section_word_targets.items():
            if section_name in prose_overrides:
                text = prose_overrides[section_name]
                print(
                    f"  section '{section_name}': using config prose "
                    f"({len(text.split())}w)"
                )
            elif args.no_llm:
                text = "[placeholder prose]"
                print(f"  section '{section_name}': placeholder prose")
            else:
                entity_mentions = max(1, math.ceil(
                    entity_density * section_words / WORDS_PER_PAGE
                ))
                print(f"  section '{section_name}': ~{section_words}w, "
                      f"{entity_mentions} mention(s)/entity")
                text = generate_section(
                    ollama_cfg     = ollama_cfg,
                    section_name   = section_name,
                    doc_type       = doc_type,
                    fraud_type     = fraud_type,
                    defendants     = defendants,
                    collateral     = collateral,
                    charged_orgs   = charged_orgs,
                    associated_orgs= associated_orgs,
                    case_number    = case_number,
                    word_target    = section_words,
                    entity_mentions= entity_mentions,
                    schema_context = schema_context,
                )
            llm_sections.append(text)

        # ── Render template ──
        template = env.get_template(f"en_{doc_type}.j2")
        text = template.render(
            prosecution  = PROSECUTION,
            court        = court,
            sections     = sections,
            labels       = labels,
            case_number  = case_number,
            cross_ref    = cross_ref,
            filing_date  = filing_date,
            persons      = defendants,
            orgs         = charged_orgs,
            counts       = counts_list,
            llm_sections = llm_sections,
        )

        # ── Save to per-document folder (created here, after success) ──
        doc_dir  = output_dir / doc_id
        schema_path = schema_dir / f"{doc_id}.json"
        if args.from_schema and doc_dir.exists():
            raise SystemExit(
                f"Target output folder already exists for schema-derived run: {doc_dir}"
            )
        if args.from_schema and schema_path.exists():
            raise SystemExit(
                f"Target schema file already exists for schema-derived run: {schema_path}"
            )
        doc_dir.mkdir(exist_ok=True)

        write_case_schema(schema_path, schema)

        txt_path = doc_dir / f"{doc_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        actual_words = len(text.split())
        actual_pages = round(actual_words / WORDS_PER_PAGE, 1)

        # ── Ground truth ──
        gt_rows = []

        # defendants — all surface forms
        for p in defendants:
            for form in p["surface_forms_list"]:
                gt_rows.append((doc_id, form, "PERSON", "yes", "defendant surface form"))
            gt_rows.append((doc_id, p["address"], "LOCATION", "yes", "defendant address"))

        # collateral — full name only
        for p in collateral:
            gt_rows.append((doc_id, p["name"], "PERSON", "yes", "collateral person"))

        # charged orgs — name + address parts
        for o in charged_orgs:
            gt_rows.append((doc_id, o["name"],    "ORG",      "yes", "charged org"))
            gt_rows.append((doc_id, o["street"],  "LOCATION", "yes", "org street"))
            gt_rows.append((doc_id, o["city_postcode"], "LOCATION", "yes", "org city/postcode"))

        # associated orgs — name only (prose mentions)
        for o in associated_orgs:
            gt_rows.append((doc_id, o["name"], "ORG", "yes", "associated org"))

        # negative controls
        gt_rows.append((doc_id, PROSECUTION, "NEGATIVE_CONTROL", "no", "prosecution"))
        gt_rows.append((doc_id, court,       "NEGATIVE_CONTROL", "no", "court"))

        gt_path = doc_dir / "groundtruth.tsv"
        write_groundtruth(gt_path, gt_rows)

        print(f"  Schema : {schema_path}")
        print(f"  Saved  : {txt_path}  ({actual_words}w ≈ {actual_pages} pages)")
        print(f"  GT rows: {len(gt_rows)}  →  {gt_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
