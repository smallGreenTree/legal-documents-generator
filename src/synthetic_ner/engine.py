"""Core document generation engine."""

import re
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.synthetic_ner.case import (
    build_amounts,
    resolve_case_entities,
    resolve_case_metadata,
    resolve_counts,
    resolve_prose_overrides,
)
from src.synthetic_ner.config import load_app_config
from src.synthetic_ner.constants import (
    EN_LABELS,
    EN_SECTIONS,
    INCOMPLETE_SECTION_MARKERS,
    PROSECUTION,
)
from src.synthetic_ner.schema import (
    counter_from_doc_id,
    load_case_schema,
    make_case_schema,
    make_doc_id,
    next_counter,
    normalize_schema,
    write_case_schema,
)
from src.synthetic_ner.types.app_config import ProfileConfig
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext
from src.synthetic_ner.utils import (
    is_auto,
    make_initials,
    resolve_project_path,
    write_groundtruth,
)

_AMOUNT_RE = re.compile(
    r"(?:£|€|\b(?:GBP|EUR)\s*)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|m|thousand|k))?",
    re.IGNORECASE,
)



def build_section_word_targets(
    profile: ProfileConfig,
) -> dict[str, int]:
    configured = profile.section_words
    section_order = list(configured)

    invalid = [
        name
        for name in section_order
        if name in configured
        and (not isinstance(configured[name], int) or configured[name] <= 0)
    ]

    problems = []
    if invalid:
        problems.append(f"non-positive integer values: {', '.join(invalid)}")
    if not section_order:
        problems.append("at least one section is required")
    if problems:
        raise ValueError(
            f"Invalid profile.section_words: {'; '.join(problems)}"
        )

    return {name: configured[name] for name in section_order}


def resolve_documents_to_generate(profile: ProfileConfig) -> int:
    return profile.documents


def build_groundtruth_rows(
    doc_id: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
    metadata: dict,
    counts_list: list[dict],
    amounts: dict | None = None,
    address_surface_forms: int = 3,
) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    all_people = defendants + collateral
    all_orgs = charged_orgs + associated_orgs

    _append_person_rows(rows, doc_id, all_people)
    _append_org_rows(rows, doc_id, all_orgs, charged_orgs)
    _append_reference_rows(rows, doc_id, metadata)
    _append_date_rows(rows, doc_id, metadata, all_people)
    _append_amount_rows(rows, doc_id, counts_list, amounts or {})
    _append_initial_rows(rows, doc_id, all_people)
    _append_title_rows(rows, doc_id, all_people)
    _append_all_address_rows(rows, doc_id, defendants, all_orgs, address_surface_forms)
    _append_vat_rows(rows, doc_id, all_orgs)
    _append_row(rows, doc_id, PROSECUTION, "NEGATIVE_CONTROL", "no", "prosecution")
    _append_row(rows, doc_id, metadata["court"], "NEGATIVE_CONTROL", "no", "court")
    return rows


def filter_groundtruth_rows_for_rendered_text(
    rows: list[tuple[str, str, str, str, str]],
    rendered_text: str,
) -> list[tuple[str, str, str, str, str]]:
    searchable_text = _normalize_groundtruth_surface(rendered_text)
    return [
        row
        for row in rows
        if _normalize_groundtruth_surface(row[1]) in searchable_text
    ]


def _normalize_groundtruth_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _append_person_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    people: list[dict],
) -> None:
    for person in people:
        _append_row(rows, doc_id, person["name"], "PERSON", "yes", _person_note(person))


def _append_org_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    orgs: list[dict],
    charged_orgs: list[dict],
) -> None:
    for org in orgs:
        _append_row(rows, doc_id, org["name"], "ORG", "yes", _org_note(org, charged_orgs))


def _append_reference_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    metadata: dict,
) -> None:
    _append_row(rows, doc_id, metadata["case_number"], "CASE_REFERENCE", "yes", "case number")
    _append_row(rows, doc_id, metadata["cross_ref"], "CASE_REFERENCE", "yes", "cross reference")


def _append_date_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    metadata: dict,
    people: list[dict],
) -> None:
    _append_row(rows, doc_id, metadata["filing_date"], "DATE", "yes", "filing date")
    offence_period = metadata.get("offence_period")
    if offence_period:
        _append_row(rows, doc_id, offence_period[0], "DATE", "yes", "offence period start")
        _append_row(rows, doc_id, offence_period[1], "DATE", "yes", "offence period end")
    for person in people:
        _append_row(
            rows,
            doc_id,
            person.get("dob"),
            "DATE",
            "yes",
            f"{person['name']} date of birth",
        )


def _append_amount_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    counts_list: list[dict],
    amounts: dict,
) -> None:
    seen: set[str] = set()
    for amount in _extract_count_values(counts_list, _AMOUNT_RE):
        if amount not in seen:
            _append_row(rows, doc_id, amount, "AMOUNT", "yes", "amount in count particulars")
            seen.add(amount)
    for label, amount in _amount_values(amounts):
        if amount not in seen:
            _append_row(rows, doc_id, amount, "AMOUNT", "yes", label)
            seen.add(amount)


def _amount_values(amounts: dict) -> list[tuple[str, str]]:
    values = []
    total_loss = amounts.get("total_loss")
    if total_loss:
        values.append(("total alleged loss", total_loss))
    invoice_value = amounts.get("inflated_invoice_value")
    if invoice_value:
        values.append(("inflated invoice value", invoice_value))
    for transfer in amounts.get("transfers", []):
        if not isinstance(transfer, dict) or not transfer.get("amount"):
            continue
        values.append(
            (
                f"transfer from {transfer.get('from')} to {transfer.get('to')}",
                transfer["amount"],
            )
        )
    return values


def _append_initial_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    people: list[dict],
) -> None:
    for person in people:
        _append_row(
            rows,
            doc_id,
            person.get("initials") or make_initials(person["name"]),
            "INITIAL",
            "yes",
            f"{person['name']} initials",
        )


def _append_title_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    people: list[dict],
) -> None:
    for person in people:
        title_surname = person.get("title_surname")
        if title_surname and title_surname != person["name"].split()[-1]:
            _append_row(rows, doc_id, title_surname, "TITLE", "yes", f"{person['name']} title")


def _append_all_address_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    defendants: list[dict],
    orgs: list[dict],
    address_surface_forms: int,
) -> None:
    for person in defendants:
        _append_address_rows(rows, doc_id, person, "defendant", address_surface_forms)
    for org in orgs:
        _append_address_rows(rows, doc_id, org, "organisation", address_surface_forms)


def _append_vat_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    orgs: list[dict],
) -> None:
    for org in orgs:
        _append_row(rows, doc_id, org["vat"], "VAT", "yes", f"{org['name']} VAT number")


def _append_row(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    value: str | None,
    label: str,
    should_propose: str,
    notes: str,
) -> None:
    if value:
        rows.append((doc_id, value, label, should_propose, notes))


def _person_note(person: dict) -> str:
    return "defendant person" if person.get("is_defendant") else "collateral person"


def _org_note(org: dict, charged_orgs: list[dict]) -> str:
    return "charged organisation" if org in charged_orgs else "associated organisation"


def _extract_count_values(counts_list: list[dict], pattern: re.Pattern[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for count in counts_list:
        for match in pattern.findall(count.get("particulars", "")):
            normalized = str(match).strip().rstrip(".,;:")
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
    return values


def _append_address_rows(
    rows: list[tuple[str, str, str, str, str]],
    doc_id: str,
    record: dict,
    owner_type: str,
    address_surface_forms: int,
) -> None:
    owner = record["name"]
    if address_surface_forms >= 1:
        _append_row(
            rows,
            doc_id,
            record.get("address"),
            "ADDRESS",
            "yes",
            f"{owner} full address",
        )
    if address_surface_forms >= 2:
        _append_row(
            rows,
            doc_id,
            record.get("street"),
            "ADDRESS",
            "yes",
            f"{owner_type} building/street identifier for {owner}",
        )
    if address_surface_forms >= 3:
        _append_row(
            rows,
            doc_id,
            record.get("city_postcode"),
            "ADDRESS",
            "yes",
            f"{owner_type} city/postcode for {owner}",
        )


def build_template_environment(template_path: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def build_section_labels(doc_type: str, section_order: list[str]) -> dict[str, str]:
    configured = EN_SECTIONS.get(doc_type)
    if configured is not None:
        return dict(configured)

    labels = {"title": doc_type.replace("_", " ").upper()}
    for index, section_name in enumerate(section_order, start=1):
        labels[f"section_{section_name}"] = (
            f"SECTION {index} - {section_name.replace('_', ' ').upper()}"
        )
    return labels


def build_runtime_context(args: Namespace, project_root: Path) -> RuntimeContext:
    case_config_path = resolve_project_path(project_root, args.case_config)
    if not args.template:
        raise SystemExit("--template is required")
    template_path = resolve_project_path(project_root, args.template)
    try:
        app_config = load_app_config(
            project_root / "config.yaml",
            case_config_path,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    profile = app_config.profile
    if args.documents is not None:
        if args.documents <= 0:
            raise SystemExit("--documents must be a positive integer")
        profile = replace(profile, documents=args.documents)

    if args.doc_type is not None:
        profile = replace(profile, doc_type=args.doc_type)
    if args.fraud_type is not None:
        profile = replace(profile, fraud_type=args.fraud_type)

    doc_type = profile.doc_type
    fraud_type = profile.fraud_type

    output_dir = resolve_project_path(project_root, app_config.paths.output_dir)
    schema_dir = resolve_project_path(project_root, app_config.paths.schema_dir)
    memory_dir = resolve_project_path(project_root, app_config.paths.memory_dir)
    output_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(exist_ok=True)
    memory_dir.mkdir(exist_ok=True)

    try:
        section_word_targets = build_section_word_targets(
            profile,
        )
        section_order = list(section_word_targets)
        documents = resolve_documents_to_generate(profile)
        prose_overrides = resolve_prose_overrides(
            app_config.case,
            section_order,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    schema_source_path = (
        resolve_project_path(project_root, args.from_schema)
        if args.from_schema
        else None
    )

    return RuntimeContext(
        project_root=project_root,
        app_config=app_config,
        paths=app_config.paths,
        generation_cfg=app_config.generation,
        profile=profile,
        case_cfg=app_config.case,
        langfuse_cfg=app_config.langfuse,
        model_routing_cfg=app_config.model_routing,
        workflow_cfg=app_config.workflow,
        nat_locales=app_config.nationality_locales,
        vat_prefixes=app_config.vat_prefixes,
        doc_type=doc_type,
        fraud_type=fraud_type,
        output_dir=output_dir,
        schema_dir=schema_dir,
        memory_dir=memory_dir,
        template_path=template_path,
        template_env=build_template_environment(template_path),
        template_name=template_path.name,
        sections=build_section_labels(doc_type, section_order),
        labels=EN_LABELS,
        section_word_targets=section_word_targets,
        documents=documents,
        prose_overrides=prose_overrides,
        schema_source_path=schema_source_path,
    )


def build_size_label(context: RuntimeContext) -> str:
    total_prose_words = sum(context.section_word_targets.values())
    return f"{total_prose_words}w prose"


def resolve_document_inputs(context: RuntimeContext) -> DocumentInputs:
    try:
        defendants, collateral, charged_orgs, associated_orgs = resolve_case_entities(
            context.case_cfg,
            context.nat_locales,
            context.vat_prefixes,
            context.app_config.entity_variants.persons,
        )
        metadata = resolve_case_metadata(context.case_cfg, context.doc_type)
        amounts = build_amounts(charged_orgs, associated_orgs)
        counts_list = resolve_counts(
            context.app_config.fraud_statutes,
            context.case_cfg,
            context.doc_type,
            context.fraud_type,
            defendants,
            charged_orgs,
            amounts,
            metadata["offence_period"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    return DocumentInputs(
        defendants=defendants,
        collateral=collateral,
        charged_orgs=charged_orgs,
        associated_orgs=associated_orgs,
        metadata=metadata,
        amounts=amounts,
        counts_list=counts_list,
        evidence_categories=context.case_cfg.evidence_categories,
    )


def resolve_schema_for_document(
    context: RuntimeContext,
    document: DocumentInputs,
    document_index: int,
    doc_id_override: str | None = None,
) -> tuple[str, dict]:
    if context.schema_source_path:
        loaded_schema = load_case_schema(context.schema_source_path)
        try:
            if doc_id_override is None:
                source_counter = counter_from_doc_id(
                    loaded_schema.get("doc_id"),
                    context.doc_type,
                    context.fraud_type,
                )
                doc_id = make_doc_id(
                    context.doc_type,
                    context.fraud_type,
                    source_counter + document_index + 1,
                )
            else:
                counter_from_doc_id(
                    doc_id_override,
                    context.doc_type,
                    context.fraud_type,
                )
                doc_id = doc_id_override
            schema = normalize_schema(
                loaded_schema,
                doc_id,
                context.fraud_type,
                document.defendants,
                document.collateral,
                document.charged_orgs,
                document.associated_orgs,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"  Schema  : loaded from {context.schema_source_path} → {doc_id}")
        return doc_id, schema

    if doc_id_override is None:
        counter = next_counter(context.output_dir, context.doc_type, context.fraud_type)
        doc_id = make_doc_id(context.doc_type, context.fraud_type, counter)
    else:
        try:
            counter_from_doc_id(doc_id_override, context.doc_type, context.fraud_type)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        doc_id = doc_id_override
    try:
        if is_auto(context.case_cfg.schema):
            schema = make_case_schema(
                doc_id,
                context.fraud_type,
                document.defendants,
                document.collateral,
                document.charged_orgs,
                document.associated_orgs,
            )
            print(f"  Schema  : {len(schema['edges'])} edges (auto)")
        else:
            schema = normalize_schema(
                context.case_cfg.schema,
                doc_id,
                context.fraud_type,
                document.defendants,
                document.collateral,
                document.charged_orgs,
                document.associated_orgs,
            )
            print(f"  Schema  : {len(schema['edges'])} edges (from config)")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return doc_id, schema



def collect_section_output_problems(
    section_targets: dict[str, int],
    section_texts: list[str],
    min_completion_ratio: float = 0.7,
) -> list[str]:
    problems = []
    section_names = list(section_targets.keys())

    if len(section_texts) != len(section_names):
        problems.append(
            f"expected {len(section_names)} sections, got {len(section_texts)}"
        )

    for index, section_name in enumerate(section_names):
        if index >= len(section_texts):
            problems.append(f"section '{section_name}' is missing")
            continue

        text = section_texts[index].strip()
        if not text:
            problems.append(f"section '{section_name}' is empty")
            continue
        if text in INCOMPLETE_SECTION_MARKERS:
            problems.append(f"section '{section_name}' is incomplete: {text}")
            continue

        minimum_words = max(60, int(section_targets[section_name] * min_completion_ratio))
        if len(text.split()) < minimum_words:
            problems.append(
                f"section '{section_name}' is too short for its target "
                f"({len(text.split())}w < {minimum_words}w minimum)"
            )

    return problems


def render_document_text(
    context: RuntimeContext,
    document: DocumentInputs,
    llm_sections: list[str],
) -> str:
    template = context.template_env.get_template(context.template_name)
    metadata = document.metadata
    return template.render(
        prosecution=PROSECUTION,
        court=metadata["court"],
        sections=context.sections,
        labels=context.labels,
        case_number=metadata["case_number"],
        cross_ref=metadata["cross_ref"],
        filing_date=metadata["filing_date"],
        persons=document.defendants,
        orgs=document.charged_orgs,
        counts=document.counts_list,
        llm_sections=llm_sections,
    )


def ensure_target_paths_available(
    context: RuntimeContext,
    doc_dir: Path,
    schema_path: Path,
) -> None:
    if context.schema_source_path and doc_dir.exists():
        raise SystemExit(
            f"Target output folder already exists for schema-derived run: {doc_dir}"
        )
    if context.schema_source_path and schema_path.exists():
        raise SystemExit(
            f"Target schema file already exists for schema-derived run: {schema_path}"
        )


def save_document_artifacts(
    context: RuntimeContext,
    document: DocumentInputs,
    doc_id: str,
    schema: dict,
    rendered_text: str,
) -> None:
    doc_dir = context.output_dir / doc_id
    schema_path = context.schema_dir / f"{doc_id}.json"
    ensure_target_paths_available(context, doc_dir, schema_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    write_case_schema(schema_path, schema)

    txt_path = doc_dir / f"{doc_id}.txt"
    txt_path.write_text(rendered_text, encoding="utf-8")

    candidate_gt_rows = build_groundtruth_rows(
        doc_id,
        document.defendants,
        document.collateral,
        document.charged_orgs,
        document.associated_orgs,
        document.metadata,
        document.counts_list,
        document.amounts,
        context.case_cfg.cast.address_surface_forms,
    )
    gt_rows = filter_groundtruth_rows_for_rendered_text(candidate_gt_rows, rendered_text)
    gt_path = doc_dir / "groundtruth.tsv"
    write_groundtruth(gt_path, gt_rows)

    actual_words = len(rendered_text.split())
    actual_pages = round(actual_words / context.generation_cfg.words_per_page, 1)
    print(f"  Schema : {schema_path}")
    print(f"  Saved  : {txt_path}  ({actual_words}w ≈ {actual_pages} pages)")
    removed_gt_rows = len(candidate_gt_rows) - len(gt_rows)
    print(f"  GT rows: {len(gt_rows)}  →  {gt_path}")
    if removed_gt_rows:
        print(f"  GT trim: removed {removed_gt_rows} row(s) absent from rendered text")
