"""Core document generation engine."""

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
    resolve_scenario_brief,
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
from src.synthetic_ner.tasks.groundtruth import write_document_reference_artifacts
from src.synthetic_ner.types.app_config import ProfileConfig
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext
from src.synthetic_ner.utils import (
    is_auto,
    resolve_project_path,
)


def build_section_word_targets(
    profile: ProfileConfig,
) -> dict[str, int]:
    configured = profile.section_words
    section_order = list(configured)

    invalid = [
        name
        for name in section_order
        if name in configured and (not isinstance(configured[name], int) or configured[name] <= 0)
    ]

    problems = []
    if invalid:
        problems.append(f"non-positive integer values: {', '.join(invalid)}")
    if not section_order:
        problems.append("at least one section is required")
    if problems:
        raise ValueError(f"Invalid profile.section_words: {'; '.join(problems)}")

    return {name: configured[name] for name in section_order}


def resolve_documents_to_generate(profile: ProfileConfig) -> int:
    return profile.documents


def build_template_environment(template_path: Path) -> Environment:
    # Templates produce plain-text legal documents, never HTML.
    return Environment(  # nosec B701
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
        resolve_project_path(project_root, args.from_schema) if args.from_schema else None
    )

    return RuntimeContext(
        project_root=project_root,
        app_config=app_config,
        paths=app_config.paths,
        generation_cfg=app_config.generation,
        profile=profile,
        case_cfg=app_config.case,
        mlflow_cfg=app_config.mlflow,
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
            metadata=metadata,
        )
        scenario_brief = resolve_scenario_brief(
            context.case_cfg,
            metadata,
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
        scenario_brief=scenario_brief,
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
        problems.append(f"expected {len(section_names)} sections, got {len(section_texts)}")

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
        raise SystemExit(f"Target output folder already exists for schema-derived run: {doc_dir}")
    if context.schema_source_path and schema_path.exists():
        raise SystemExit(f"Target schema file already exists for schema-derived run: {schema_path}")


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

    reference_path, manifest_path = write_document_reference_artifacts(
        doc_dir=doc_dir,
        doc_id=doc_id,
        document=document,
        document_path=txt_path,
        address_surface_forms=context.case_cfg.cast.address_surface_forms,
    )

    actual_words = len(rendered_text.split())
    actual_pages = round(actual_words / context.generation_cfg.words_per_page, 1)
    print(f"  Schema : {schema_path}")
    print(f"  Saved  : {txt_path}  ({actual_words}w ≈ {actual_pages} pages)")
    print(f"  GT refs : {reference_path}")
    print(f"  Manifest: {manifest_path}")
