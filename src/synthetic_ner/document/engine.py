"""Build, render, validate, and persist generated documents."""

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.synthetic_ner.case_generation.case import (
    build_amounts,
    resolve_case_entities,
    resolve_case_metadata,
    resolve_counts,
    resolve_prose_overrides,
    resolve_scenario_brief,
)
from src.synthetic_ner.configuration.loader import load_app_config
from src.synthetic_ner.core.constants import (
    EN_LABELS,
    EN_SECTIONS,
    INCOMPLETE_SECTION_MARKERS,
    PROSECUTION,
)
from src.synthetic_ner.core.paths import resolve_project_path
from src.synthetic_ner.document.inputs import write_document_inputs
from src.synthetic_ner.types.app_config import ProfileConfig
from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME, DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext


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
    memory_dir = resolve_project_path(project_root, app_config.paths.memory_dir)
    output_dir.mkdir(exist_ok=True)
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
        memory_dir=memory_dir,
        template_path=template_path,
        template_env=build_template_environment(template_path),
        template_name=template_path.name,
        sections=build_section_labels(doc_type, section_order),
        labels=EN_LABELS,
        section_word_targets=section_word_targets,
        documents=documents,
        prose_overrides=prose_overrides,
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


def save_document_artifacts(
    context: RuntimeContext,
    document: DocumentInputs,
    doc_id: str,
    rendered_text: str,
) -> None:
    doc_dir = context.output_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    txt_path = doc_dir / f"{doc_id}.txt"
    txt_path.write_text(rendered_text, encoding="utf-8")
    inputs_path = write_document_inputs(doc_dir / DOCUMENT_INPUTS_FILENAME, document)

    actual_words = len(rendered_text.split())
    actual_pages = round(actual_words / context.generation_cfg.words_per_page, 1)
    print(f"  Saved  : {txt_path}  ({actual_words}w ≈ {actual_pages} pages)")
    print(f"  Inputs : {inputs_path}")
