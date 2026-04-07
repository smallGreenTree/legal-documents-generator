"""Core document generation engine."""

import math
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import requests
from jinja2 import Environment, FileSystemLoader

from .case import (
    resolve_case_entities,
    resolve_case_metadata,
    resolve_counts,
    resolve_prose_overrides,
)
from .constants import (
    CHUNK_SIZE,
    EN_LABELS,
    EN_SECTIONS,
    MAX_CONTEXT_CHARS,
    PROSECUTION,
    SECTION_DESCRIPTIONS,
    SECTION_WEIGHTS,
    WORDS_PER_PAGE,
)
from .schema import (
    counter_from_doc_id,
    load_case_schema,
    make_case_schema,
    make_doc_id,
    next_counter,
    normalize_schema,
    schema_to_context,
    write_case_schema,
)
from .utils import is_auto, load_config, resolve_project_path, write_groundtruth


@dataclass
class RuntimeContext:
    cfg: dict
    profile: dict
    case_cfg: dict
    ollama_cfg: dict
    nat_locales: dict
    vat_prefixes: dict
    doc_type: str
    fraud_type: str
    output_dir: Path
    schema_dir: Path
    template_env: Environment
    sections: dict
    labels: dict
    section_word_targets: dict[str, int]
    documents: int
    prose_overrides: dict[str, str]
    schema_source_path: Path | None
    no_llm: bool


@dataclass
class DocumentInputs:
    defendants: list[dict]
    collateral: list[dict]
    charged_orgs: list[dict]
    associated_orgs: list[dict]
    metadata: dict
    counts_list: list[dict]


def call_ollama(ollama_cfg: dict, prompt: str) -> str | None:
    try:
        response = requests.post(
            f"{ollama_cfg['base_url']}/api/generate",
            json={
                "model": ollama_cfg["model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=ollama_cfg.get("timeout", 180),
        )
        response.raise_for_status()
        return response.json()["response"].strip()
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
    all_orgs = charged_orgs + associated_orgs

    persons_str = ", ".join(
        f"{person['name_plain']} ({person['initials']})"
        for person in all_persons
    )
    orgs_str = ", ".join(org["name"] for org in all_orgs)
    fraud_label = fraud_type.replace("_", " ")
    description = SECTION_DESCRIPTIONS.get(section_name, section_name)

    chunks = []
    words_so_far = 0

    while words_so_far < word_target:
        remaining = word_target - words_so_far
        chunk_target = min(CHUNK_SIZE, remaining)
        is_first_chunk = words_so_far == 0

        context_fragment = ""
        if chunks:
            last_chunk = chunks[-1]
            context_fragment = (
                "\nContinue directly from where the previous passage ended. "
                f"Last sentences: ...{last_chunk[-MAX_CONTEXT_CHARS:]}"
            )

        mention_instruction = (
            "In this passage, mention each of the following persons at least "
            f"{entity_mentions} time(s): {persons_str}. "
            "Mention each of the following companies at least "
            f"{entity_mentions} time(s): {orgs_str}. "
            "Refer to defendants both by full name and by initials in different sentences."
        )

        schema_block = f"\n\n{schema_context}" if schema_context else ""

        if is_first_chunk:
            prompt = (
                f"You are drafting part of a formal English {doc_type.replace('_', ' ')} "
                f"concerning {fraud_label} fraud in England and Wales.\n"
                f"Case reference: {case_number}\n"
                f"{schema_block}\n\n"
                "Write the following section in plain, formal legal prose.\n"
                f"Section: {description}\n\n"
                f"{mention_instruction}\n\n"
                f"Write approximately {chunk_target} words. "
                "Do NOT include a section heading. Do NOT use markdown. "
                "Do NOT add any preamble or thinking text."
            )
        else:
            prompt = (
                f"You are continuing a formal English {doc_type.replace('_', ' ')} "
                f"concerning {fraud_label} fraud. Case: {case_number}."
                f"{schema_block}{context_fragment}\n\n"
                f"{mention_instruction}\n\n"
                f"Write the next approximately {chunk_target} words of the same section "
                f"({description}). Plain formal legal prose, no headings, no markdown, "
                "no preamble."
            )

        print(
            f"    chunk ~{chunk_target}w ({words_so_far}/{word_target} done)…",
            end=" ",
            flush=True,
        )
        raw = call_ollama(ollama_cfg, prompt)
        if raw is None:
            print("failed.")
            break

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        chunks.append(raw)
        words_so_far += len(raw.split())
        print(f"got {len(raw.split())}w")

    return "\n\n".join(chunks) if chunks else "[section not generated]"


def build_section_word_targets(profile: dict, doc_type: str) -> dict[str, int]:
    section_order = list(SECTION_WEIGHTS[doc_type].keys())
    configured = profile.get("section_words")

    if configured is not None:
        missing = [name for name in section_order if name not in configured]
        extra = [name for name in configured if name not in section_order]
        invalid = [
            name
            for name in section_order
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
                f"Invalid profile.section_words for {doc_type}: {'; '.join(problems)}"
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
        raise ValueError("Profile must define a positive integer documents value")
    return configured


def build_groundtruth_rows(
    doc_id: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
    court: str,
) -> list[tuple[str, str, str, str, str]]:
    rows = []

    for person in defendants:
        for form in person["surface_forms_list"]:
            rows.append((doc_id, form, "PERSON", "yes", "defendant surface form"))
        rows.append((doc_id, person["address"], "LOCATION", "yes", "defendant address"))

    for person in collateral:
        rows.append((doc_id, person["name"], "PERSON", "yes", "collateral person"))

    for org in charged_orgs:
        rows.append((doc_id, org["name"], "ORG", "yes", "charged org"))
        rows.append((doc_id, org["street"], "LOCATION", "yes", "org street"))
        rows.append((doc_id, org["city_postcode"], "LOCATION", "yes", "org city/postcode"))

    for org in associated_orgs:
        rows.append((doc_id, org["name"], "ORG", "yes", "associated org"))

    rows.append((doc_id, PROSECUTION, "NEGATIVE_CONTROL", "no", "prosecution"))
    rows.append((doc_id, court, "NEGATIVE_CONTROL", "no", "court"))
    return rows


def build_template_environment(project_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(project_root / "templates")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def build_runtime_context(args: Namespace, project_root: Path) -> RuntimeContext:
    cfg = load_config(project_root / "config.yaml")
    profile = cfg["profile"]
    case_cfg = cfg.get("case") or {}
    if not isinstance(case_cfg, dict):
        raise SystemExit("Top-level case section must be a mapping")

    doc_type = args.doc_type if args.doc_type is not None else profile["doc_type"]
    fraud_type = args.fraud_type if args.fraud_type is not None else profile["fraud_type"]

    output_dir = resolve_project_path(project_root, cfg["output_dir"])
    schema_dir = resolve_project_path(project_root, cfg.get("schema_dir", "schemas"))
    output_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(exist_ok=True)

    try:
        section_word_targets = build_section_word_targets(profile, doc_type)
        documents = resolve_documents_to_generate(profile, args.documents)
        prose_overrides = resolve_prose_overrides(case_cfg, doc_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    schema_source_path = (
        resolve_project_path(project_root, args.from_schema)
        if args.from_schema
        else None
    )

    return RuntimeContext(
        cfg=cfg,
        profile=profile,
        case_cfg=case_cfg,
        ollama_cfg=cfg["ollama"],
        nat_locales=cfg["nationality_locales"],
        vat_prefixes=cfg["vat_prefixes"],
        doc_type=doc_type,
        fraud_type=fraud_type,
        output_dir=output_dir,
        schema_dir=schema_dir,
        template_env=build_template_environment(project_root),
        sections=EN_SECTIONS[doc_type],
        labels=EN_LABELS,
        section_word_targets=section_word_targets,
        documents=documents,
        prose_overrides=prose_overrides,
        schema_source_path=schema_source_path,
        no_llm=args.no_llm,
    )


def build_size_label(context: RuntimeContext) -> str:
    total_prose_words = sum(context.section_word_targets.values())
    if context.profile.get("section_words") is not None:
        return f"{total_prose_words}w prose"
    return f"{context.profile['pages']}p target (~{total_prose_words}w prose)"


def resolve_document_inputs(context: RuntimeContext) -> DocumentInputs:
    try:
        defendants, collateral, charged_orgs, associated_orgs = resolve_case_entities(
            context.profile,
            context.case_cfg,
            context.nat_locales,
            context.vat_prefixes,
        )
        metadata = resolve_case_metadata(context.case_cfg, context.doc_type)
        counts_list = resolve_counts(
            context.cfg,
            context.case_cfg,
            context.doc_type,
            context.fraud_type,
            defendants,
            charged_orgs,
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
        counts_list=counts_list,
    )


def resolve_schema_for_document(
    context: RuntimeContext,
    document: DocumentInputs,
    document_index: int,
) -> tuple[str, dict]:
    if context.schema_source_path:
        loaded_schema = load_case_schema(context.schema_source_path)
        try:
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

    counter = next_counter(context.output_dir, context.doc_type, context.fraud_type)
    doc_id = make_doc_id(context.doc_type, context.fraud_type, counter)
    try:
        if is_auto(context.case_cfg.get("schema")):
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
                context.case_cfg["schema"],
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


def build_llm_sections(
    context: RuntimeContext,
    document: DocumentInputs,
    schema_context: str,
) -> list[str]:
    llm_sections = []
    entity_density = 1.5
    case_number = document.metadata["case_number"]

    for section_name, section_words in context.section_word_targets.items():
        if section_name in context.prose_overrides:
            text = context.prose_overrides[section_name]
            print(
                f"  section '{section_name}': using config prose "
                f"({len(text.split())}w)"
            )
        elif context.no_llm:
            text = "[placeholder prose]"
            print(f"  section '{section_name}': placeholder prose")
        else:
            entity_mentions = max(
                1,
                math.ceil(entity_density * section_words / WORDS_PER_PAGE),
            )
            print(
                f"  section '{section_name}': ~{section_words}w, "
                f"{entity_mentions} mention(s)/entity"
            )
            text = generate_section(
                ollama_cfg=context.ollama_cfg,
                section_name=section_name,
                doc_type=context.doc_type,
                fraud_type=context.fraud_type,
                defendants=document.defendants,
                collateral=document.collateral,
                charged_orgs=document.charged_orgs,
                associated_orgs=document.associated_orgs,
                case_number=case_number,
                word_target=section_words,
                entity_mentions=entity_mentions,
                schema_context=schema_context,
            )
        llm_sections.append(text)

    return llm_sections


def render_document_text(
    context: RuntimeContext,
    document: DocumentInputs,
    llm_sections: list[str],
) -> str:
    template = context.template_env.get_template(f"en_{context.doc_type}.j2")
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
    doc_dir.mkdir(exist_ok=True)

    write_case_schema(schema_path, schema)

    txt_path = doc_dir / f"{doc_id}.txt"
    txt_path.write_text(rendered_text, encoding="utf-8")

    gt_rows = build_groundtruth_rows(
        doc_id,
        document.defendants,
        document.collateral,
        document.charged_orgs,
        document.associated_orgs,
        document.metadata["court"],
    )
    gt_path = doc_dir / "groundtruth.tsv"
    write_groundtruth(gt_path, gt_rows)

    actual_words = len(rendered_text.split())
    actual_pages = round(actual_words / WORDS_PER_PAGE, 1)
    print(f"  Schema : {schema_path}")
    print(f"  Saved  : {txt_path}  ({actual_words}w ≈ {actual_pages} pages)")
    print(f"  GT rows: {len(gt_rows)}  →  {gt_path}")


def run_generation(args: Namespace, project_root: Path) -> None:
    context = build_runtime_context(args, project_root)
    size_label = build_size_label(context)

    for document_index in range(context.documents):
        print(
            f"\n[{document_index + 1}/{context.documents}] Generating "
            f"{context.doc_type} / {context.fraud_type} / {size_label} …"
        )

        document = resolve_document_inputs(context)
        doc_id, schema = resolve_schema_for_document(context, document, document_index)
        llm_sections = build_llm_sections(
            context,
            document,
            schema_to_context(schema),
        )
        rendered_text = render_document_text(context, document, llm_sections)
        save_document_artifacts(context, document, doc_id, schema, rendered_text)

    print("\nDone.")
