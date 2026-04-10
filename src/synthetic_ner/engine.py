"""Core document generation engine."""

import math
import re
from argparse import Namespace
from pathlib import Path

import requests
from jinja2 import Environment, FileSystemLoader
from src.synthetic_ner.case import (
    resolve_case_entities,
    resolve_case_metadata,
    resolve_counts,
    resolve_prose_overrides,
)
from src.synthetic_ner.config import build_generation_config, resolve_section_order
from src.synthetic_ner.constants import (
    EN_LABELS,
    EN_SECTIONS,
    INCOMPLETE_SECTION_MARKERS,
    PROSECUTION,
    SECTION_DESCRIPTIONS,
)
from src.synthetic_ner.schema import (
    counter_from_doc_id,
    load_case_schema,
    make_case_schema,
    make_doc_id,
    next_counter,
    normalize_schema,
    schema_to_context,
    write_case_schema,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext
from src.synthetic_ner.utils import (
    is_auto,
    load_config,
    resolve_project_path,
    write_groundtruth,
)


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
    chunk_words: int,
    context_tail_chars: int,
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
        chunk_target = min(chunk_words, remaining)
        is_first_chunk = words_so_far == 0

        context_fragment = ""
        if chunks:
            last_chunk = chunks[-1]
            context_fragment = (
                "\nContinue directly from where the previous passage ended. "
                f"Last sentences: ...{last_chunk[-context_tail_chars:]}"
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


def build_section_word_targets(
    profile: dict,
    generation_cfg,
    doc_type: str,
) -> dict[str, int]:
    section_order = resolve_section_order(generation_cfg, doc_type)
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

    total_words = pages * generation_cfg.words_per_page
    prose_words = max(300, total_words - 200)
    weights = generation_cfg.section_weights[doc_type]
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
    try:
        generation_cfg = build_generation_config(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    profile = dict(cfg["profile"])
    case_cfg = cfg.get("case") or {}
    if not isinstance(case_cfg, dict):
        raise SystemExit("Top-level case section must be a mapping")
    if args.pages is not None:
        if args.pages <= 0:
            raise SystemExit("--pages must be a positive integer")
        profile["pages"] = args.pages
        profile.pop("section_words", None)

    doc_type = args.doc_type if args.doc_type is not None else profile["doc_type"]
    fraud_type = args.fraud_type if args.fraud_type is not None else profile["fraud_type"]

    workflow_cfg = cfg.get("workflow") or {}
    if not isinstance(workflow_cfg, dict):
        raise SystemExit("Top-level workflow section must be a mapping")
    output_dir = resolve_project_path(project_root, cfg["output_dir"])
    schema_dir = resolve_project_path(project_root, cfg.get("schema_dir", "schemas"))
    memory_dir = resolve_project_path(project_root, cfg.get("memory_dir", "memory"))
    trace_dir = resolve_project_path(project_root, cfg.get("trace_dir", "traces"))
    output_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(exist_ok=True)
    memory_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)

    try:
        section_word_targets = build_section_word_targets(
            profile,
            generation_cfg,
            doc_type,
        )
        documents = resolve_documents_to_generate(profile, args.documents)
        prose_overrides = resolve_prose_overrides(
            case_cfg,
            generation_cfg,
            doc_type,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    schema_source_path = (
        resolve_project_path(project_root, args.from_schema)
        if args.from_schema
        else None
    )

    return RuntimeContext(
        cfg=cfg,
        generation_cfg=generation_cfg,
        profile=profile,
        case_cfg=case_cfg,
        langfuse_cfg=cfg.get("langfuse") or {},
        ollama_cfg=cfg["ollama"],
        workflow_cfg=workflow_cfg,
        nat_locales=cfg["nationality_locales"],
        vat_prefixes=cfg["vat_prefixes"],
        doc_type=doc_type,
        fraud_type=fraud_type,
        output_dir=output_dir,
        schema_dir=schema_dir,
        memory_dir=memory_dir,
        trace_dir=trace_dir,
        template_env=build_template_environment(project_root),
        sections=EN_SECTIONS[doc_type],
        labels=EN_LABELS,
        section_word_targets=section_word_targets,
        documents=documents,
        prose_overrides=prose_overrides,
        schema_source_path=schema_source_path,
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
        else:
            entity_mentions = max(
                1,
                math.ceil(entity_density * section_words / context.generation_cfg.words_per_page),
            )
            print(
                f"  section '{section_name}': ~{section_words}w, "
                f"{entity_mentions} mention(s)/entity"
            )
            writer_cfg = context.workflow_cfg.get("writer", {})
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
                chunk_words=writer_cfg.get("chunk_words", 700),
                context_tail_chars=writer_cfg.get("context_tail_chars", 600),
                schema_context=schema_context,
            )
        llm_sections.append(text)

    return llm_sections


def collect_section_output_problems(
    section_targets: dict[str, int],
    section_texts: list[str],
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

        minimum_words = max(60, section_targets[section_name] // 4)
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
    actual_pages = round(actual_words / context.generation_cfg.words_per_page, 1)
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
        problems = collect_section_output_problems(
            context.section_word_targets,
            llm_sections,
        )
        if problems:
            raise SystemExit(
                "Generation aborted because one or more sections are incomplete: "
                + "; ".join(problems)
            )
        rendered_text = render_document_text(context, document, llm_sections)
        save_document_artifacts(context, document, doc_id, schema, rendered_text)

    print("\nDone.")
