"""Prefect orchestration for audited synthetic NER generation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.context import get_run_context
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from src.synthetic_ner.case import resolve_counts
from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.engine import (
    build_runtime_context,
    resolve_document_inputs,
    resolve_project_path,
    resolve_schema_for_document,
)
from src.synthetic_ner.schema import counter_from_doc_id, make_doc_id
from src.synthetic_ner.tasks.orchestrator import run_document_graph
from src.synthetic_ner.tasks.quality_report import (
    DEFAULT_QUALITY_CONFIG_PATH,
    build_quality_report,
    format_markdown_report,
    load_quality_scoring_config,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.utils import load_config

ARTIFACT_TEXT_LIMIT = 24_000


class ScenarioReviewInput(RunInput):
    action: str = "continue"
    scenario_summary: str = ""
    document_type_details: str = ""
    faker_generation_plan: str = ""
    llm_language_plan: str = ""
    fraud_type_details: str = ""
    editable_fields_help: str = ""
    case_config: str = ""
    documents: int | None = None
    doc_type: str = ""
    fraud_type: str = ""
    from_schema: str = ""


class EntityReviewInput(RunInput):
    action: str = "continue"
    document_json: str = ""
    refresh_counts: bool = True


def _current_flow_run_id() -> str | None:
    """Return the active Prefect flow run id when running inside a flow."""
    try:
        flow_run = getattr(get_run_context(), "flow_run", None)
    except Exception:
        return None
    return str(flow_run.id) if flow_run is not None else None


@task(name="scenario-selection")
def select_scenario(
    *,
    project_root: Path,
    case_config: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
    quality_config: str | None = None,
) -> dict[str, Any]:
    """Resolve the selected scenario and publish the input files used by the run."""
    scenario = _build_scenario(
        project_root=project_root,
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
        quality_config=quality_config,
    )
    _publish_scenario_artifacts(scenario)
    get_run_logger().info(
        "Selected scenario=%s doc_type=%s fraud_type=%s documents=%s",
        case_config,
        scenario["doc_type"],
        scenario["fraud_type"],
        scenario["documents"],
    )
    return scenario


@task(name="scenario-review-request")
def publish_scenario_review_request(
    scenario: dict[str, Any],
    timeout_seconds: int,
) -> None:
    """Publish a visible human-review checkpoint before the flow pauses."""
    review_details = _scenario_review_initial_data(scenario)
    config_rows = _scenario_config_review_rows(scenario)
    if config_rows:
        create_table_artifact(
            key=_artifact_key("scenario-review-config-fields"),
            description="Human-readable scenario fields loaded from config",
            table=config_rows,
        )
    create_markdown_artifact(
        key=_artifact_key("scenario-review-request"),
        description="Human review checkpoint for the selected scenario",
        markdown="\n".join(
            [
                "# Human Review Required: Scenario",
                "",
                "The flow is about to pause for scenario review.",
                "",
                "## Scenario Summary",
                "",
                review_details["scenario_summary"],
                "",
                "## Document Type",
                "",
                review_details["document_type_details"],
                "",
                "## Faker Generation Plan",
                "",
                review_details["faker_generation_plan"],
                "",
                "## LLM Language Plan",
                "",
                review_details["llm_language_plan"],
                "",
                "## Fraud Type",
                "",
                review_details["fraud_type_details"],
                "",
                "## What To Check",
                "",
                "| Field | Current Value |",
                "| --- | --- |",
                f"| Scenario config | `{scenario['case_config']}` |",
                f"| Document type | `{scenario['doc_type']}` |",
                f"| Fraud type | `{scenario['fraud_type']}` |",
                f"| Documents | `{scenario['documents']}` |",
                f"| Source schema | `{scenario['from_schema'] or 'none'}` |",
                "",
                "## Input Files",
                "",
                "| Role | Exists | Path |",
                "| --- | --- | --- |",
                *[
                    f"| {item['role']} | `{item['exists']}` | `{item['path']}` |"
                    for item in scenario["input_files"]
                ],
                "",
                "## How To Continue",
                "",
                "Open the parent flow run page, use the resume control, and submit one of:",
                "",
                "- `action=continue` to proceed with these values.",
                "- `action=reload` to apply edited fields from the resume form.",
                "- `action=cancel` to stop the run.",
                "",
                f"The pause timeout is `{timeout_seconds}` seconds.",
            ]
        ),
    )
    get_run_logger().info(
        "Human scenario review requested. Resume the parent flow run with "
        "action=continue, action=reload, or action=cancel."
    )


def review_selected_scenario(
    *,
    project_root: Path,
    scenario: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Pause the flow so a reviewer can approve or alter the selected scenario."""
    publish_scenario_review_request(scenario, timeout_seconds)
    review_details = _scenario_review_initial_data(scenario)
    review_input = ScenarioReviewInput.with_initial_data(
        description=(
            "Review the selected scenario before Faker and the LLM run. The fields below "
            "summarise the config, document shape, generated-data plan, and LLM language "
            "plan. Use action='continue' to proceed, action='reload' after changing "
            "editable fields, or action='cancel' to stop."
        ),
        action="continue",
        **review_details,
        case_config=scenario["case_config"],
        documents=scenario["documents"],
        doc_type=scenario["doc_type"] or "",
        fraud_type=scenario["fraud_type"] or "",
        from_schema=scenario["from_schema"] or "",
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="scenario-review",
    )
    if response is None:
        return scenario

    action = response.action.strip().lower()
    if action in {"cancel", "abort", "stop"}:
        raise SystemExit("Scenario review cancelled the Prefect run.")
    if action not in {"continue", "reload"}:
        raise SystemExit(
            "Scenario review action must be one of: continue, reload, cancel."
        )

    reviewed_scenario = _build_scenario(
        project_root=project_root,
        case_config=response.case_config.strip() or scenario["case_config"],
        documents=(
            response.documents
            if response.documents is not None
            else scenario["documents"]
        ),
        doc_type=response.doc_type.strip() or scenario["doc_type"],
        fraud_type=response.fraud_type.strip() or scenario["fraud_type"],
        from_schema=response.from_schema.strip() or None,
        quality_config=scenario.get("quality_config"),
    )
    _publish_scenario_artifacts(reviewed_scenario)
    get_run_logger().info(
        "Scenario review completed: scenario=%s doc_type=%s fraud_type=%s documents=%s",
        reviewed_scenario["case_config"],
        reviewed_scenario["doc_type"],
        reviewed_scenario["fraud_type"],
        reviewed_scenario["documents"],
    )
    return reviewed_scenario


@task(name="entity-review-request")
def publish_entity_review_request(
    document_payload: dict[str, Any],
    timeout_seconds: int,
) -> None:
    """Publish a visible human-review checkpoint before entity review pauses."""
    people_rows = _entity_people_review_rows(document_payload)
    org_rows = _entity_org_review_rows(document_payload)
    metadata_rows = _key_value_rows(document_payload.get("metadata", {}))
    counts_rows = _counts_review_rows(document_payload.get("counts_list", []))
    if people_rows:
        create_table_artifact(
            key=_artifact_key("entity-review-people"),
            description="People generated for human entity review",
            table=people_rows,
        )
    if org_rows:
        create_table_artifact(
            key=_artifact_key("entity-review-organisations"),
            description="Organisations generated for human entity review",
            table=org_rows,
        )
    if metadata_rows:
        create_table_artifact(
            key=_artifact_key("entity-review-metadata"),
            description="Case metadata generated for human entity review",
            table=metadata_rows,
        )
    if counts_rows:
        create_table_artifact(
            key=_artifact_key("entity-review-counts"),
            description="Counts generated for human entity review",
            table=counts_rows,
        )
    _publish_document_inputs_artifact(
        key=_artifact_key("entity-review-document-inputs"),
        description="Editable document inputs for human entity review",
        document_payload=document_payload,
    )
    names_summary = _entity_names_summary(document_payload)
    create_markdown_artifact(
        key=_artifact_key("entity-review-request"),
        description="Human review checkpoint for resolved document entities",
        markdown="\n".join(
            [
                "# Human Review Required: Document Inputs",
                "",
                "The flow is about to pause for entity and document-input review.",
                "",
                "## Generated Names",
                "",
                names_summary,
                "",
                "## What To Check",
                "",
                "- The `entity-review-people` table for defendant and collateral names.",
                "- The `entity-review-organisations` table for company names and VAT IDs.",
                "- The `entity-review-metadata` table for court, case number, dates, and refs.",
                "- The `entity-review-counts` table for generated count particulars.",
                "- Metadata",
                "- Counts list",
                "",
                "The editable JSON payload is published as the "
                "`entity-review-document-inputs` artifact.",
                "",
                "## How To Continue",
                "",
                "Open the parent flow run page, use the resume control, and submit one of:",
                "",
                "- `action=continue` to proceed unchanged.",
                "- `action=apply_json` to use edited `document_json` from the resume form.",
                "- `action=cancel` to stop the run.",
                "",
                f"The pause timeout is `{timeout_seconds}` seconds.",
            ]
        ),
    )
    get_run_logger().info(
        "Human entity review requested. Resume the parent flow run with "
        "action=continue, action=apply_json, or action=cancel."
    )


def review_document_entities(
    context: Any,
    document: Any,
    timeout_seconds: int,
) -> Any:
    """Pause the flow so a reviewer can approve or edit resolved document inputs."""
    document_payload = _document_to_payload(document)
    publish_entity_review_request(document_payload, timeout_seconds)
    names_summary = _entity_names_summary(document_payload)
    review_input = EntityReviewInput.with_initial_data(
        description=(
            "Review resolved people and organisations before document generation. "
            f"Generated names: {names_summary} "
            "Use action='continue' to proceed, action='apply_json' to use edited "
            "document_json, or action='cancel' to stop."
        ),
        action="continue",
        document_json=json.dumps(document_payload, indent=2, ensure_ascii=False),
        refresh_counts=True,
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="entity-review",
    )
    if response is None:
        return document

    action = response.action.strip().lower()
    if action in {"cancel", "abort", "stop"}:
        raise SystemExit("Entity review cancelled the Prefect run.")
    if action == "continue":
        return document
    if action != "apply_json":
        raise SystemExit(
            "Entity review action must be one of: continue, apply_json, cancel."
        )

    reviewed_document = _document_from_review_json(response.document_json)
    if response.refresh_counts:
        reviewed_document.counts_list = resolve_counts(
            context.app_config.fraud_statutes,
            context.case_cfg,
            context.doc_type,
            context.fraud_type,
            reviewed_document.defendants,
            reviewed_document.charged_orgs,
            reviewed_document.metadata.get("offence_period"),
        )
    _publish_entity_artifacts(reviewed_document)
    _publish_document_inputs_artifact(
        key=_artifact_key("entity-review-applied-document-inputs"),
        description="Document inputs after human entity review edits",
        document_payload=_document_to_payload(reviewed_document),
    )
    return reviewed_document


def _build_scenario(
    *,
    project_root: Path,
    case_config: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
    quality_config: str | None = None,
) -> dict[str, Any]:
    root_config_path = project_root / "config.yaml"
    case_config_path = resolve_project_path(project_root, case_config)
    root_raw = load_config(root_config_path)
    case_raw = load_config(case_config_path)
    profile = case_raw.get("profile", {}) if isinstance(case_raw, dict) else {}
    case_section = case_raw.get("case", {}) if isinstance(case_raw, dict) else {}
    workflow = root_raw.get("workflow", {}) if isinstance(root_raw, dict) else {}
    model_routing = root_raw.get("model_routing", {}) if isinstance(root_raw, dict) else {}
    generation = root_raw.get("generation", {}) if isinstance(root_raw, dict) else {}

    selected_doc_type = doc_type or profile.get("doc_type")
    selected_fraud_type = fraud_type or profile.get("fraud_type")
    selected_documents = documents if documents is not None else profile.get("documents")
    prompts_config = workflow.get("prompts_config_path", "prompts/workflow_prompts.yaml")
    prompts_path = resolve_project_path(project_root, prompts_config)
    prompts_raw = load_config(prompts_path) if prompts_path.exists() else {}
    template_path = (
        project_root / "templates" / f"en_{selected_doc_type}.j2"
        if selected_doc_type
        else None
    )

    input_files = [
        _input_file_record("root config", root_config_path, required=True),
        _input_file_record("scenario config", case_config_path, required=True),
        _input_file_record(
            "workflow prompts",
            resolve_project_path(project_root, prompts_config),
            required=True,
        ),
    ]
    if selected_doc_type:
        input_files.append(
            _input_file_record(
                "document template",
                template_path or project_root / "templates",
                required=True,
            )
        )
    if from_schema:
        input_files.append(
            _input_file_record(
                "source schema",
                resolve_project_path(project_root, from_schema),
                required=True,
            )
        )
    if quality_config:
        input_files.append(
            _input_file_record(
                "quality scoring config",
                resolve_project_path(project_root, quality_config),
                required=True,
            )
        )
    for env_name in (".env", ".env.langfuse"):
        env_path = project_root / env_name
        if env_path.exists():
            input_files.append(_input_file_record(env_name, env_path, required=False))

    return {
        "case_config": case_config,
        "documents": selected_documents,
        "doc_type": selected_doc_type,
        "fraud_type": selected_fraud_type,
        "from_schema": from_schema,
        "quality_config": quality_config,
        "input_files": input_files,
        "profile": profile,
        "case": case_section,
        "fraud_statutes": case_raw.get("fraud_statutes", {})
        if isinstance(case_raw, dict)
        else {},
        "workflow": workflow,
        "model_routing": model_routing,
        "generation": generation,
        "prompts_config": prompts_config,
        "prompt_names": sorted((prompts_raw.get("prompts") or {}).keys())
        if isinstance(prompts_raw, dict)
        else [],
        "template_path": str(template_path) if template_path else "",
        "template_preview": _template_preview(template_path),
    }


@task(name="configs-ingestion")
def ingest_configs(
    *,
    project_root: Path,
    scenario: dict[str, Any],
) -> Any:
    """Load config files and build the runtime context."""
    load_env_files(project_root)
    args = _build_args(
        case_config=scenario["case_config"],
        documents=scenario["documents"],
        doc_type=scenario["doc_type"],
        fraud_type=scenario["fraud_type"],
        from_schema=scenario["from_schema"],
    )
    context = build_runtime_context(args, project_root)
    logger = get_run_logger()
    logger.info(
        "Loaded configs: doc_type=%s fraud_type=%s documents=%s output_dir=%s",
        context.doc_type,
        context.fraud_type,
        context.documents,
        context.output_dir,
    )
    _publish_config_artifacts(
        project_root=project_root,
        case_config=scenario["case_config"],
        context=context,
    )
    return context


@task(name="faker-entities")
def resolve_entities(context: Any) -> Any:
    """Generate or normalize people, organisations, addresses and counts."""
    document = resolve_document_inputs(context)
    logger = get_run_logger()
    logger.info(
        "Resolved entities: defendants=%s collateral=%s charged_orgs=%s associated_orgs=%s",
        len(document.defendants),
        len(document.collateral),
        len(document.charged_orgs),
        len(document.associated_orgs),
    )
    _publish_entity_artifacts(document)
    return document


@task(name="select-doc-id")
def select_doc_id(context: Any) -> str:
    """Select the next document id using all known artifact roots."""
    used_counters = _used_doc_counters(context)
    next_counter = (max(used_counters) + 1) if used_counters else 1
    doc_id = make_doc_id(context.doc_type, context.fraud_type, next_counter)
    get_run_logger().info(
        "Selected doc_id=%s after scanning output, partials, schemas and memory",
        doc_id,
    )
    return doc_id


@task(name="case-schema")
def build_case_schema(
    context: Any,
    document: Any,
    document_index: int,
    doc_id: str,
) -> tuple[str, dict]:
    """Build or load the relationship schema for one document."""
    doc_id, schema = resolve_schema_for_document(
        context,
        document,
        document_index,
        doc_id_override=doc_id,
    )
    get_run_logger().info("Resolved schema for %s with %s edges", doc_id, len(schema["edges"]))
    _publish_schema_artifacts(doc_id, schema)
    return doc_id, schema


@task(name="langgraph-langfuse-generation")
def run_langgraph_langfuse(
    context: Any,
    document: Any,
    schema: dict,
    doc_id: str,
    prefect_flow_run_id: str | None = None,
) -> str:
    """Run the LangGraph generation workflow with Langfuse tracing enabled by config."""
    try:
        run_document_graph(
            context=context,
            document=document,
            schema=schema,
            doc_id=doc_id,
            workflow_run_id=prefect_flow_run_id or doc_id,
            prefect_flow_run_id=prefect_flow_run_id,
        )
    finally:
        _publish_memory_artifacts(context, doc_id)
    get_run_logger().info("Completed LangGraph/Langfuse generation for %s", doc_id)
    return doc_id


@task(name="end-of-pipeline-file-audit")
def audit_created_files(context: Any, doc_id: str) -> Path:
    """Write a document-level manifest of generated files and checksums."""
    audit_path = context.output_dir / doc_id / "file_audit.json"
    files = _collect_document_files(context, doc_id, exclude={audit_path})
    payload = {
        "doc_id": doc_id,
        "created_at": _utc_now(),
        "roots": {
            "output": str(context.output_dir / doc_id),
            "partial_output": str(context.output_dir / "_partial" / doc_id),
            "schema": str(context.schema_dir / f"{doc_id}.json"),
            "memory": str(context.memory_dir / f"case_{doc_id}"),
        },
        "files": files,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["files"].append(_file_record(audit_path))
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_prefect_artifacts(context, doc_id, payload)
    get_run_logger().info("Wrote file audit for %s to %s", doc_id, audit_path)
    return audit_path


@task(name="score-document-quality")
def score_document_quality(
    context: Any,
    doc_id: str,
    quality_config: str,
) -> dict[str, Any]:
    """Score one existing generated document from current artifacts."""
    quality_config_path = resolve_project_path(context.project_root, quality_config)
    scoring_config = load_quality_scoring_config(quality_config_path)
    report = build_quality_report(context, doc_id, scoring_config)
    _publish_quality_artifacts(doc_id, report)
    get_run_logger().info(
        "Scored %s quality as %s (%s)",
        doc_id,
        report["overall_score"],
        report["verdict"],
    )
    return report


@flow(name="synthetic-ner-generation")
def generate_dataset(
    case_config: str = "config_case/case_1.yaml",
    documents: int | None = None,
    doc_type: str | None = None,
    fraud_type: str | None = None,
    from_schema: str | None = None,
    project_root: str | None = None,
    review_scenario: bool = False,
    review_entities: bool = False,
    review_timeout_seconds: int = 3600,
) -> list[str]:
    """Orchestrate a complete generation run with Prefect-visible pipeline stages."""
    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else Path(__file__).resolve().parents[2]
    )
    scenario = select_scenario(
        project_root=resolved_project_root,
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
    )
    if review_scenario:
        scenario = review_selected_scenario(
            project_root=resolved_project_root,
            scenario=scenario,
            timeout_seconds=review_timeout_seconds,
        )
    context = ingest_configs(
        project_root=resolved_project_root,
        scenario=scenario,
    )
    prefect_flow_run_id = _current_flow_run_id()

    doc_ids: list[str] = []
    for document_index in range(context.documents):
        document = resolve_entities(context)
        if review_entities:
            document = review_document_entities(
                context,
                document,
                review_timeout_seconds,
            )
        selected_doc_id = select_doc_id(context)
        doc_id, schema = build_case_schema(
            context,
            document,
            document_index,
            selected_doc_id,
        )
        run_langgraph_langfuse(context, document, schema, doc_id, prefect_flow_run_id)
        audit_created_files(context, doc_id)
        doc_ids.append(doc_id)

    get_run_logger().info("Generation flow completed for documents: %s", ", ".join(doc_ids))
    return doc_ids


@flow(name="synthetic-ner-document-quality")
def score_existing_document(
    doc_id: str,
    case_config: str = "config_case/case_1.yaml",
    quality_config: str = DEFAULT_QUALITY_CONFIG_PATH,
    doc_type: str | None = None,
    fraud_type: str | None = None,
    project_root: str | None = None,
    review_scenario: bool = False,
    review_timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Score an existing document without regenerating it."""
    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else Path(__file__).resolve().parents[2]
    )
    scenario = select_scenario(
        project_root=resolved_project_root,
        case_config=case_config,
        documents=1,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=None,
        quality_config=quality_config,
    )
    if review_scenario:
        scenario = review_selected_scenario(
            project_root=resolved_project_root,
            scenario=scenario,
            timeout_seconds=review_timeout_seconds,
        )
    context = ingest_configs(
        project_root=resolved_project_root,
        scenario=scenario,
    )
    return score_document_quality(context, doc_id, quality_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic NER generation through Prefect.")
    parser.add_argument(
        "--score-doc-id",
        default=None,
        help="Score an existing document id instead of running generation.",
    )
    parser.add_argument(
        "--quality-config",
        default=DEFAULT_QUALITY_CONFIG_PATH,
        help="Quality scoring config path relative to project root.",
    )
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case recipe config path relative to project root.",
    )
    parser.add_argument("--documents", "--count", dest="documents", type=int, default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--fraud-type", default=None)
    parser.add_argument("--from-schema", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--review-scenario",
        action="store_true",
        help="Pause after scenario selection so a human can approve or alter inputs.",
    )
    parser.add_argument(
        "--review-entities",
        action="store_true",
        help="Pause after entity resolution so a human can approve or edit document inputs.",
    )
    parser.add_argument(
        "--review-timeout-seconds",
        type=int,
        default=3600,
        help="How long a human review pause waits for input before continuing unchanged.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.score_doc_id:
        score_existing_document(
            doc_id=args.score_doc_id,
            case_config=args.case_config,
            quality_config=args.quality_config,
            doc_type=args.doc_type,
            fraud_type=args.fraud_type,
            project_root=args.project_root,
            review_scenario=args.review_scenario,
            review_timeout_seconds=args.review_timeout_seconds,
        )
        return
    generate_dataset(
        case_config=args.case_config,
        documents=args.documents,
        doc_type=args.doc_type,
        fraud_type=args.fraud_type,
        from_schema=args.from_schema,
        project_root=args.project_root,
        review_scenario=args.review_scenario,
        review_entities=args.review_entities,
        review_timeout_seconds=args.review_timeout_seconds,
    )


def _build_args(
    *,
    case_config: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
) -> Namespace:
    return Namespace(
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
        workflow_mode="langgraph",
    )


def _input_file_record(role: str, path: Path, *, required: bool) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": None,
        "sha256": None,
        "modified_at": None,
    }
    if exists and path.is_file():
        file_record = _file_record(path)
        record.update(
            {
                "size_bytes": file_record["size_bytes"],
                "sha256": file_record["sha256"],
                "modified_at": file_record["modified_at"],
            }
        )
    return record


def _template_preview(path: Path | None, max_lines: int = 18) -> str:
    if path is None or not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:max_lines])


def _scenario_review_initial_data(scenario: dict[str, Any]) -> dict[str, str]:
    return {
        "scenario_summary": _scenario_summary_text(scenario),
        "document_type_details": _document_type_details_text(scenario),
        "faker_generation_plan": _faker_generation_plan_text(scenario),
        "llm_language_plan": _llm_language_plan_text(scenario),
        "fraud_type_details": _fraud_type_details_text(scenario),
        "editable_fields_help": (
            "Only edit case_config, documents, doc_type, fraud_type, or from_schema. "
            "Use action=reload to re-read those choices, action=continue to accept, "
            "or action=cancel to stop."
        ),
    }


def _scenario_summary_text(scenario: dict[str, Any]) -> str:
    case = scenario.get("case", {})
    metadata = case.get("metadata", {}) if isinstance(case, dict) else {}
    prose = case.get("prose", {}) if isinstance(case, dict) else {}
    auto_metadata = _auto_keys(metadata)
    auto_prose = _auto_keys(prose)
    return (
        f"Create {scenario.get('documents')} English `{scenario.get('doc_type')}` "
        f"document(s) for `{scenario.get('fraud_type')}` using "
        f"`{scenario.get('case_config')}`. Metadata auto fields: "
        f"{_join_or_none(auto_metadata)}. Prose sections set to auto: "
        f"{_join_or_none(auto_prose)}."
    )


def _document_type_details_text(scenario: dict[str, Any]) -> str:
    profile = scenario.get("profile", {})
    section_words = profile.get("section_words", {}) if isinstance(profile, dict) else {}
    sections = [
        f"{name} (~{words} words)" for name, words in section_words.items()
    ]
    template_path = scenario.get("template_path") or "not resolved"
    preview = scenario.get("template_preview") or "Template preview unavailable."
    return (
        f"`doc_type={scenario.get('doc_type')}` uses template `{template_path}`. "
        "The template supplies fixed headings, case references, count blocks, and "
        "places generated LLM prose into `llm_sections`. Section targets: "
        f"{_join_or_none(sections)}.\n\nTemplate opening:\n{preview}"
    )


def _faker_generation_plan_text(scenario: dict[str, Any]) -> str:
    case = scenario.get("case", {})
    cast = case.get("cast", {}) if isinstance(case, dict) else {}
    defendants = _person_specs_summary(cast.get("defendants", []))
    collateral = _person_specs_summary(cast.get("collateral", []))
    return (
        "Before the LLM writes anything, Faker/config resolution will create or load "
        f"people and organisations. Defendants: {defendants}. Collateral people: "
        f"{collateral}. Charged organisations: {cast.get('charged_orgs', 0)}. "
        f"Associated organisations: {cast.get('associated_orgs', 0)}. "
        "Generated names, addresses, VAT IDs, dates, and counts are reviewed at the "
        "next pause."
    )


def _llm_language_plan_text(scenario: dict[str, Any]) -> str:
    workflow = scenario.get("workflow", {})
    model_routing = scenario.get("model_routing", {})
    stages = model_routing.get("stages", {}) if isinstance(model_routing, dict) else {}
    stage_text = ", ".join(
        f"{name}={stage.get('provider')}:{stage.get('model')}"
        for name, stage in stages.items()
        if isinstance(stage, dict)
    )
    prompt_names = scenario.get("prompt_names", [])
    prompt_text = _join_or_none(prompt_names[:8])
    if len(prompt_names) > 8:
        prompt_text = f"{prompt_text}, ..."
    return (
        "Language is English because the selected template is `en_*` and the prompt "
        "pack is written in English. Workflow mode: "
        f"`{workflow.get('mode', 'unknown') if isinstance(workflow, dict) else 'unknown'}`. "
        f"Prompts file: `{scenario.get('prompts_config')}`. Prompt blocks loaded: "
        f"{prompt_text}. Model routing: {stage_text or 'not configured'}."
    )


def _fraud_type_details_text(scenario: dict[str, Any]) -> str:
    fraud_type = scenario.get("fraud_type")
    statutes = scenario.get("fraud_statutes", {})
    selected = statutes.get(fraud_type, []) if isinstance(statutes, dict) else []
    if not isinstance(selected, list) or not selected:
        return f"`fraud_type={fraud_type}` has no statute entries in this config."
    offences = [
        f"{item.get('offence', 'unknown offence')} ({item.get('statute', 'unknown statute')})"
        for item in selected
        if isinstance(item, dict)
    ]
    return (
        f"`fraud_type={fraud_type}` loads these count/offence templates: "
        f"{_join_or_none(offences)}. Particulars are filled after Faker resolves "
        "defendant names, company names, and offence dates."
    )


def _scenario_config_review_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    profile = scenario.get("profile", {})
    case = scenario.get("case", {})
    workflow = scenario.get("workflow", {})
    generation = scenario.get("generation", {})
    cast = case.get("cast", {}) if isinstance(case, dict) else {}
    return [
        {
            "field": "profile.doc_type",
            "current_value": scenario.get("doc_type"),
            "meaning": "Selects the document template and section structure.",
        },
        {
            "field": "profile.fraud_type",
            "current_value": scenario.get("fraud_type"),
            "meaning": "Selects offence/count templates from fraud_statutes.",
        },
        {
            "field": "profile.documents",
            "current_value": scenario.get("documents"),
            "meaning": "Number of synthetic documents to generate.",
        },
        {
            "field": "profile.section_words",
            "current_value": _review_value(profile.get("section_words", {}))
            if isinstance(profile, dict)
            else "",
            "meaning": "Target prose length by generated section.",
        },
        {
            "field": "case.cast",
            "current_value": _review_value(cast),
            "meaning": "Faker input for people and organisation counts.",
        },
        {
            "field": "case.metadata",
            "current_value": _review_value(case.get("metadata", {}))
            if isinstance(case, dict)
            else "",
            "meaning": "Case refs and dates; auto values are generated before document writing.",
        },
        {
            "field": "case.prose",
            "current_value": _review_value(case.get("prose", {}))
            if isinstance(case, dict)
            else "",
            "meaning": "Controls which sections are LLM-generated.",
        },
        {
            "field": "case.counts",
            "current_value": case.get("counts") if isinstance(case, dict) else "",
            "meaning": "Controls whether legal counts are generated from fraud_statutes.",
        },
        {
            "field": "workflow",
            "current_value": _review_value(workflow),
            "meaning": "Controls LangGraph planner/writer/critic flow and prompt path.",
        },
        {
            "field": "generation.words_per_page",
            "current_value": generation.get("words_per_page")
            if isinstance(generation, dict)
            else "",
            "meaning": "Used for document sizing assumptions.",
        },
    ]


def _auto_keys(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    keys: list[str] = []
    for key, value in payload.items():
        if value == "auto":
            keys.append(key)
        elif isinstance(value, dict):
            nested = _auto_keys(value)
            keys.extend(f"{key}.{item}" for item in nested)
    return keys


def _person_specs_summary(specs: Any) -> str:
    if not isinstance(specs, list) or not specs:
        return "none"
    parts = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            continue
        title = spec.get("title") or "no title"
        parts.append(
            f"{index}: {spec.get('nationality', 'unknown nationality')} "
            f"{title}, {spec.get('surface_forms', 1)} surface form(s)"
        )
    return "; ".join(parts) if parts else "none"


def _join_or_none(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if value not in (None, "")]
    return ", ".join(cleaned) if cleaned else "none"


def _publish_scenario_artifacts(scenario: dict[str, Any]) -> None:
    create_table_artifact(
        key=_artifact_key("scenario-input-files"),
        description="Input files selected for this Prefect run",
        table=[
            {
                "role": item["role"],
                "path": item["path"],
                "required": item["required"],
                "exists": item["exists"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "modified_at": item["modified_at"],
            }
            for item in scenario["input_files"]
        ],
    )
    create_markdown_artifact(
        key=_artifact_key("scenario-selection"),
        description="Selected scenario and effective run inputs",
        markdown="\n".join(
            [
                "# Scenario Selection",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Scenario config | `{scenario['case_config']}` |",
                f"| Document type | `{scenario['doc_type']}` |",
                f"| Fraud type | `{scenario['fraud_type']}` |",
                f"| Documents | `{scenario['documents']}` |",
                f"| Source schema | `{scenario['from_schema'] or 'none'}` |",
                f"| Quality config | `{scenario['quality_config'] or 'none'}` |",
                "",
                "## Input Files",
                "",
                "| Role | Exists | Path |",
                "| --- | --- | --- |",
                *[
                    f"| {item['role']} | `{item['exists']}` | `{item['path']}` |"
                    for item in scenario["input_files"]
                ],
            ]
        ),
    )


def _document_to_payload(document: Any) -> dict[str, Any]:
    return {
        "defendants": document.defendants,
        "collateral": document.collateral,
        "charged_orgs": document.charged_orgs,
        "associated_orgs": document.associated_orgs,
        "metadata": document.metadata,
        "counts_list": document.counts_list,
    }


def _entity_names_summary(document_payload: dict[str, Any]) -> str:
    parts = [
        _names_for_group("Defendants", document_payload.get("defendants", [])),
        _names_for_group("Collateral", document_payload.get("collateral", [])),
        _names_for_group("Charged organisations", document_payload.get("charged_orgs", [])),
        _names_for_group(
            "Associated organisations",
            document_payload.get("associated_orgs", []),
        ),
    ]
    return " | ".join(part for part in parts if part)


def _names_for_group(label: str, records: Any) -> str:
    if not isinstance(records, list) or not records:
        return f"{label}: none"
    names = [
        str(record.get("name", "unnamed"))
        for record in records
        if isinstance(record, dict)
    ]
    return f"{label}: {', '.join(names) if names else 'none'}"


def _entity_people_review_rows(document_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = (
        ("defendant", document_payload.get("defendants", [])),
        ("collateral", document_payload.get("collateral", [])),
    )
    for group, records in groups:
        if not isinstance(records, list):
            continue
        for index, person in enumerate(records, start=1):
            if not isinstance(person, dict):
                continue
            rows.append(
                {
                    "group": group,
                    "index": index,
                    "name": person.get("name", ""),
                    "role": person.get("role", ""),
                    "dob": person.get("dob", ""),
                    "birthplace": person.get("birthplace", ""),
                    "nationality": person.get("nationality", ""),
                    "address": person.get("address", ""),
                    "surface_forms": ", ".join(
                        str(form) for form in person.get("surface_forms_list", [])
                    ),
                }
            )
    return rows


def _entity_org_review_rows(document_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = (
        ("charged", document_payload.get("charged_orgs", [])),
        ("associated", document_payload.get("associated_orgs", [])),
    )
    for group, records in groups:
        if not isinstance(records, list):
            continue
        for index, org in enumerate(records, start=1):
            if not isinstance(org, dict):
                continue
            rows.append(
                {
                    "group": group,
                    "index": index,
                    "name": org.get("name", ""),
                    "vat": org.get("vat", ""),
                    "nationality": org.get("nationality", ""),
                    "address": org.get("address", ""),
                }
            )
    return rows


def _key_value_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    return [
        {"field": key, "value": _review_value(value)}
        for key, value in payload.items()
    ]


def _counts_review_rows(counts_list: Any) -> list[dict[str, Any]]:
    if not isinstance(counts_list, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts_list, start=1):
        if isinstance(count, dict):
            row = {"index": index}
            row.update({key: _review_value(value) for key, value in count.items()})
            rows.append(row)
        else:
            rows.append({"index": index, "value": _review_value(count)})
    return rows


def _review_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _document_from_review_json(document_json: str) -> DocumentInputs:
    if not document_json.strip():
        raise SystemExit("Entity review action apply_json requires document_json.")
    try:
        payload = json.loads(document_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Entity review document_json is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Entity review document_json must be a JSON object.")

    required_lists = (
        "defendants",
        "collateral",
        "charged_orgs",
        "associated_orgs",
        "counts_list",
    )
    for key in required_lists:
        if not isinstance(payload.get(key), list):
            raise SystemExit(f"Entity review document_json.{key} must be a list.")
    if not isinstance(payload.get("metadata"), dict):
        raise SystemExit("Entity review document_json.metadata must be an object.")

    return DocumentInputs(
        defendants=payload["defendants"],
        collateral=payload["collateral"],
        charged_orgs=payload["charged_orgs"],
        associated_orgs=payload["associated_orgs"],
        metadata=payload["metadata"],
        counts_list=payload["counts_list"],
    )


def _publish_document_inputs_artifact(
    *,
    key: str,
    description: str,
    document_payload: dict[str, Any],
) -> None:
    create_markdown_artifact(
        key=key,
        description=description,
        markdown=(
            "# Document Inputs\n\n"
            "```json\n"
            f"{json.dumps(document_payload, indent=2, ensure_ascii=False)}\n"
            "```"
        ),
    )


def _collect_document_files(
    context: Any,
    doc_id: str,
    *,
    exclude: set[Path],
) -> list[dict[str, Any]]:
    roots = [
        context.output_dir / doc_id,
        context.output_dir / "_partial" / doc_id,
        context.schema_dir / f"{doc_id}.json",
        context.memory_dir / f"case_{doc_id}",
    ]
    excluded = {path.resolve() for path in exclude}
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        paths = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in paths:
            resolved = path.resolve()
            if resolved in excluded or resolved in seen:
                continue
            records.append(_file_record(path))
            seen.add(resolved)
    return sorted(records, key=lambda item: item["path"])


def _used_doc_counters(context: Any) -> set[int]:
    counters: set[int] = set()
    for doc_id in _artifact_doc_ids(context):
        try:
            counters.add(counter_from_doc_id(doc_id, context.doc_type, context.fraud_type))
        except ValueError:
            continue
    return counters


def _artifact_doc_ids(context: Any) -> set[str]:
    doc_ids: set[str] = set()
    prefix = f"en_{context.doc_type}_{context.fraud_type}_"
    for root in (context.output_dir, context.output_dir / "_partial"):
        if not root.exists():
            continue
        doc_ids.update(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )

    if context.schema_dir.exists():
        doc_ids.update(
            path.stem
            for path in context.schema_dir.glob(f"{prefix}*.json")
            if path.is_file()
        )

    memory_prefix = f"case_{prefix}"
    if context.memory_dir.exists():
        doc_ids.update(
            path.name.removeprefix("case_")
            for path in context.memory_dir.iterdir()
            if path.is_dir() and path.name.startswith(memory_prefix)
        )
    return doc_ids


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
    }


def _publish_config_artifacts(
    *,
    project_root: Path,
    case_config: str,
    context: Any,
) -> None:
    case_config_path = project_root / case_config
    template_path = project_root / "templates" / f"en_{context.doc_type}.j2"
    create_markdown_artifact(
        key=_artifact_key("run-config-summary"),
        description="Run configuration summary",
        markdown="\n".join(
            [
                "# Run Configuration",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Document type | `{context.doc_type}` |",
                f"| Fraud type | `{context.fraud_type}` |",
                f"| Documents | `{context.documents}` |",
                f"| Output directory | `{context.output_dir}` |",
                f"| Schema directory | `{context.schema_dir}` |",
                f"| Memory directory | `{context.memory_dir}` |",
                f"| Case config | `{case_config_path}` |",
                f"| Template | `{template_path}` |",
                f"| Workflow mode | `{context.workflow_cfg.mode}` |",
            ]
        ),
    )
    _publish_file_markdown(
        key=_artifact_key("config-yaml"),
        description="Root config.yaml used for the run",
        path=project_root / "config.yaml",
        language="yaml",
    )
    _publish_file_markdown(
        key=_artifact_key("case-config"),
        description="Case/scenario config used for the run",
        path=case_config_path,
        language="yaml",
    )
    _publish_file_markdown(
        key=_artifact_key("document-template"),
        description="Jinja document template used for the run",
        path=template_path,
        language="jinja",
    )


def _publish_entity_artifacts(document: Any) -> None:
    create_table_artifact(
        key=_artifact_key("entity-summary"),
        description="Resolved entity counts from Faker or explicit config",
        table=[
            {"entity_group": "defendants", "count": len(document.defendants)},
            {"entity_group": "collateral", "count": len(document.collateral)},
            {"entity_group": "charged_orgs", "count": len(document.charged_orgs)},
            {"entity_group": "associated_orgs", "count": len(document.associated_orgs)},
        ],
    )
    create_table_artifact(
        key=_artifact_key("resolved-people"),
        description="Resolved people for this run",
        table=[
            {
                "name": person["name"],
                "role": person["role"],
                "nationality": person["nationality"],
                "is_defendant": person["is_defendant"],
                "surface_forms": ", ".join(person["surface_forms_list"]),
            }
            for person in [*document.defendants, *document.collateral]
        ],
    )
    create_table_artifact(
        key=_artifact_key("resolved-organisations"),
        description="Resolved organisations for this run",
        table=[
            {
                "name": org["name"],
                "vat": org["vat"],
                "address": org["address"],
                "group": "charged"
                if org in document.charged_orgs
                else "associated",
            }
            for org in [*document.charged_orgs, *document.associated_orgs]
        ],
    )


def _publish_schema_artifacts(doc_id: str, schema: dict) -> None:
    create_markdown_artifact(
        key=_artifact_key(doc_id, "schema-json"),
        description=f"Case schema JSON for {doc_id}",
        markdown=(
            f"# Case Schema: `{doc_id}`\n\n"
            "```json\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
            "```"
        ),
    )
    create_table_artifact(
        key=_artifact_key(doc_id, "schema-edges"),
        description=f"Relationship graph edges for {doc_id}",
        table=[
            {
                "from": edge.get("from"),
                "to": edge.get("to"),
                "type": edge.get("type"),
                "label": edge.get("label"),
            }
            for edge in schema.get("edges", [])
        ],
    )


def _publish_memory_artifacts(context: Any, doc_id: str) -> None:
    memory_dir = context.memory_dir / f"case_{doc_id}"
    _publish_file_markdown(
        key=_artifact_key(doc_id, "case-memory"),
        description=f"CASE_MEMORY.md for {doc_id}",
        path=memory_dir / "CASE_MEMORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "run-history"),
        description=f"RUN_HISTORY.md for {doc_id}",
        path=memory_dir / "RUN_HISTORY.md",
    )


def _publish_prefect_artifacts(context: Any, doc_id: str, audit_payload: dict[str, Any]) -> None:
    doc_dir = context.output_dir / doc_id
    memory_dir = context.memory_dir / f"case_{doc_id}"
    schema_path = context.schema_dir / f"{doc_id}.json"
    document_path = doc_dir / f"{doc_id}.txt"
    report_path = doc_dir / "generation_report.md"
    audit_path = doc_dir / "file_audit.json"

    create_markdown_artifact(
        key=_artifact_key(doc_id, "run-summary"),
        description=f"Generated document run summary for {doc_id}",
        markdown=_run_summary_markdown(
            doc_id=doc_id,
            document_path=document_path,
            schema_path=schema_path,
            memory_dir=memory_dir,
            report_path=report_path,
            audit_path=audit_path,
        ),
    )
    create_table_artifact(
        key=_artifact_key(doc_id, "created-files"),
        description=f"Created files for {doc_id}",
        table=[
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in audit_payload["files"]
        ],
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "case-memory-final"),
        description=f"CASE_MEMORY.md for {doc_id}",
        path=memory_dir / "CASE_MEMORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "run-history-final"),
        description=f"RUN_HISTORY.md for {doc_id}",
        path=memory_dir / "RUN_HISTORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "document-text"),
        description=f"Generated document text for {doc_id}",
        path=document_path,
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "generation-report"),
        description=f"Generation report for {doc_id}",
        path=report_path,
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "schema-json-final"),
        description=f"Case schema JSON for {doc_id}",
        path=schema_path,
        language="json",
    )


def _publish_quality_artifacts(
    doc_id: str,
    report: dict[str, Any],
) -> None:
    create_table_artifact(
        key=_artifact_key(doc_id, "quality-scores"),
        description=f"Quality scores for {doc_id}",
        table=[
            {
                "section": section["section"],
                "score": section["score"],
                "verdict": section["verdict"],
                "revision": section["revision"],
                "word_count": section["word_count"],
                "issues": len(section["issues"]),
            }
            for section in report["sections"]
        ],
    )
    create_markdown_artifact(
        key=_artifact_key(doc_id, "quality-summary"),
        description=f"Quality summary for {doc_id}",
        markdown=format_markdown_report(report),
    )


def _run_summary_markdown(
    *,
    doc_id: str,
    document_path: Path,
    schema_path: Path,
    memory_dir: Path,
    report_path: Path,
    audit_path: Path,
) -> str:
    return "\n".join(
        [
            f"# Generated Document: `{doc_id}`",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Document text | `{document_path}` |",
            f"| Case schema | `{schema_path}` |",
            f"| Case memory | `{memory_dir / 'CASE_MEMORY.md'}` |",
            f"| Run history | `{memory_dir / 'RUN_HISTORY.md'}` |",
            f"| Generation report | `{report_path}` |",
            f"| File audit | `{audit_path}` |",
        ]
    )


def _publish_file_markdown(
    *,
    key: str,
    description: str,
    path: Path,
    language: str = "markdown",
) -> None:
    if not path.exists():
        create_markdown_artifact(
            key=key,
            description=description,
            markdown=f"# Missing Artifact\n\nFile was not found:\n\n`{path}`",
        )
        return

    text = path.read_text(encoding="utf-8")
    suffix = ""
    if len(text) > ARTIFACT_TEXT_LIMIT:
        text = text[:ARTIFACT_TEXT_LIMIT].rstrip()
        suffix = (
            "\n\n_Artifact truncated for Prefect UI display. "
            f"Open the local file for full content: `{path}`_"
        )
    create_markdown_artifact(
        key=key,
        description=description,
        markdown=f"# `{path.name}`\n\n```{language}\n{text}\n```{suffix}",
    )


def _artifact_key(*parts: object) -> str:
    try:
        run_id = str(get_run_context().flow_run.id)[:8]
    except Exception:
        run_id = "local"
    raw = "-".join([run_id, *(str(part) for part in parts if part is not None)])
    key = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    return re.sub(r"-{2,}", "-", key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
