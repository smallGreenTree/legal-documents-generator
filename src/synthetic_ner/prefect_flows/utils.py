"""Prefect orchestration for audited synthetic NER generation runs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from prefect import get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.context import get_run_context
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from pydantic import Field, create_model

from src.synthetic_ner.case import resolve_counts
from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.engine import (
    build_runtime_context,
    resolve_document_inputs,
    resolve_project_path,
    resolve_schema_for_document,
)
from src.synthetic_ner.schema import counter_from_doc_id, doc_id_prefix, make_doc_id
from src.synthetic_ner.tasks.document_generation.orchestrator import run_document_graph
from src.synthetic_ner.tasks.document_quality.quality_overview import (
    build_quality_overview,
    fetch_langfuse_rubric_summary,
    format_audit_confidence_markdown,
    format_model_workflow_markdown,
    format_run_health_markdown,
)
from src.synthetic_ner.tasks.document_quality.quality_report import (
    build_quality_report,
    format_markdown_report,
    load_quality_scoring_config,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.utils import load_config

ARTIFACT_TEXT_LIMIT = 24_000
DEFAULT_GENERATED_CASE_CONFIG = "config_case/generated/case.yaml"
MAX_PERSON_SETUP_ROWS = 8
DEFAULT_SCENARIO_OPTIONS = {
    "procurement_fraud": "Procurement fraud and corruption",
    "eu_subsidy_fraud": "Non-procurement expenditure fraud (subsidy fraud)",
}

ScenarioChoice = Literal[
    "Procurement fraud and corruption",
    "Non-procurement expenditure fraud (subsidy fraud)",
]
DocTypeChoice = Literal["indictment", "court_decision"]
PersonCountChoice = Literal[1, 2, 3, 4, 5, 6, 7, 8]
PersonGroupChoice = Literal["defendant", "collateral"]
NationalityChoice = Literal[
    "GB",
    "DE",
    "FR",
    "IT",
    "NL",
    "CZ",
    "PL",
    "ES",
    "PT",
    "BE",
    "AT",
    "SE",
    "DK",
    "FI",
    "HU",
    "RO",
    "BG",
    "GR",
    "HR",
    "SK",
    "SI",
]
TitleChoice = Literal["No title", "Dr", "Mr", "Ms", "Mrs", "Prof"]
SurfaceFormsChoice = Literal[1, 2, 3, 4]
OrgCountChoice = Literal[0, 1, 2, 3, 4, 5]
OrganisationGroupChoice = Literal["charged", "associated"]


def resolve_flow_project_root(project_root: str | None) -> Path:
    """Resolve the project root for Prefect flows in this package."""
    return (
        Path(project_root).expanduser().resolve()
        if project_root
        else Path(__file__).resolve().parents[3]
    )


class ScenarioReviewInput(RunInput):
    scenario: ScenarioChoice = "Procurement fraud and corruption"
    documents: int = 1
    doc_type: DocTypeChoice = "indictment"
    person_entities: PersonCountChoice = 5
    charged_orgs: OrgCountChoice = 2
    associated_orgs: OrgCountChoice = 1


class PersonSetupReviewInput(RunInput):
    pass


class OrganisationSetupReviewInput(RunInput):
    pass


class EntityReviewInput(RunInput):
    document_json: str = ""
    refresh_counts: bool = True


class QualityDocumentSelectionInput(RunInput):
    doc_id: str = ""
    candidate_documents: str = ""


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
    template: str | None = None,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
    quality_config: str | None = None,
    publish_artifacts: bool = True,
) -> dict[str, Any]:
    """Resolve the selected scenario and publish the input files used by the run."""
    scenario = _build_scenario(
        project_root=project_root,
        case_config=case_config,
        template=template,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
        quality_config=quality_config,
    )
    if publish_artifacts:
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
                "Open the parent flow run page, use the resume control, review the "
                "case setup fields, and submit the form to continue. Use the Prefect "
                "run controls if you need to cancel the run.",
                "",
                f"The pause timeout is `{timeout_seconds}` seconds.",
            ]
        ),
    )
    get_run_logger().info(
        "Human scenario review requested. Resume the parent flow run with the "
        "completed case setup form."
    )


def review_selected_scenario(
    *,
    project_root: Path,
    scenario: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Pause the flow so a reviewer can approve or alter the selected scenario."""
    publish_scenario_review_request(scenario, timeout_seconds)
    review_input = _required_prefilled_input_model(
        ScenarioReviewInput,
        description=(
            "Configure the case before Faker and the LLM run. Select one of the "
            "supported scenarios, choose person and organisation counts, then submit "
            "the form to continue."
        ),
        **_case_setup_initial_data(scenario),
        documents=scenario["documents"],
        doc_type=_doc_type_choice(scenario.get("doc_type")),
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="scenario-review",
    )
    if response is None:
        return scenario

    reviewed_scenario = _build_scenario(
        project_root=project_root,
        case_config=scenario["case_config"],
        template=None,
        documents=response.documents,
        doc_type=response.doc_type,
        fraud_type=_reviewed_fraud_type(response, scenario),
        from_schema=scenario["from_schema"],
        quality_config=scenario.get("quality_config"),
    )
    person_specs = review_person_setup(
        scenario=reviewed_scenario,
        person_count=response.person_entities,
        timeout_seconds=timeout_seconds,
    )
    organisation_specs = review_organisation_setup(
        scenario=reviewed_scenario,
        charged_count=response.charged_orgs,
        associated_count=response.associated_orgs,
        timeout_seconds=timeout_seconds,
    )
    reviewed_scenario["case_setup"] = _case_setup_from_review_response(
        response,
        reviewed_scenario,
        person_specs,
        organisation_specs,
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


def review_person_setup(
    *,
    scenario: dict[str, Any],
    person_count: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Pause with exactly one row per configured person entity."""
    initial_specs = _initial_person_specs_for_setup(scenario, person_count)
    review_input = _person_setup_review_input_model(initial_specs)
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="person-setup-review",
    )
    if response is None:
        return initial_specs
    return _person_specs_from_review_response(response, len(initial_specs))


def review_organisation_setup(
    *,
    scenario: dict[str, Any],
    charged_count: int,
    associated_count: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Pause with exactly one row per configured organisation entity."""
    initial_specs = _initial_organisation_specs_for_setup(
        scenario,
        charged_count,
        associated_count,
    )
    if not initial_specs:
        return []
    review_input = _organisation_setup_review_input_model(initial_specs)
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="organisation-setup-review",
    )
    if response is None:
        return initial_specs
    return _organisation_specs_from_review_response(response, len(initial_specs))


def _required_prefilled_input_model(
    base_cls: type[RunInput],
    *,
    description: str,
    field_types: dict[str, Any] | None = None,
    **initial_data: Any,
) -> type[RunInput]:
    fields: dict[str, tuple[Any, Any]] = {}
    field_types = field_types or {}
    for key, value in initial_data.items():
        original_field = base_cls.model_fields.get(key)
        field_type = (
            field_types[key]
            if key in field_types
            else original_field.annotation
            if original_field
            else type(value)
        )
        fields[key] = (
            field_type,
            Field(..., json_schema_extra={"default": value}),
        )

    model = create_model(base_cls.__name__, **fields, __base__=base_cls)
    model._description = description
    return model


@task(name="case-yaml-construction")
def construct_case_yaml_from_setup(
    *,
    project_root: Path,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Write the Stage 1 case setup into an effective case.yaml for Stage 2."""
    case_setup = scenario.get("case_setup")
    if not isinstance(case_setup, dict):
        return scenario

    source_case_config = scenario["case_config"]
    source_path = resolve_project_path(project_root, source_case_config)
    generated_case_config = (
        case_setup.get("generated_case_config") or DEFAULT_GENERATED_CASE_CONFIG
    )
    generated_path = resolve_project_path(project_root, generated_case_config)
    source_raw = load_config(source_path)
    if not isinstance(source_raw, dict):
        raise SystemExit(f"{source_path} must load into a top-level mapping")

    generated_raw = _apply_case_setup_to_config(source_raw, scenario, case_setup)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(
        yaml.safe_dump(
            generated_raw,
            sort_keys=False,
            allow_unicode=True,
            width=88,
        ),
        encoding="utf-8",
    )

    rebuilt = _build_scenario(
        project_root=project_root,
        case_config=generated_case_config,
        template=scenario.get("template") or None,
        documents=scenario["documents"],
        doc_type=scenario["doc_type"],
        fraud_type=scenario["fraud_type"],
        from_schema=scenario["from_schema"],
        quality_config=scenario.get("quality_config"),
    )
    rebuilt["source_case_config"] = source_case_config
    rebuilt["case_setup"] = case_setup
    _publish_generated_case_yaml_artifact(generated_path, rebuilt)
    get_run_logger().info("Constructed generated case.yaml at %s", generated_path)
    return rebuilt


@task(name="entity-review-request")
def publish_entity_review_request(
    document_payload: dict[str, Any],
    scenario_summary: str,
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
                "## Scenario",
                "",
                scenario_summary,
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
                "Open the parent flow run page, use the resume control, edit "
                "`document_json` only if needed, and submit the form to accept. Use "
                "the Prefect run controls if you need to cancel the run.",
                "",
                f"The pause timeout is `{timeout_seconds}` seconds.",
            ]
        ),
    )
    get_run_logger().info(
        "Human entity review requested. Resume the parent flow run with the accepted "
        "document input JSON."
    )


def review_document_entities(
    context: Any,
    document: Any,
    timeout_seconds: int,
) -> Any:
    """Pause the flow so a reviewer can approve or edit resolved document inputs."""
    document_payload = _document_to_payload(document, context=context)
    initial_document_json = json.dumps(document_payload, indent=2, ensure_ascii=False)
    initial_document_payload = json.loads(initial_document_json)
    scenario_summary = _entity_review_scenario_summary(context)
    publish_entity_review_request(document_payload, scenario_summary, timeout_seconds)
    review_input = _required_prefilled_input_model(
        EntityReviewInput,
        description=_entity_review_description(context, document_payload),
        document_json=initial_document_json,
        refresh_counts=True,
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="entity-review",
    )
    if response is None:
        return document

    if _document_json_matches_payload(
        response.document_json,
        initial_document_payload,
    ):
        return document

    reviewed_document = _document_from_review_json(response.document_json)
    if response.refresh_counts:
        reviewed_document.counts_list = resolve_counts(
            context.app_config.fraud_statutes,
            context.case_cfg,
            context.doc_type,
            context.fraud_type,
            reviewed_document.defendants,
            reviewed_document.charged_orgs,
            reviewed_document.amounts,
            reviewed_document.metadata.get("offence_period"),
        )
    _publish_entity_artifacts(reviewed_document)
    _publish_document_inputs_artifact(
        key=_artifact_key("entity-review-applied-document-inputs"),
        description="Document inputs after human entity review edits",
        document_payload=_document_to_payload(reviewed_document, context=context),
    )
    return reviewed_document


@task(name="quality-document-selection")
def publish_quality_document_selection(
    context: Any,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Collect candidate document ids before pausing for quality scoring input."""
    candidates = _quality_document_candidates(context)
    get_run_logger().info(
        "Quality document selection has %s candidate(s). Pause timeout is %s seconds.",
        len(candidates),
        timeout_seconds,
    )
    return candidates


def select_quality_document(
    *,
    context: Any,
    doc_id: str | None,
    timeout_seconds: int,
    review_document_selection: bool = True,
) -> str:
    """Resolve or request the document id to analyze in the quality flow."""
    requested_doc_id = (doc_id or "").strip()
    if requested_doc_id and not review_document_selection:
        _ensure_quality_document_exists(context, requested_doc_id)
        return requested_doc_id

    candidates = publish_quality_document_selection(context, timeout_seconds)
    if not candidates:
        raise SystemExit(
            "No generated documents are available for this doc_type/fraud_type."
        )

    review_input = _required_prefilled_input_model(
        QualityDocumentSelectionInput,
        description=(
            "Select the generated document to analyze. Enter one doc_id from "
            "candidate_documents, then submit the form to score it."
        ),
        doc_id=requested_doc_id,
        candidate_documents=_quality_candidate_summary(candidates),
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="quality-document-selection",
    )
    if response is None:
        raise SystemExit("Document quality selection timed out without a doc_id.")

    selected_doc_id = response.doc_id.strip()
    if not selected_doc_id:
        raise SystemExit("Document quality selection requires a doc_id.")
    _ensure_quality_document_exists(context, selected_doc_id)
    get_run_logger().info("Selected document for quality analysis: %s", selected_doc_id)
    return selected_doc_id


def _quality_document_candidates(context: Any) -> list[dict[str, Any]]:
    prefix = doc_id_prefix(context.doc_type, context.fraud_type)
    doc_ids: set[str] = set()

    if context.output_dir.exists():
        doc_ids.update(
            path.name
            for path in context.output_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )

    partial_root = context.output_dir / "_partial"
    if partial_root.exists():
        doc_ids.update(
            path.name
            for path in partial_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )

    if context.memory_dir.exists():
        doc_ids.update(
            path.name.removeprefix("case_")
            for path in context.memory_dir.iterdir()
            if path.is_dir()
            and path.name.startswith(f"case_{prefix}")
        )

    if context.schema_dir.exists():
        doc_ids.update(
            path.stem
            for path in context.schema_dir.glob(f"{prefix}*.json")
            if path.is_file()
        )

    return [
        _quality_document_candidate_record(context, doc_id)
        for doc_id in sorted(
            doc_ids,
            key=lambda item: _quality_document_sort_key(context, item),
        )
    ]


def _quality_document_sort_key(context: Any, doc_id: str) -> tuple[int, str]:
    try:
        counter = counter_from_doc_id(doc_id, context.doc_type, context.fraud_type)
    except ValueError:
        counter = -1
    return (-counter, doc_id)


def _quality_document_candidate_record(context: Any, doc_id: str) -> dict[str, Any]:
    doc_dir = context.output_dir / doc_id
    partial_sections = context.output_dir / "_partial" / doc_id / "sections"
    memory_dir = context.memory_dir / f"case_{doc_id}"
    schema_path = context.schema_dir / f"{doc_id}.json"
    document_path = doc_dir / f"{doc_id}.txt"
    generation_report_path = doc_dir / "generation_report.md"

    return {
        "doc_id": doc_id,
        "final_document": document_path.exists(),
        "generation_report": generation_report_path.exists(),
        "case_memory": (memory_dir / "CASE_MEMORY.md").exists(),
        "schema": schema_path.exists(),
        "section_artifacts": _section_artifact_count(partial_sections),
        "last_modified": _latest_modified_at(
            [doc_dir, context.output_dir / "_partial" / doc_id, memory_dir, schema_path]
        ),
    }


def _section_artifact_count(partial_sections: Path) -> int:
    if not partial_sections.exists():
        return 0
    return sum(1 for path in partial_sections.iterdir() if path.is_dir())


def _latest_modified_at(paths: list[Path]) -> str:
    latest_timestamp = 0.0
    for path in paths:
        if not path.exists():
            continue
        latest_timestamp = max(latest_timestamp, path.stat().st_mtime)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.exists():
                    latest_timestamp = max(latest_timestamp, child.stat().st_mtime)
    if not latest_timestamp:
        return ""
    return datetime.fromtimestamp(latest_timestamp, UTC).isoformat(timespec="seconds")


def _quality_candidate_summary(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No generated documents found for this doc_type/fraud_type."
    lines = []
    for candidate in candidates:
        readiness = []
        if candidate["final_document"]:
            readiness.append("final document")
        if candidate["generation_report"]:
            readiness.append("generation report")
        if candidate["case_memory"]:
            readiness.append("case memory")
        readiness_text = ", ".join(readiness) if readiness else "incomplete artifacts"
        lines.append(
            f"{candidate['doc_id']} | sections={candidate['section_artifacts']} | "
            f"{readiness_text} | modified={candidate['last_modified'] or 'unknown'}"
        )
    return "\n".join(lines)


def _quality_candidate_markdown_table(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No generated documents found for this doc_type/fraud_type."

    lines = [
        "| Document ID | Final | Report | Memory | Schema | Sections | Modified |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for candidate in candidates:
        lines.append(
            "| "
            f"`{candidate['doc_id']}` | "
            f"`{candidate['final_document']}` | "
            f"`{candidate['generation_report']}` | "
            f"`{candidate['case_memory']}` | "
            f"`{candidate['schema']}` | "
            f"{candidate['section_artifacts']} | "
            f"`{candidate['last_modified'] or 'unknown'}` |"
        )
    return "\n".join(lines)


def _ensure_quality_document_exists(context: Any, doc_id: str) -> None:
    candidates = _quality_document_candidates(context)
    candidate_ids = {candidate["doc_id"] for candidate in candidates}
    if doc_id in candidate_ids:
        return
    known = ", ".join(sorted(candidate_ids)) or "none"
    expected_prefix = doc_id_prefix(context.doc_type, context.fraud_type)
    raise SystemExit(
        f"Unknown document id for this quality context: {doc_id}. "
        f"Expected prefix {expected_prefix!r}. Available documents: {known}."
    )


def _build_scenario(
    *,
    project_root: Path,
    case_config: str,
    template: str | None = None,
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
    fraud_statutes = (
        case_raw.get("fraud_statutes", {}) if isinstance(case_raw, dict) else {}
    )
    workflow = root_raw.get("workflow", {}) if isinstance(root_raw, dict) else {}
    model_routing = root_raw.get("model_routing", {}) if isinstance(root_raw, dict) else {}
    generation = root_raw.get("generation", {}) if isinstance(root_raw, dict) else {}
    scenario_options = _scenario_options(workflow, fraud_statutes)

    selected_doc_type = doc_type or profile.get("doc_type")
    selected_fraud_type = _resolve_scenario_choice(
        fraud_type or profile.get("fraud_type"),
        scenario_options,
        workflow,
    )
    selected_documents = documents if documents is not None else profile.get("documents")
    prompts_config = workflow.get("prompts_config_path", "prompts/workflow_prompts.yaml")
    prompts_path = resolve_project_path(project_root, prompts_config)
    prompts_raw = load_config(prompts_path) if prompts_path.exists() else {}
    template_path = None
    if template:
        template_path = resolve_project_path(project_root, template)
    elif selected_doc_type:
        template_path = project_root / "templates" / f"en_{selected_doc_type}.j2"

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
        "template": template or "",
        "documents": selected_documents,
        "doc_type": selected_doc_type,
        "fraud_type": selected_fraud_type,
        "scenario_name": scenario_options.get(selected_fraud_type, selected_fraud_type),
        "scenario_options": scenario_options,
        "from_schema": from_schema,
        "quality_config": quality_config,
        "input_files": input_files,
        "profile": profile,
        "case": case_section,
        "fraud_statutes": fraud_statutes,
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
    publish_artifacts: bool = True,
) -> Any:
    """Load config files and build the runtime context."""
    load_env_files(project_root)
    args = _build_args(
        case_config=scenario["case_config"],
        template=scenario.get("template_path") or "",
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
    if publish_artifacts:
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
    rubric_summary = fetch_langfuse_rubric_summary(context, doc_id)
    overview = build_quality_overview(
        context=context,
        doc_id=doc_id,
        quality_report=report,
        rubric_summary=rubric_summary,
    )
    _publish_quality_analysis_artifact(doc_id, report, overview)
    get_run_logger().info(
        "Scored %s quality as %s (%s)",
        doc_id,
        report["overall_score"],
        report["verdict"],
    )
    return report


def _build_args(
    *,
    case_config: str,
    template: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
) -> Namespace:
    return Namespace(
        case_config=case_config,
        template=template,
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


def _scenario_options(
    workflow: Any,
    fraud_statutes: Any,
) -> dict[str, str]:
    dialogue = _prefect_dialogue_config(workflow)
    configured = dialogue.get("scenario_options", {})
    available_statutes = set(fraud_statutes) if isinstance(fraud_statutes, dict) else set()
    options: dict[str, str] = {}
    if isinstance(configured, dict):
        for key, label in configured.items():
            key_text = str(key)
            if available_statutes and key_text not in available_statutes:
                continue
            options[key_text] = str(label)

    if not options:
        for key, label in DEFAULT_SCENARIO_OPTIONS.items():
            if available_statutes and key not in available_statutes:
                continue
            options[key] = label

    if not options and isinstance(fraud_statutes, dict):
        options = {str(key): str(key).replace("_", " ") for key in fraud_statutes}
    return options


def _resolve_scenario_choice(
    value: Any,
    scenario_options: dict[str, str],
    workflow: Any,
) -> str | None:
    if not scenario_options:
        return str(value) if value else None

    cleaned = str(value or "").strip()
    if cleaned in scenario_options:
        return cleaned

    lowered = cleaned.casefold()
    for key, label in scenario_options.items():
        if lowered and lowered in {key.casefold(), label.casefold()}:
            return key

    default_scenario = _prefect_dialogue_config(workflow).get("default_scenario")
    if default_scenario in scenario_options:
        return str(default_scenario)
    return next(iter(scenario_options))


def _prefect_dialogue_config(workflow: Any) -> dict[str, Any]:
    if not isinstance(workflow, dict):
        return {}
    dialogue = workflow.get("prefect_dialogue", {})
    return dialogue if isinstance(dialogue, dict) else {}


def _reviewed_fraud_type(response: ScenarioReviewInput, scenario: dict[str, Any]) -> str:
    scenario_options = scenario.get("scenario_options", {})
    if not isinstance(scenario_options, dict):
        scenario_options = {}
    selected = _resolve_scenario_choice(
        response.scenario,
        scenario_options,
        scenario.get("workflow", {}),
    )
    return selected or scenario["fraud_type"]


def _case_setup_initial_data(scenario: dict[str, Any]) -> dict[str, Any]:
    case = scenario.get("case", {})
    cast = case.get("cast", {}) if isinstance(case, dict) else {}
    person_specs = _combined_person_specs(cast)
    scenario_options = scenario.get("scenario_options", {})
    return {
        "scenario": _scenario_choice_label(scenario, scenario_options),
        "person_entities": _person_count_choice(len(person_specs)),
        "charged_orgs": _org_count_choice(
            cast.get("charged_orgs", 0) if isinstance(cast, dict) else 0
        ),
        "associated_orgs": _org_count_choice(
            cast.get("associated_orgs", 0) if isinstance(cast, dict) else 0
        ),
    }


def _case_setup_from_review_response(
    response: ScenarioReviewInput,
    scenario: dict[str, Any],
    person_specs: list[dict[str, Any]],
    organisation_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    person_count = _positive_review_int(response.person_entities, "person_entities")
    charged_orgs, associated_orgs = _organisation_group_counts(
        organisation_specs,
    )
    if (charged_orgs or associated_orgs) and not any(
        spec["group"] == "defendant" for spec in person_specs
    ):
        raise SystemExit(
            "At least one defendant PERSON spec is required when organisation "
            "auto-generation is enabled."
        )

    return {
        "generated_case_config": _generated_case_config_from_scenario(scenario),
        "scenario": scenario["fraud_type"],
        "person_entities": person_count,
        "person_specs": person_specs,
        "charged_orgs": charged_orgs,
        "associated_orgs": associated_orgs,
        "organisation_entities": len(organisation_specs),
        "organisation_specs": organisation_specs,
    }


def _generated_case_config_from_scenario(scenario: dict[str, Any]) -> str:
    dialogue = _prefect_dialogue_config(scenario.get("workflow", {}))
    return str(dialogue.get("generated_case_config") or DEFAULT_GENERATED_CASE_CONFIG)


def _scenario_choice_label(
    scenario: dict[str, Any],
    scenario_options: Any,
) -> ScenarioChoice:
    if isinstance(scenario_options, dict):
        label = scenario_options.get(scenario.get("fraud_type"))
    else:
        label = None
    if label == "Non-procurement expenditure fraud (subsidy fraud)":
        return "Non-procurement expenditure fraud (subsidy fraud)"
    return "Procurement fraud and corruption"


def _doc_type_choice(value: Any) -> DocTypeChoice:
    return "court_decision" if value == "court_decision" else "indictment"


def _initial_person_specs_for_setup(
    scenario: dict[str, Any],
    person_count: int,
) -> list[dict[str, Any]]:
    case = scenario.get("case", {})
    cast = case.get("cast", {}) if isinstance(case, dict) else {}
    person_specs = _combined_person_specs(cast)
    return _resize_person_specs(person_specs, person_count)


def _initial_organisation_specs_for_setup(
    scenario: dict[str, Any],
    charged_count: int,
    associated_count: int,
) -> list[dict[str, Any]]:
    case = scenario.get("case", {})
    cast = case.get("cast", {}) if isinstance(case, dict) else {}
    organisation_specs = _combined_organisation_specs(cast)
    return _resize_organisation_specs(
        organisation_specs,
        charged_count,
        associated_count,
        default_country=_default_organisation_country(cast),
    )


def _person_setup_review_input_model(
    person_specs: list[dict[str, Any]],
) -> type[RunInput]:
    person_count = len(person_specs)
    return _required_prefilled_input_model(
        PersonSetupReviewInput,
        description=(
            f"Configure {person_count} PERSON entit"
            f"{'y' if person_count == 1 else 'ies'}. Each row will become one "
            "Faker-generated person in case.yaml."
        ),
        field_types=_person_row_field_types(person_count),
        **_person_row_initial_data(person_specs, person_count),
    )


def _organisation_setup_review_input_model(
    organisation_specs: list[dict[str, Any]],
) -> type[RunInput]:
    organisation_count = len(organisation_specs)
    return _required_prefilled_input_model(
        OrganisationSetupReviewInput,
        description=(
            f"Configure {organisation_count} organisation entit"
            f"{'y' if organisation_count == 1 else 'ies'}. Each row captures the "
            "organisation group and country."
        ),
        field_types=_organisation_row_field_types(organisation_count),
        **_organisation_row_initial_data(organisation_specs, organisation_count),
    )


def _person_row_field_types(person_count: int) -> dict[str, Any]:
    field_types: dict[str, Any] = {}
    for index in range(1, person_count + 1):
        field_types[f"person_{index}_group"] = PersonGroupChoice
        field_types[f"person_{index}_nationality"] = NationalityChoice
        field_types[f"person_{index}_title"] = TitleChoice
        field_types[f"person_{index}_surface_forms"] = SurfaceFormsChoice
    return field_types


def _organisation_row_field_types(organisation_count: int) -> dict[str, Any]:
    field_types: dict[str, Any] = {}
    for index in range(1, organisation_count + 1):
        field_types[f"organisation_{index}_group"] = OrganisationGroupChoice
        field_types[f"organisation_{index}_country"] = NationalityChoice
    return field_types


def _person_row_initial_data(
    person_specs: list[dict[str, Any]],
    person_count: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index in range(1, person_count + 1):
        spec = (
            person_specs[index - 1]
            if index <= len(person_specs)
            else _default_person_spec(person_specs)
        )
        rows[f"person_{index}_group"] = _group_choice(spec.get("group"))
        rows[f"person_{index}_nationality"] = _nationality_choice(
            spec.get("nationality")
        )
        rows[f"person_{index}_title"] = _title_choice(spec.get("title"))
        rows[f"person_{index}_surface_forms"] = _surface_forms_choice(
            spec.get("surface_forms")
        )
    return rows


def _organisation_row_initial_data(
    organisation_specs: list[dict[str, Any]],
    organisation_count: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index in range(1, organisation_count + 1):
        spec = organisation_specs[index - 1]
        rows[f"organisation_{index}_group"] = _organisation_group_choice(
            spec.get("group")
        )
        rows[f"organisation_{index}_country"] = _nationality_choice(
            spec.get("country")
        )
    return rows


def _person_specs_from_review_response(
    response: ScenarioReviewInput,
    person_count: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index in range(1, person_count + 1):
        specs.append(
            {
                "group": getattr(response, f"person_{index}_group"),
                "nationality": getattr(response, f"person_{index}_nationality"),
                "title": _title_config(getattr(response, f"person_{index}_title")),
                "surface_forms": _positive_review_int(
                    getattr(response, f"person_{index}_surface_forms"),
                    f"person_{index}_surface_forms",
                ),
            }
        )
    return specs


def _organisation_specs_from_review_response(
    response: Any,
    organisation_count: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index in range(1, organisation_count + 1):
        specs.append(
            {
                "group": _organisation_group_choice(
                    getattr(response, f"organisation_{index}_group")
                ),
                "country": _nationality_choice(
                    getattr(response, f"organisation_{index}_country")
                ),
            }
        )
    return specs


def _group_choice(value: Any) -> PersonGroupChoice:
    return "collateral" if value == "collateral" else "defendant"


def _organisation_group_choice(value: Any) -> OrganisationGroupChoice:
    cleaned = str(value or "").strip().lower()
    if cleaned in {"associated", "associated_org", "associated_orgs"}:
        return "associated"
    return "charged"


def _nationality_choice(value: Any) -> NationalityChoice:
    allowed = set(NationalityChoice.__args__)
    return value if value in allowed else "GB"


def _title_choice(value: Any) -> TitleChoice:
    return value if value in set(TitleChoice.__args__) else "No title"


def _title_config(value: TitleChoice) -> str:
    return "" if value == "No title" else value


def _surface_forms_choice(value: Any) -> SurfaceFormsChoice:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return min(max(parsed, 1), 4)


def _person_count_choice(value: Any) -> PersonCountChoice:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return min(max(parsed, 1), MAX_PERSON_SETUP_ROWS)


def _org_count_choice(value: Any) -> OrgCountChoice:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return min(max(parsed, 0), 5)


def _combined_person_specs(cast: Any) -> list[dict[str, Any]]:
    if not isinstance(cast, dict):
        return []
    specs: list[dict[str, Any]] = []
    for group, key in (("defendant", "defendants"), ("collateral", "collateral")):
        records = cast.get(key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            spec = {"group": group, **record}
            specs.append(spec)
    return specs


def _combined_organisation_specs(cast: Any) -> list[dict[str, Any]]:
    if not isinstance(cast, dict):
        return []
    records = cast.get("organisation_specs", [])
    if not isinstance(records, list):
        return []
    specs: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        specs.append(
            {
                "group": _organisation_group_choice(record.get("group")),
                "country": _nationality_choice(record.get("country")),
            }
        )
    return specs


def _person_specs_yaml(person_specs: list[dict[str, Any]]) -> str:
    return yaml.safe_dump(
        person_specs,
        sort_keys=False,
        allow_unicode=True,
        width=80,
    ).strip()


def _parse_person_specs_yaml(value: str) -> list[dict[str, Any]]:
    try:
        loaded = yaml.safe_load(value) if value.strip() else []
    except yaml.YAMLError as exc:
        raise SystemExit(f"person_specs_yaml is invalid YAML: {exc}") from exc
    if not isinstance(loaded, list):
        raise SystemExit("person_specs_yaml must be a YAML list of person specs.")

    specs: list[dict[str, Any]] = []
    for index, item in enumerate(loaded):
        path = f"person_specs_yaml[{index}]"
        if not isinstance(item, dict):
            raise SystemExit(f"{path} must be a mapping.")
        group = str(item.get("group", "defendant")).strip().lower()
        if group not in {"defendant", "collateral"}:
            raise SystemExit(f"{path}.group must be defendant or collateral.")
        nationality = str(item.get("nationality", "")).strip()
        if not nationality:
            raise SystemExit(f"{path}.nationality is required.")
        surface_forms = _positive_review_int(
            item.get("surface_forms"),
            f"{path}.surface_forms",
        )
        spec: dict[str, Any] = {
            "group": group,
            "nationality": nationality,
            "title": str(item.get("title", "")),
            "surface_forms": surface_forms,
        }
        if "variants" in item:
            spec["variants"] = item["variants"]
        specs.append(spec)
    return specs


def _resize_person_specs(
    person_specs: list[dict[str, Any]],
    person_count: int,
) -> list[dict[str, Any]]:
    count = _positive_review_int(person_count, "person_entities")
    resized = list(person_specs[:count])
    while len(resized) < count:
        resized.append(_default_person_spec(resized))
    return resized


def _resize_organisation_specs(
    organisation_specs: list[dict[str, Any]],
    charged_count: int,
    associated_count: int,
    *,
    default_country: NationalityChoice,
) -> list[dict[str, Any]]:
    charged_total = _non_negative_review_int(charged_count, "charged_orgs")
    associated_total = _non_negative_review_int(associated_count, "associated_orgs")
    charged_specs = [
        spec
        for spec in organisation_specs
        if _organisation_group_choice(spec.get("group")) == "charged"
    ]
    associated_specs = [
        spec
        for spec in organisation_specs
        if _organisation_group_choice(spec.get("group")) == "associated"
    ]
    return [
        *_resize_organisation_group_specs(charged_specs, "charged", charged_total, default_country),
        *_resize_organisation_group_specs(
            associated_specs,
            "associated",
            associated_total,
            default_country,
        ),
    ]


def _resize_organisation_group_specs(
    organisation_specs: list[dict[str, Any]],
    group: OrganisationGroupChoice,
    count: int,
    default_country: NationalityChoice,
) -> list[dict[str, Any]]:
    resized = [
        {
            "group": group,
            "country": _nationality_choice(spec.get("country")),
        }
        for spec in organisation_specs[:count]
    ]
    while len(resized) < count:
        resized.append({"group": group, "country": default_country})
    return resized


def _default_person_spec(existing_specs: list[dict[str, Any]]) -> dict[str, Any]:
    nationality = existing_specs[0].get("nationality", "GB") if existing_specs else "GB"
    return {
        "group": "defendant",
        "nationality": nationality,
        "title": "",
        "surface_forms": 1,
    }


def _default_organisation_country(cast: Any) -> NationalityChoice:
    person_specs = _combined_person_specs(cast)
    if person_specs:
        return _nationality_choice(person_specs[0].get("nationality"))
    return "GB"


def _organisation_group_counts(
    organisation_specs: list[dict[str, Any]],
) -> tuple[int, int]:
    charged_orgs = 0
    associated_orgs = 0
    for spec in organisation_specs:
        if _organisation_group_choice(spec.get("group")) == "associated":
            associated_orgs += 1
        else:
            charged_orgs += 1
    return charged_orgs, associated_orgs


def _positive_review_int(value: Any, path: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} must be a positive integer.") from exc
    if parsed <= 0:
        raise SystemExit(f"{path} must be a positive integer.")
    return parsed


def _non_negative_review_int(value: Any, path: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} must be a non-negative integer.") from exc
    if parsed < 0:
        raise SystemExit(f"{path} must be a non-negative integer.")
    return parsed


def _apply_case_setup_to_config(
    source_raw: dict[str, Any],
    scenario: dict[str, Any],
    case_setup: dict[str, Any],
) -> dict[str, Any]:
    generated = copy.deepcopy(source_raw)
    profile = generated.setdefault("profile", {})
    if not isinstance(profile, dict):
        raise SystemExit("profile must be a mapping in the source case config.")
    profile["doc_type"] = scenario["doc_type"]
    profile["fraud_type"] = scenario["fraud_type"]
    profile["documents"] = scenario["documents"]

    case = generated.setdefault("case", {})
    if not isinstance(case, dict):
        raise SystemExit("case must be a mapping in the source case config.")
    cast = case.setdefault("cast", {})
    if not isinstance(cast, dict):
        raise SystemExit("case.cast must be a mapping in the source case config.")

    defendants, collateral = _split_person_specs(case_setup["person_specs"])
    cast["defendants"] = defendants
    cast["collateral"] = collateral
    cast["charged_orgs"] = case_setup["charged_orgs"]
    cast["associated_orgs"] = case_setup["associated_orgs"]
    cast["organisation_specs"] = case_setup["organisation_specs"]
    cast.pop("address_surface_forms", None)

    return generated


def _split_person_specs(
    person_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    defendants: list[dict[str, Any]] = []
    collateral: list[dict[str, Any]] = []
    for spec in person_specs:
        record = {key: value for key, value in spec.items() if key != "group"}
        if spec["group"] == "collateral":
            collateral.append(record)
        else:
            defendants.append(record)
    return defendants, collateral


def _scenario_options_text(options: Any) -> str:
    if not isinstance(options, dict) or not options:
        return "No configured scenario options."
    return "\n".join(f"{key}: {label}" for key, label in options.items())


def _publish_generated_case_yaml_artifact(
    generated_path: Path,
    scenario: dict[str, Any],
) -> None:
    generated_person_count = len(
        _combined_person_specs(scenario.get("case", {}).get("cast", {}))
    )
    _publish_file_markdown(
        key=_artifact_key("generated-case-yaml"),
        description="Generated case.yaml constructed from Prefect Stage 1 inputs",
        path=generated_path,
        language="yaml",
    )
    create_markdown_artifact(
        key=_artifact_key("generated-case-summary"),
        description="Summary of the generated Stage 1 case YAML",
        markdown="\n".join(
            [
                "# Generated Case YAML",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Source case config | `{scenario.get('source_case_config')}` |",
                f"| Generated case config | `{scenario.get('case_config')}` |",
                f"| Scenario | `{scenario.get('scenario_name')}` |",
                f"| Fraud type | `{scenario.get('fraud_type')}` |",
                f"| Person entities | `{generated_person_count}` |",
                "",
                "Stage 2 will load this generated case config before Faker resolves "
                "names, organisations, addresses, dates, and counts.",
            ]
        ),
    )


def _scenario_review_initial_data(scenario: dict[str, Any]) -> dict[str, str]:
    return {
        "scenario_summary": _scenario_summary_text(scenario),
        "document_type_details": _document_type_details_text(scenario),
        "faker_generation_plan": _faker_generation_plan_text(scenario),
        "llm_language_plan": _llm_language_plan_text(scenario),
        "fraud_type_details": _fraud_type_details_text(scenario),
        "editable_fields_help": (
            "Edit scenario, documents, doc_type, person_entities, organisation counts, "
            "and organisation countries. The next pauses collect exactly one row per "
            "selected PERSON and ORG entity."
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
        f"document(s) for `{scenario.get('scenario_name')}` "
        f"(`{scenario.get('fraud_type')}`) using "
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
    options_text = _scenario_options_text(scenario.get("scenario_options", {}))
    if not isinstance(selected, list) or not selected:
        return (
            f"`fraud_type={fraud_type}` has no statute entries in this config.\n\n"
            f"Available Stage 1 scenario options:\n{options_text}"
        )
    offences = [
        f"{item.get('offence', 'unknown offence')} ({item.get('statute', 'unknown statute')})"
        for item in selected
        if isinstance(item, dict)
    ]
    return (
        f"`fraud_type={fraud_type}` loads these count/offence templates: "
        f"{_join_or_none(offences)}. Particulars are filled after Faker resolves "
        "defendant names, company names, and offence dates.\n\n"
        f"Available Stage 1 scenario options:\n{options_text}"
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


def _document_to_payload(document: Any, *, context: Any | None = None) -> dict[str, Any]:
    payload = {
        "defendants": document.defendants,
        "collateral": document.collateral,
        "charged_orgs": document.charged_orgs,
        "associated_orgs": document.associated_orgs,
        "metadata": document.metadata,
        "amounts": document.amounts,
        "counts_list": document.counts_list,
        "evidence_categories": document.evidence_categories,
    }
    scenario = _document_payload_scenario(context)
    if scenario:
        return {"scenario": scenario, **payload}
    return payload


def _document_payload_scenario(context: Any | None) -> dict[str, str]:
    if context is None:
        return {}
    fraud_type = str(getattr(context, "fraud_type", "") or "")
    doc_type = str(getattr(context, "doc_type", "") or "")
    if not fraud_type and not doc_type:
        return {}
    return {
        "id": fraud_type,
        "label": _scenario_label_for_fraud_type(fraud_type),
        "doc_type": doc_type,
    }


def _entity_review_scenario_summary(context: Any) -> str:
    fraud_type = str(getattr(context, "fraud_type", "") or "")
    doc_type = str(getattr(context, "doc_type", "") or "")
    scenario_label = _scenario_label_for_fraud_type(fraud_type)
    doc_label = doc_type.replace("_", " ") if doc_type else "unknown document type"
    return "\n".join(
        [
            f"Scenario: {scenario_label} ({fraud_type or 'unknown scenario id'})",
            f"Document type: {doc_label}",
        ]
    )


def _entity_review_description(context: Any, document_payload: dict[str, Any]) -> str:
    names_summary = _entity_names_summary(document_payload, separator="\n")
    surface_forms_summary = _entity_surface_forms_summary(
        document_payload,
        separator="\n",
    )
    return "\n\n".join(
        [
            "Review resolved people and organisations before document generation.",
            _entity_review_scenario_summary(context),
            f"Generated names:\n{names_summary}",
            f"Person surface forms:\n{surface_forms_summary}",
            (
                "Review the specifics below. Submit the form to accept the current "
                "document_json, or edit it before submitting."
            ),
        ]
    )


def _scenario_label_for_fraud_type(fraud_type: str) -> str:
    return DEFAULT_SCENARIO_OPTIONS.get(
        fraud_type,
        fraud_type.replace("_", " ") if fraud_type else "unknown scenario",
    )


def _entity_names_summary(
    document_payload: dict[str, Any],
    *,
    separator: str = " | ",
) -> str:
    parts = [
        _names_for_group("Defendants", document_payload.get("defendants", [])),
        _names_for_group("Collateral", document_payload.get("collateral", [])),
        _names_for_group("Charged organisations", document_payload.get("charged_orgs", [])),
        _names_for_group(
            "Associated organisations",
            document_payload.get("associated_orgs", []),
        ),
    ]
    return separator.join(part for part in parts if part)


def _entity_surface_forms_summary(
    document_payload: dict[str, Any],
    *,
    separator: str = " | ",
) -> str:
    parts = [
        _surface_forms_for_group("Defendants", document_payload.get("defendants", [])),
        _surface_forms_for_group("Collateral", document_payload.get("collateral", [])),
    ]
    return separator.join(part for part in parts if part)


def _surface_forms_for_group(label: str, records: Any) -> str:
    if not isinstance(records, list) or not records:
        return f"{label}: none"
    people: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "unnamed")
        forms = record.get("surface_forms_list", [])
        if isinstance(forms, list) and forms:
            form_text = ", ".join(str(form) for form in forms if form)
        else:
            form_text = "none"
        people.append(f"{name}: {form_text}")
    return f"{label}: {'; '.join(people) if people else 'none'}"


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
        raise SystemExit("Entity review requires document_json.")
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
    if not isinstance(payload.get("amounts"), dict):
        raise SystemExit("Entity review document_json.amounts must be an object.")
    evidence_categories = payload.get("evidence_categories", [])
    if not isinstance(evidence_categories, list) or not all(
        isinstance(category, str) for category in evidence_categories
    ):
        raise SystemExit(
            "Entity review document_json.evidence_categories must be a string list."
        )

    return DocumentInputs(
        defendants=payload["defendants"],
        collateral=payload["collateral"],
        charged_orgs=payload["charged_orgs"],
        associated_orgs=payload["associated_orgs"],
        metadata=payload["metadata"],
        amounts=payload["amounts"],
        counts_list=payload["counts_list"],
        evidence_categories=evidence_categories,
    )


def _document_json_matches_payload(document_json: str, payload: dict[str, Any]) -> bool:
    try:
        loaded = json.loads(document_json)
    except json.JSONDecodeError:
        return False
    return loaded == payload


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
    template_path = context.template_path
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


def _publish_quality_analysis_artifact(
    doc_id: str,
    report: dict[str, Any],
    overview: dict[str, Any],
) -> None:
    create_markdown_artifact(
        key=_artifact_key(doc_id, "document-quality-analysis"),
        description=f"Document quality analysis for {doc_id}",
        markdown=_quality_analysis_markdown(doc_id, report, overview),
    )


def _quality_analysis_markdown(
    doc_id: str,
    report: dict[str, Any],
    overview: dict[str, Any],
) -> str:
    report_with_links = _quality_report_with_langfuse_links(report, overview)
    return "\n\n".join(
        [
            f"# Document Quality Analysis: `{doc_id}`",
            _demote_markdown_headings(format_run_health_markdown(overview)),
            _demote_markdown_headings(format_model_workflow_markdown(overview)),
            _demote_markdown_headings(format_audit_confidence_markdown(overview)),
            _demote_markdown_headings(format_markdown_report(report_with_links)),
        ]
    ).rstrip() + "\n"


def _quality_report_with_langfuse_links(
    report: dict[str, Any],
    overview: dict[str, Any],
) -> dict[str, Any]:
    workflow = overview.get("model_workflow") or {}
    reference_by_section = {
        str(row.get("section")): row
        for row in workflow.get("prompt_response_refs", [])
        if row.get("section")
    }
    rubric_by_section = {
        str(row.get("section")): row
        for row in workflow.get("section_rubrics", [])
        if row.get("section")
    }
    enriched = dict(report)
    enriched_sections = []
    for section in report.get("sections", []):
        section_name = str(section.get("section") or "")
        reference = reference_by_section.get(section_name, {})
        rubric = rubric_by_section.get(section_name, {})
        enriched_section = dict(section)
        enriched_section["langfuse_url"] = (
            reference.get("text_url")
            or reference.get("critic_url")
            or rubric.get("langfuse_url")
        )
        enriched_sections.append(enriched_section)
    enriched["sections"] = enriched_sections
    return enriched


def _demote_markdown_headings(markdown: str) -> str:
    return re.sub(r"^(#{1,5}) ", r"#\1 ", markdown.strip(), flags=re.MULTILINE)


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
