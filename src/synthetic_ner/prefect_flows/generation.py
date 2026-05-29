"""Prefect generation flow."""

from __future__ import annotations

from prefect import flow, get_run_logger
from src.synthetic_ner.prefect_flows.utils import (
    _current_flow_run_id,
    audit_created_files,
    build_case_schema,
    ingest_configs,
    resolve_entities,
    resolve_flow_project_root,
    review_document_entities,
    review_selected_scenario,
    run_langgraph_langfuse,
    select_doc_id,
    select_scenario,
)


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
    resolved_project_root = resolve_flow_project_root(project_root)
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
