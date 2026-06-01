"""Prefect quality scoring flow."""

from __future__ import annotations

from typing import Any

from prefect import flow
from src.synthetic_ner.prefect_flows.utils import (
    ingest_configs,
    resolve_flow_project_root,
    review_selected_scenario,
    score_document_quality,
    select_quality_document,
    select_scenario,
)
from src.synthetic_ner.tasks.document_quality.quality_report import DEFAULT_QUALITY_CONFIG_PATH


@flow(name="synthetic-ner-document-quality")
def score_existing_document(
    doc_id: str | None = None,
    case_config: str = "config_case/case_1.yaml",
    quality_config: str = DEFAULT_QUALITY_CONFIG_PATH,
    doc_type: str | None = None,
    fraud_type: str | None = None,
    project_root: str | None = None,
    review_scenario: bool = False,
    review_document_selection: bool = True,
    review_timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Score an existing document without regenerating it."""
    resolved_project_root = resolve_flow_project_root(project_root)
    scenario = select_scenario(
        project_root=resolved_project_root,
        case_config=case_config,
        documents=1,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=None,
        quality_config=quality_config,
        publish_artifacts=False,
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
        publish_artifacts=False,
    )
    selected_doc_id = select_quality_document(
        context=context,
        doc_id=doc_id,
        timeout_seconds=review_timeout_seconds,
        review_document_selection=review_document_selection,
    )
    return score_document_quality(context, selected_doc_id, quality_config)
