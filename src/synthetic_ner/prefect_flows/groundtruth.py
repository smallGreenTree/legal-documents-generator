"""Prefect flows for reproducible occurrence-level ground truth."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

from src.synthetic_ner.core.paths import resolve_project_path
from src.synthetic_ner.prefect_flows.utils import resolve_flow_project_root
from src.synthetic_ner.tasks.groundtruth import (
    calculate_groundtruth_offsets,
    discover_document_packages,
    groundtruth_failure_boundary,
    load_groundtruth_source,
    record_groundtruth_failure,
    select_used_initial_entities,
    validate_and_publish_groundtruth,
)

DEFAULT_GROUNDTRUTH_CONTRACT_PATH = "groundtruth_contract.yaml"


@task(name="load-frozen-groundtruth-inputs")
def load_frozen_groundtruth_inputs(document_dir: str) -> dict[str, Any]:
    """Load the final text and the initial entities saved before generation."""
    source = load_groundtruth_source(Path(document_dir))
    get_run_logger().info("Loaded frozen ground-truth inputs for %s", source["doc_id"])
    return source


@task(name="select-used-initial-entities")
def select_used_groundtruth_entities(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exact initial entity surfaces present in the final document."""
    references = select_used_initial_entities(source)
    get_run_logger().info(
        "Selected %s used initial entity surfaces for %s",
        len(references),
        source["doc_id"],
    )
    return references


@task(name="calculate-groundtruth-offsets")
def calculate_groundtruth_annotations(
    *,
    source: dict[str, Any],
    references: list[dict[str, Any]],
    contract_path: str,
) -> list[dict[str, Any]]:
    """Calculate every occurrence offset for the selected surfaces."""
    annotations = calculate_groundtruth_offsets(
        source=source,
        references=references,
        contract_path=contract_path,
    )
    get_run_logger().info(
        "Calculated %s occurrence offsets for %s",
        len(annotations),
        source["doc_id"],
    )
    return annotations


@task(name="validate-publish-groundtruth")
def publish_validated_groundtruth(
    *,
    source: dict[str, Any],
    references: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    contract_path: str,
) -> dict[str, Any]:
    """Validate all offsets and atomically publish the TSV and manifest."""
    result = validate_and_publish_groundtruth(
        source=source,
        references=references,
        annotation_rows=annotation_rows,
        contract_path=contract_path,
    )
    get_run_logger().info(
        "Ground truth completed for %s with %s annotations%s",
        result["doc_id"],
        result["annotation_count"],
        " (reused)" if result.get("reused") else "",
    )
    return result


@task(name="record-groundtruth-validation-failure")
def record_groundtruth_validation_failure(document_dir: str, issues: list[str]) -> None:
    """Write the validation error artifact for a failed modular stage."""
    record_groundtruth_failure(document_dir, issues)


@flow(name="synthetic-ner-groundtruth-document")
def generate_document_groundtruth(
    document_dir: str,
    project_root: str | None = None,
    contract_path: str = DEFAULT_GROUNDTRUTH_CONTRACT_PATH,
) -> dict[str, Any]:
    """Generate validated ground truth for one frozen document package."""
    resolved_project_root = resolve_flow_project_root(project_root)
    resolved_document_dir = resolve_project_path(resolved_project_root, document_dir).resolve()
    resolved_contract_path = resolve_project_path(resolved_project_root, contract_path).resolve()
    with groundtruth_failure_boundary(
        resolved_document_dir,
        failure_recorder=lambda issues: record_groundtruth_validation_failure(
            str(resolved_document_dir),
            issues,
        ),
    ):
        source = load_frozen_groundtruth_inputs(str(resolved_document_dir))
        references = select_used_groundtruth_entities(source)
        annotation_rows = calculate_groundtruth_annotations(
            source=source,
            references=references,
            contract_path=str(resolved_contract_path),
        )
        return publish_validated_groundtruth(
            source=source,
            references=references,
            annotation_rows=annotation_rows,
            contract_path=str(resolved_contract_path),
        )


@flow(name="synthetic-ner-generate-groundtruth")
def generate_groundtruth_directory(
    input_directory: str,
    project_root: str | None = None,
    contract_path: str = DEFAULT_GROUNDTRUTH_CONTRACT_PATH,
) -> dict[str, Any]:
    """Process every complete document package under a manually selected directory."""
    resolved_project_root = resolve_flow_project_root(project_root)
    resolved_input_directory = resolve_project_path(
        resolved_project_root,
        input_directory,
    ).resolve()
    document_directories = discover_document_packages(resolved_input_directory)
    if not document_directories:
        raise ValueError(
            "No document packages containing a matching .txt file and document_inputs.json "
            f"found in {resolved_input_directory}"
        )

    results: list[dict[str, Any]] = []
    for document_dir in document_directories:
        try:
            result = generate_document_groundtruth(
                document_dir=str(document_dir),
                project_root=str(resolved_project_root),
                contract_path=contract_path,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "doc_id": document_dir.name,
                "document_dir": str(document_dir),
                "error": str(exc),
            }
            get_run_logger().error(
                "Ground truth failed for %s: %s",
                document_dir.name,
                exc,
            )
        results.append(result)

    completed = sum(result.get("status") == "completed" for result in results)
    failed = len(results) - completed
    report = {
        "status": "completed" if failed == 0 else "failed",
        "input_directory": str(resolved_input_directory),
        "documents_discovered": len(results),
        "documents_completed": completed,
        "documents_failed": failed,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
    }
    report_path = resolved_input_directory / "groundtruth_batch_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    if failed:
        raise RuntimeError(
            f"Ground-truth batch processed all packages but {failed} of {len(results)} failed; "
            f"see {report_path}"
        )
    return report
