"""Prefect flows for reproducible occurrence-level ground truth."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

from src.synthetic_ner.prefect_flows.utils import resolve_flow_project_root
from src.synthetic_ner.tasks.groundtruth import (
    discover_document_packages,
    generate_groundtruth_for_document,
)
from src.synthetic_ner.utils import resolve_project_path

DEFAULT_GROUNDTRUTH_CONTRACT_PATH = "groundtruth_contract.yaml"


@task(name="generate-validated-groundtruth")
def generate_validated_groundtruth(
    *,
    document_dir: str,
    contract_path: str,
) -> dict[str, Any]:
    """Build and publish one document's ground truth after complete validation."""
    result = generate_groundtruth_for_document(
        document_dir=Path(document_dir),
        contract_path=Path(contract_path),
    )
    get_run_logger().info(
        "Ground truth completed for %s with %s annotations%s",
        result["doc_id"],
        result["annotation_count"],
        " (reused)" if result.get("reused") else "",
    )
    return result


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
    return generate_validated_groundtruth(
        document_dir=str(resolved_document_dir),
        contract_path=str(resolved_contract_path),
    )


@flow(name="synthetic-ner-groundtruth-directory")
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
            f"No document packages with document_manifest.json found in {resolved_input_directory}"
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
