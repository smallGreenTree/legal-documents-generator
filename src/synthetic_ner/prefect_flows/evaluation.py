"""Prefect NER evaluation flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput

from src.synthetic_ner.prefect_flows.utils import (
    _artifact_key,
    resolve_flow_project_root,
)
from src.synthetic_ner.tasks.ner_evaluation import evaluate_document_ner


class EvaluationDocumentSelectionInput(RunInput):
    doc_id: str = ""
    available_generated_documents: str = ""


@flow(name="synthetic-ner-document-evaluation")
def evaluate_existing_document(
    doc_id: str | None = None,
    predictions_path: str | None = None,
    memory_path: str | None = None,
    calibration_mode: str = "apply_safe",
    case_config: str = "config_case/case_1.yaml",
    doc_type: str | None = None,
    fraud_type: str | None = None,
    project_root: str | None = None,
    review_document_selection: bool = True,
    review_timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Evaluate NER output for an existing generated document."""
    resolved_project_root = resolve_flow_project_root(project_root)
    del case_config, doc_type, fraud_type
    selected_doc_id = (
        _validate_requested_document(resolved_project_root, doc_id)
        if doc_id
        else select_evaluation_document(
            project_root=resolved_project_root,
            timeout_seconds=review_timeout_seconds,
            review_document_selection=review_document_selection,
        )
    )
    return run_ner_evaluation(
        project_root=resolved_project_root,
        doc_id=selected_doc_id,
        predictions_path=predictions_path,
        memory_path=memory_path,
        calibration_mode=calibration_mode,
    )


@task(name="evaluate-ner-predictions")
def run_ner_evaluation(
    *,
    project_root: Path,
    doc_id: str,
    predictions_path: str | None,
    memory_path: str | None,
    calibration_mode: str,
) -> dict[str, Any]:
    """Run deterministic NER evaluation and publish the report in Prefect."""
    result = evaluate_document_ner(
        project_root=project_root,
        doc_id=doc_id,
        predictions_path=predictions_path,
        memory_path=memory_path,
        calibration_mode=calibration_mode,
    )
    report_path = Path(result["paths"]["report"])
    create_markdown_artifact(
        key=_artifact_key(doc_id, "ner-evaluation-report"),
        description=f"NER evaluation report for {doc_id}",
        markdown=report_path.read_text(encoding="utf-8"),
    )
    get_run_logger().info(
        "Evaluated NER for %s: strict_f1=%.4f soft_f1=%.4f report=%s",
        doc_id,
        result["strict"]["metrics"]["f1"],
        result["soft"]["metrics"]["f1"],
        report_path,
    )
    return result


def select_evaluation_document(
    *,
    project_root: Path,
    timeout_seconds: int,
    review_document_selection: bool,
) -> str:
    """Request one generated document id for NER evaluation."""
    candidates = evaluation_document_candidates(project_root)
    if not candidates:
        raise SystemExit("No generated documents with groundtruth.tsv are available.")
    if not review_document_selection:
        selected = candidates[0]["doc_id"]
        get_run_logger().info("Selected latest document for NER evaluation: %s", selected)
        return selected

    review_input = EvaluationDocumentSelectionInput.with_initial_data(
        description=(
            "Select one generated document to evaluate. Doc Id is the generated "
            "document folder containing the rendered text and groundtruth.tsv. "
            "NER output is loaded from predictions_path, or from the default "
            "repo_ner_predictions.jsonl inside that document folder."
        ),
        doc_id="",
        available_generated_documents=evaluation_candidate_summary(candidates),
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="ner-evaluation-document-selection",
    )
    if response is None:
        raise SystemExit("NER evaluation selection timed out without a doc_id.")
    selected_doc_id = response.doc_id.strip()
    if not selected_doc_id:
        raise SystemExit("NER evaluation requires a doc_id.")
    return _validate_requested_document(project_root, selected_doc_id)


def evaluation_document_candidates(project_root: Path) -> list[dict[str, Any]]:
    output_root = project_root / "output"
    if not output_root.exists():
        return []
    candidates = []
    for output_dir in output_root.iterdir():
        if not output_dir.is_dir() or output_dir.name.startswith("_"):
            continue
        doc_id = output_dir.name
        document_path = output_dir / f"{doc_id}.txt"
        groundtruth_path = output_dir / "groundtruth.tsv"
        if not document_path.exists() or not groundtruth_path.exists():
            continue
        predictions_path = output_dir / "repo_ner_predictions.jsonl"
        memory_path = project_root / "memory" / f"case_{doc_id}" / "CASE_MEMORY.md"
        candidates.append(
            {
                "doc_id": doc_id,
                "document": document_path.exists(),
                "groundtruth": groundtruth_path.exists(),
                "ner_output": predictions_path.exists(),
                "memory": memory_path.exists(),
                "modified": max(
                    document_path.stat().st_mtime,
                    groundtruth_path.stat().st_mtime,
                    predictions_path.stat().st_mtime if predictions_path.exists() else 0,
                ),
            }
        )
    return sorted(candidates, key=lambda item: (-item["modified"], item["doc_id"]))


def evaluation_candidate_summary(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for candidate in candidates:
        readiness = [
            "document" if candidate["document"] else "missing document",
            "groundtruth" if candidate["groundtruth"] else "missing groundtruth",
            "NER output" if candidate["ner_output"] else "NER output via predictions_path",
        ]
        if candidate["memory"]:
            readiness.append("memory")
        lines.append(f"{candidate['doc_id']} | " + " | ".join(readiness))
    return "\n".join(lines)


def _validate_requested_document(project_root: Path, doc_id: str) -> str:
    output_dir = project_root / "output" / doc_id
    document_path = output_dir / f"{doc_id}.txt"
    groundtruth_path = output_dir / "groundtruth.tsv"
    missing = [
        str(path)
        for path in (document_path, groundtruth_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot evaluate requested document; missing artifact(s): "
            + ", ".join(missing)
        )
    return doc_id
