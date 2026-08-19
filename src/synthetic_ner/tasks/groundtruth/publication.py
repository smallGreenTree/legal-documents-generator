"""Ground-truth package discovery, generation, and atomic publication."""

import csv
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.synthetic_ner.document.inputs import (
    document_inputs_from_payload,
    load_document_inputs,
)
from src.synthetic_ner.tasks.groundtruth.annotations import (
    build_mention_annotations,
    read_groundtruth_tsv,
    validate_mention_annotations,
)
from src.synthetic_ner.tasks.groundtruth.contract import load_groundtruth_contract
from src.synthetic_ner.tasks.groundtruth.files import (
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from src.synthetic_ner.tasks.groundtruth.models import (
    GROUNDTRUTH_ERRORS_FILENAME,
    GROUNDTRUTH_FILENAME,
    GROUNDTRUTH_HEADER,
    GROUNDTRUTH_MANIFEST_FILENAME,
    GroundTruthContract,
    GroundTruthError,
    MentionAnnotation,
)
from src.synthetic_ner.tasks.groundtruth.references import (
    build_entity_references,
    select_present_entity_references,
)
from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME


def load_groundtruth_source(document_dir: Path | str) -> dict[str, Any]:
    """Load and fingerprint the frozen document and its saved initial inputs."""
    doc_dir = Path(document_dir)
    doc_id = doc_dir.name
    document_path = doc_dir / f"{doc_id}.txt"
    document_inputs_path = doc_dir / DOCUMENT_INPUTS_FILENAME
    _require_source_files(doc_id, document_path, document_inputs_path)
    document_inputs = load_document_inputs(document_inputs_path)
    return {
        "document_dir": str(doc_dir),
        "doc_id": doc_id,
        "document_text": document_path.read_text(encoding="utf-8"),
        "document_sha256": sha256_file(document_path),
        "document_inputs_sha256": sha256_file(document_inputs_path),
        "document_inputs": asdict(document_inputs),
    }


def select_used_initial_entities(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exact initial entity surfaces that occur in the frozen document."""
    document = document_inputs_from_payload(source["document_inputs"])
    return select_present_entity_references(
        source["document_text"],
        build_entity_references(document),
    )


def calculate_groundtruth_offsets(
    *,
    source: dict[str, Any],
    references: list[dict[str, Any]],
    contract_path: Path | str,
) -> list[dict[str, Any]]:
    """Calculate every exact occurrence offset for the selected entity surfaces."""
    contract = load_groundtruth_contract(contract_path)
    annotations = build_mention_annotations(
        doc_id=source["doc_id"],
        document_text=source["document_text"],
        references=references,
        contract=contract,
    )
    return [asdict(annotation) for annotation in annotations]


def validate_and_publish_groundtruth(
    *,
    source: dict[str, Any],
    references: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    contract_path: Path | str,
) -> dict[str, Any]:
    """Validate offsets and atomically publish or reuse completed ground truth."""
    doc_dir = Path(source["document_dir"])
    doc_id = source["doc_id"]
    contract = load_groundtruth_contract(contract_path)
    annotations = [MentionAnnotation(**row) for row in annotation_rows]
    integrity_issues = _source_integrity_issues(source)
    if integrity_issues:
        raise GroundTruthError(doc_id, integrity_issues)
    validate_mention_annotations(
        doc_id=doc_id,
        document_text=source["document_text"],
        annotations=annotations,
        references=references,
        contract=contract,
    )
    existing = _reuse_existing_groundtruth(
        doc_dir=doc_dir,
        doc_id=doc_id,
        document_text=source["document_text"],
        document_sha256=source["document_sha256"],
        document_inputs_sha256=source["document_inputs_sha256"],
        contract=contract,
        references=references,
    )
    if existing is not None:
        return existing
    return _publish_groundtruth(
        doc_dir=doc_dir,
        doc_id=doc_id,
        document_text=source["document_text"],
        document_sha256=source["document_sha256"],
        document_inputs_sha256=source["document_inputs_sha256"],
        annotations=annotations,
        references=references,
        contract=contract,
    )


def record_groundtruth_failure(
    document_dir: Path | str,
    issues: list[str],
) -> None:
    """Persist deterministic validation failure details for a document package."""
    doc_dir = Path(document_dir)
    _write_validation_errors(doc_dir, doc_dir.name, issues)


@contextmanager
def groundtruth_failure_boundary(
    document_dir: Path | str,
    *,
    failure_recorder: Callable[[list[str]], None] | None = None,
) -> Iterator[None]:
    """Record a deterministic failure artifact around a ground-truth run."""
    doc_dir = Path(document_dir)

    def record(issues: list[str]) -> None:
        if failure_recorder is None:
            record_groundtruth_failure(doc_dir, issues)
        else:
            failure_recorder(issues)

    try:
        yield
    except GroundTruthError as exc:
        record(exc.issues)
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        issues = [f"ground truth could not be generated: {exc}"]
        record(issues)
        raise GroundTruthError(doc_dir.name, issues) from exc


def generate_groundtruth_for_document(
    *, document_dir: Path | str, contract_path: Path | str
) -> dict[str, Any]:
    doc_dir = Path(document_dir)
    with groundtruth_failure_boundary(doc_dir):
        source = load_groundtruth_source(doc_dir)
        references = select_used_initial_entities(source)
        annotation_rows = calculate_groundtruth_offsets(
            source=source,
            references=references,
            contract_path=contract_path,
        )
        return validate_and_publish_groundtruth(
            source=source,
            references=references,
            annotation_rows=annotation_rows,
            contract_path=contract_path,
        )


def require_completed_groundtruth(document_dir: Path | str, doc_id: str) -> dict[str, Any]:
    doc_dir = Path(document_dir)
    document_path = doc_dir / f"{doc_id}.txt"
    document_inputs_path = doc_dir / DOCUMENT_INPUTS_FILENAME
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    manifest_path = doc_dir / GROUNDTRUTH_MANIFEST_FILENAME
    if not all(path.is_file() for path in _package_paths(doc_dir, doc_id)):
        raise RuntimeError(f"Ground truth has not completed for {doc_id}")
    manifest = read_json_object(manifest_path)
    if manifest.get("doc_id") != doc_id or manifest.get("status") != "completed":
        raise RuntimeError(f"Ground-truth manifest is not completed for {doc_id}")
    hash_checks = (
        ("groundtruth_sha256", groundtruth_path, "Ground-truth"),
        ("document_sha256", document_path, "Document"),
        ("document_inputs_sha256", document_inputs_path, "Document-inputs"),
    )
    for manifest_key, path, label in hash_checks:
        if manifest.get(manifest_key) != sha256_file(path):
            raise RuntimeError(f"{label} checksum mismatch for {doc_id}")
    return manifest


def discover_document_packages(input_directory: Path | str) -> list[Path]:
    root = Path(input_directory)
    if not root.is_dir():
        raise ValueError(f"Ground-truth input directory does not exist: {root}")
    if _has_groundtruth_inputs(root):
        return [root]
    return sorted(child for child in root.iterdir() if _has_groundtruth_inputs(child))


def _require_source_files(
    doc_id: str,
    document_path: Path,
    document_inputs_path: Path,
) -> None:
    missing = [path.name for path in (document_path, document_inputs_path) if not path.is_file()]
    if missing:
        raise GroundTruthError(doc_id, ["required input file is missing: " + ", ".join(missing)])


def _package_paths(doc_dir: Path, doc_id: str) -> tuple[Path, ...]:
    return (
        doc_dir / f"{doc_id}.txt",
        doc_dir / DOCUMENT_INPUTS_FILENAME,
        doc_dir / GROUNDTRUTH_FILENAME,
        doc_dir / GROUNDTRUTH_MANIFEST_FILENAME,
    )


def _has_groundtruth_inputs(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / f"{path.name}.txt").is_file()
        and (path / DOCUMENT_INPUTS_FILENAME).is_file()
    )


def _publish_groundtruth(
    *,
    doc_dir: Path,
    doc_id: str,
    document_text: str,
    document_sha256: str,
    document_inputs_sha256: str,
    annotations: list[MentionAnnotation],
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
) -> dict[str, Any]:
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    pending_path = doc_dir / f".{GROUNDTRUTH_FILENAME}.pending"
    with pending_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(GROUNDTRUTH_HEADER),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(annotation) for annotation in annotations)
    readback = read_groundtruth_tsv(pending_path)
    validate_mention_annotations(
        doc_id=doc_id,
        document_text=document_text,
        annotations=readback,
        references=references,
        contract=contract,
    )
    os.replace(pending_path, groundtruth_path)
    manifest = _manifest(
        doc_id=doc_id,
        document_sha256=document_sha256,
        document_inputs_sha256=document_inputs_sha256,
        groundtruth_sha256=sha256_file(groundtruth_path),
        annotation_count=len(annotations),
        contract=contract,
    )
    write_json_atomic(doc_dir / GROUNDTRUTH_MANIFEST_FILENAME, manifest)
    error_path = doc_dir / GROUNDTRUTH_ERRORS_FILENAME
    if error_path.exists():
        error_path.unlink()
    return _result_payload(doc_dir, manifest, reused=False)


def _manifest(
    *,
    doc_id: str,
    document_sha256: str,
    document_inputs_sha256: str,
    groundtruth_sha256: str,
    annotation_count: int,
    contract: GroundTruthContract,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "doc_id": doc_id,
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_sha256": contract.sha256,
        "document_sha256": document_sha256,
        "document_inputs_sha256": document_inputs_sha256,
        "groundtruth_sha256": groundtruth_sha256,
        "annotation_count": annotation_count,
        "encoding": "UTF-8",
        "delimiter": "tab",
        "line_endings": "LF",
        "offset_unit": "unicode_code_points",
        "overlap_policy": {
            "prefer_longest_same_label": contract.prefer_longest_same_label,
            "allow_nested_same_label": sorted(contract.nested_same_labels),
        },
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _reuse_existing_groundtruth(
    *,
    doc_dir: Path,
    doc_id: str,
    document_text: str,
    document_sha256: str,
    document_inputs_sha256: str,
    contract: GroundTruthContract,
    references: list[dict[str, Any]],
) -> dict[str, Any] | None:
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    manifest_path = doc_dir / GROUNDTRUTH_MANIFEST_FILENAME
    if not groundtruth_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = read_json_object(manifest_path)
        groundtruth_sha256 = sha256_file(groundtruth_path)
    except (OSError, UnicodeError, ValueError):
        return None
    expected = {
        "status": "completed",
        "doc_id": doc_id,
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_sha256": contract.sha256,
        "document_sha256": document_sha256,
        "document_inputs_sha256": document_inputs_sha256,
        "groundtruth_sha256": groundtruth_sha256,
    }
    mismatches = [
        f"existing groundtruth manifest {key} mismatch"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        return None
    try:
        annotations = read_groundtruth_tsv(groundtruth_path)
        validate_mention_annotations(
            doc_id=doc_id,
            document_text=document_text,
            annotations=annotations,
            references=references,
            contract=contract,
        )
    except (OSError, UnicodeError, ValueError, GroundTruthError):
        return None
    return _result_payload(doc_dir, manifest, reused=True)


def _result_payload(doc_dir: Path, manifest: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "status": "completed",
        "doc_id": manifest["doc_id"],
        "contract_version": manifest["contract_version"],
        "annotation_count": manifest["annotation_count"],
        "document_sha256": manifest["document_sha256"],
        "groundtruth_sha256": manifest["groundtruth_sha256"],
        "groundtruth_path": str(doc_dir / GROUNDTRUTH_FILENAME),
        "manifest_path": str(doc_dir / GROUNDTRUTH_MANIFEST_FILENAME),
        "reused": reused,
    }


def _write_validation_errors(doc_dir: Path, doc_id: str, issues: list[str]) -> None:
    write_json_atomic(
        doc_dir / GROUNDTRUTH_ERRORS_FILENAME,
        {
            "status": "failed",
            "doc_id": doc_id,
            "issues": issues,
            "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def _source_integrity_issues(source: dict[str, Any]) -> list[str]:
    doc_dir = Path(source["document_dir"])
    doc_id = source["doc_id"]
    checks = (
        (
            "document",
            doc_dir / f"{doc_id}.txt",
            source["document_sha256"],
        ),
        (
            "document inputs",
            doc_dir / DOCUMENT_INPUTS_FILENAME,
            source["document_inputs_sha256"],
        ),
    )
    issues = []
    for label, path, expected_sha256 in checks:
        if not path.is_file():
            issues.append(f"{label} file disappeared during ground-truth generation")
        elif sha256_file(path) != expected_sha256:
            issues.append(f"{label} changed during ground-truth generation")
    return issues
