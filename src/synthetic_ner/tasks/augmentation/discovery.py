"""Discover validated source packages for morphological augmentation."""

import csv
from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.augmentation.constants import (
    FLAT_GROUNDTRUTH_HEADER,
    FLAT_GROUNDTRUTH_PREFIX,
)
from src.synthetic_ner.tasks.groundtruth import (
    build_mention_annotations,
    discover_document_packages,
    load_groundtruth_contract,
    read_groundtruth_tsv,
    require_completed_groundtruth,
)
from src.synthetic_ner.tasks.groundtruth.models import GROUNDTRUTH_FILENAME, GroundTruthError
from src.synthetic_ner.types.augmentation import MorphologyError, MorphologySource
from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME


def discover_morphology_sources(
    input_path: Path | str,
    *,
    contract_path: Path | str | None = None,
) -> list[MorphologySource]:
    """Resolve a text file, document package, or parent package directory."""
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".txt":
            raise MorphologyError("Morphology input file must be a .txt document")
        if resolved.name == f"{resolved.parent.name}.txt":
            sources = [_load_package_source(resolved.parent)]
        else:
            sources = [_load_flat_source(resolved, contract_path)]
    elif resolved.is_dir():
        package_sources = [
            _load_package_source(package_dir)
            for package_dir in discover_document_packages(resolved)
        ]
        flat_sources = [
            _load_flat_source(document_path, contract_path)
            for document_path in _flat_document_paths(resolved)
        ]
        sources = sorted(package_sources + flat_sources, key=lambda source: source.doc_id)
    else:
        raise MorphologyError(f"Morphology input path does not exist: {resolved}")

    if not sources:
        raise MorphologyError(
            f"No packaged document or flat text/ground-truth pair was found at {resolved}"
        )
    return sources


def _load_package_source(package_dir: Path) -> MorphologySource:
    doc_id = package_dir.name
    try:
        require_completed_groundtruth(package_dir, doc_id)
        annotations = tuple(read_groundtruth_tsv(package_dir / GROUNDTRUTH_FILENAME))
    except (OSError, RuntimeError, ValueError) as exc:
        raise MorphologyError(
            f"Document package {doc_id} must have complete ground truth: {exc}"
        ) from exc
    references = _references_from_annotations(annotations)
    return MorphologySource(
        doc_id=doc_id,
        package_dir=package_dir,
        document_path=package_dir / f"{doc_id}.txt",
        document_inputs_path=package_dir / DOCUMENT_INPUTS_FILENAME,
        groundtruth_path=package_dir / GROUNDTRUTH_FILENAME,
        text=(package_dir / f"{doc_id}.txt").read_text(encoding="utf-8"),
        annotations=annotations,
        entity_references=references,
    )


def _load_flat_source(
    document_path: Path,
    contract_path: Path | str | None,
) -> MorphologySource:
    doc_id = document_path.stem
    groundtruth_path = document_path.with_name(f"{FLAT_GROUNDTRUTH_PREFIX}{doc_id}.tsv")
    if not groundtruth_path.is_file():
        raise MorphologyError(
            f"Flat morphology input {document_path.name} requires companion {groundtruth_path.name}"
        )
    if contract_path is None:
        raise MorphologyError("Ground-truth contract path is required for flat morphology input")
    try:
        text = document_path.read_text(encoding="utf-8")
        references = _read_flat_references(groundtruth_path, doc_id)
        annotations = tuple(
            build_mention_annotations(
                doc_id=doc_id,
                document_text=text,
                references=list(references),
                contract=load_groundtruth_contract(contract_path),
            )
        )
    except (OSError, UnicodeError, ValueError, GroundTruthError) as exc:
        raise MorphologyError(f"Invalid flat morphology source {doc_id}: {exc}") from exc
    return MorphologySource(
        doc_id=doc_id,
        package_dir=document_path.parent,
        document_path=document_path,
        document_inputs_path=None,
        groundtruth_path=groundtruth_path,
        text=text,
        annotations=annotations,
        entity_references=references,
    )


def _flat_document_paths(directory: Path) -> list[Path]:
    return sorted(
        document_path
        for document_path in directory.glob("*.txt")
        if document_path.with_name(f"{FLAT_GROUNDTRUTH_PREFIX}{document_path.stem}.tsv").is_file()
    )


def _read_flat_references(
    groundtruth_path: Path,
    doc_id: str,
) -> tuple[dict[str, Any], ...]:
    references: dict[tuple[str, str], dict[str, Any]] = {}
    with groundtruth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FLAT_GROUNDTRUTH_HEADER:
            raise ValueError("flat ground-truth TSV header is invalid")
        for row_number, row in enumerate(reader, start=2):
            if row["doc_id"] != doc_id:
                raise ValueError(f"row {row_number} doc_id does not match {doc_id}")
            entity_text = row["entity_text"].strip()
            label = row["label"].strip()
            if not entity_text or not label:
                raise ValueError(f"row {row_number} has an empty entity or label")
            references[(entity_text, label)] = {
                "entity_text": entity_text,
                "label": label,
                "source_fields": [f"flat_groundtruth[{row_number}]"],
                "notes": [row["notes"].strip()] if row["notes"].strip() else [],
            }
    if not references:
        raise ValueError("flat ground-truth TSV contains no entities")
    return tuple(references.values())


def _references_from_annotations(
    annotations,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"entity_text": entity_text, "label": label}
        for entity_text, label in sorted({(row.entity_text, row.label) for row in annotations})
    )
