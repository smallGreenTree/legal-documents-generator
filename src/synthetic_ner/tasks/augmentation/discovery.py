"""Discover validated source packages for morphological augmentation."""

from pathlib import Path

from src.synthetic_ner.tasks.groundtruth import (
    discover_document_packages,
    read_groundtruth_tsv,
    require_completed_groundtruth,
)
from src.synthetic_ner.tasks.groundtruth.models import GROUNDTRUTH_FILENAME
from src.synthetic_ner.types.augmentation import MorphologyError, MorphologySource
from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME


def discover_morphology_sources(input_path: Path | str) -> list[MorphologySource]:
    """Resolve a text file, document package, or parent package directory."""
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".txt":
            raise MorphologyError("Morphology input file must be a .txt document")
        candidates = [resolved.parent]
        if resolved.name != f"{resolved.parent.name}.txt":
            raise MorphologyError(
                "Selected .txt file must belong to a document package named after its doc_id"
            )
    elif resolved.is_dir():
        candidates = discover_document_packages(resolved)
    else:
        raise MorphologyError(f"Morphology input path does not exist: {resolved}")

    if not candidates:
        raise MorphologyError(
            f"No document package with complete ground truth was found at {resolved}"
        )
    return [_load_source(package_dir) for package_dir in candidates]


def _load_source(package_dir: Path) -> MorphologySource:
    doc_id = package_dir.name
    try:
        require_completed_groundtruth(package_dir, doc_id)
        annotations = tuple(read_groundtruth_tsv(package_dir / GROUNDTRUTH_FILENAME))
    except (OSError, RuntimeError, ValueError) as exc:
        raise MorphologyError(
            f"Document package {doc_id} must have complete ground truth: {exc}"
        ) from exc
    return MorphologySource(
        doc_id=doc_id,
        package_dir=package_dir,
        document_path=package_dir / f"{doc_id}.txt",
        document_inputs_path=package_dir / DOCUMENT_INPUTS_FILENAME,
        groundtruth_path=package_dir / GROUNDTRUTH_FILENAME,
        text=(package_dir / f"{doc_id}.txt").read_text(encoding="utf-8"),
        annotations=annotations,
    )
