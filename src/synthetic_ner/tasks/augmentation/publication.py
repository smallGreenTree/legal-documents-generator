"""Atomic publication of morphology variants as complete NER packages."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.synthetic_ner.document.inputs import ENTITY_REFERENCES_FIELD
from src.synthetic_ner.tasks.augmentation.constants import (
    AUGMENTATION_DIRECTORY_NAME,
    AUGMENTATION_MANIFEST_FILENAME,
    MORPHOLOGY_BATCH_REPORT_FILENAME,
    TRANSFORMATION_INSTRUCTIONS,
)
from src.synthetic_ner.tasks.groundtruth import (
    generate_groundtruth_for_document,
    read_groundtruth_tsv,
    require_completed_groundtruth,
)
from src.synthetic_ner.tasks.groundtruth.files import (
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from src.synthetic_ner.tasks.groundtruth.models import GROUNDTRUTH_FILENAME
from src.synthetic_ner.types.augmentation import (
    MorphologyError,
    MorphologySource,
    MorphologyTransformation,
    MorphologyVariant,
)
from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME


def existing_variant_result(
    source: MorphologySource,
    variant_doc_id: str,
    transformation: MorphologyTransformation,
    *,
    style: str | None = None,
    style_temperature: float | None = None,
    reformat_with_style: bool = False,
) -> dict[str, Any] | None:
    target = _variant_directory(source, variant_doc_id)
    if not target.exists():
        return None
    manifest_path = target / AUGMENTATION_MANIFEST_FILENAME
    try:
        manifest = read_json_object(manifest_path)
        require_completed_groundtruth(target, variant_doc_id)
    except (OSError, RuntimeError, ValueError):
        raise MorphologyError(
            f"Existing morphology output is incomplete and will not be overwritten: {target}"
        ) from None
    expected = {
        "status": "completed",
        "source_doc_id": source.doc_id,
        "variant_doc_id": variant_doc_id,
        "transformation": transformation.value,
        "source_document_sha256": sha256_file(source.document_path),
        "source_groundtruth_sha256": sha256_file(source.groundtruth_path),
    }
    if style is not None:
        expected["style"] = style
        expected["style_temperature"] = style_temperature
        expected["reformat_with_style"] = reformat_with_style
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise MorphologyError(f"Existing morphology output does not match this request: {target}")
    return _result(target, manifest, reused=True)


def publish_morphology_variant(
    *,
    source: MorphologySource,
    variant: MorphologyVariant,
    contract_path: Path | str,
) -> dict[str, Any]:
    """Publish text, regenerated ground truth, inputs, and an audit manifest."""
    output_root = source.package_dir / AUGMENTATION_DIRECTORY_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / variant.doc_id
    if target.exists():
        raise MorphologyError(
            f"Morphology output already exists and will not be overwritten: {target}"
        )

    with tempfile.TemporaryDirectory(prefix=".pending-", dir=output_root) as pending_root:
        pending_package = Path(pending_root) / variant.doc_id
        pending_package.mkdir()
        document_path = pending_package / f"{variant.doc_id}.txt"
        document_path.write_text(variant.text, encoding="utf-8")
        _write_variant_inputs(source, pending_package / DOCUMENT_INPUTS_FILENAME)
        groundtruth = generate_groundtruth_for_document(
            document_dir=pending_package,
            contract_path=contract_path,
        )
        published_annotations = read_groundtruth_tsv(pending_package / GROUNDTRUTH_FILENAME)
        if _annotation_values(published_annotations) != _annotation_values(variant.annotations):
            raise MorphologyError(
                "Regenerated ground truth does not match the protected morphology annotations"
            )
        manifest = {
            "status": "completed",
            "source_doc_id": source.doc_id,
            "variant_doc_id": variant.doc_id,
            "transformation": variant.transformation.value,
            "transformation_explanation": _transformation_explanation(variant),
            "change_ratio": round(variant.change_ratio, 6),
            "source_document_sha256": sha256_file(source.document_path),
            "source_groundtruth_sha256": sha256_file(source.groundtruth_path),
            "variant_document_sha256": sha256_file(document_path),
            "variant_groundtruth_sha256": groundtruth["groundtruth_sha256"],
            "annotation_count": len(published_annotations),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if variant.style is not None:
            manifest["style"] = variant.style
            manifest["style_temperature"] = variant.style_temperature
            manifest["reformat_with_style"] = variant.reformat_with_style
            manifest["style_fallback_chunk_indices"] = list(variant.style_fallback_chunk_indices)
            manifest["style_fallback_chunk_count"] = len(variant.style_fallback_chunk_indices)
        write_json_atomic(pending_package / AUGMENTATION_MANIFEST_FILENAME, manifest)
        os.replace(pending_package, target)
    return _result(target, manifest, reused=False)


def publish_batch_report(
    *,
    input_path: Path | str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = Path(input_path).expanduser().resolve()
    container = selected.parent if selected.is_file() else selected
    if (container / f"{container.name}.txt").is_file():
        container = container / AUGMENTATION_DIRECTORY_NAME
    report = {
        "status": (
            "completed"
            if all(result.get("status") == "completed" for result in results)
            else "failed"
        ),
        "input_path": str(selected),
        "variants_requested": len(results),
        "variants_completed": sum(result.get("status") == "completed" for result in results),
        "variants_failed": sum(result.get("status") != "completed" for result in results),
        "variants_reused": sum(bool(result.get("reused")) for result in results),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
    }
    report_path = container / MORPHOLOGY_BATCH_REPORT_FILENAME
    write_json_atomic(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _variant_directory(source: MorphologySource, variant_doc_id: str) -> Path:
    return source.package_dir / AUGMENTATION_DIRECTORY_NAME / variant_doc_id


def _write_variant_inputs(source: MorphologySource, target: Path) -> None:
    if source.document_inputs_path is not None:
        shutil.copy2(source.document_inputs_path, target)
        return
    write_json_atomic(
        target,
        {
            "defendants": [],
            "collateral": [],
            "charged_orgs": [],
            "associated_orgs": [],
            "metadata": {},
            "amounts": {},
            "counts_list": [],
            ENTITY_REFERENCES_FIELD: list(source.entity_references),
        },
    )


def _annotation_values(rows) -> list[tuple[str, str, int, int]]:
    return sorted((row.entity_text, row.label, row.start_char, row.end_char) for row in rows)


def _transformation_explanation(variant: MorphologyVariant) -> str:
    if variant.transformation is MorphologyTransformation.CUSTOM_STYLE:
        return f"Rewrite eligible prose in the user-requested style: {variant.style}"
    return TRANSFORMATION_INSTRUCTIONS[variant.transformation]


def _result(target: Path, manifest: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "status": "completed",
        "source_doc_id": manifest["source_doc_id"],
        "variant_doc_id": manifest["variant_doc_id"],
        "transformation": manifest["transformation"],
        "style": manifest.get("style"),
        "style_temperature": manifest.get("style_temperature"),
        "reformat_with_style": manifest.get("reformat_with_style", False),
        "style_fallback_chunk_count": manifest.get("style_fallback_chunk_count", 0),
        "variant_directory": str(target),
        "document_path": str(target / f"{manifest['variant_doc_id']}.txt"),
        "groundtruth_path": str(target / GROUNDTRUTH_FILENAME),
        "manifest_path": str(target / AUGMENTATION_MANIFEST_FILENAME),
        "reused": reused,
    }
