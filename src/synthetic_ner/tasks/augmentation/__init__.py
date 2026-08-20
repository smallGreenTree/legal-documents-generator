"""Controlled morphological augmentation for validated NER documents."""

from src.synthetic_ner.tasks.augmentation.constants import VARIANT_VERSION
from src.synthetic_ner.tasks.augmentation.discovery import discover_morphology_sources
from src.synthetic_ner.tasks.augmentation.protection import (
    protect_document_text,
    reconstruct_morphology_variant,
)
from src.synthetic_ner.types.augmentation import MorphologyTransformation


def build_variant_id(
    source_doc_id: str,
    transformation: MorphologyTransformation,
) -> str:
    """Build a stable identifier that names the applied transformation."""
    return f"{source_doc_id}__morph-{transformation.value}__v{VARIANT_VERSION:02d}"


__all__ = [
    "build_variant_id",
    "discover_morphology_sources",
    "protect_document_text",
    "reconstruct_morphology_variant",
]
