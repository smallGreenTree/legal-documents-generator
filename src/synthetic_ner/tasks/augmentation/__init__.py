"""Controlled morphological augmentation for validated NER documents."""

from src.synthetic_ner.tasks.augmentation.constants import VARIANT_VERSION
from src.synthetic_ner.tasks.augmentation.discovery import discover_morphology_sources
from src.synthetic_ner.tasks.augmentation.protection import (
    protect_document_text,
    reconstruct_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.style import style_slug, style_temperature_slug
from src.synthetic_ner.types.augmentation import MorphologyTransformation


def build_variant_id(
    source_doc_id: str,
    transformation: MorphologyTransformation,
    *,
    style: str | None = None,
    style_temperature: float | None = None,
    reformat_with_style: bool = False,
) -> str:
    """Build a stable identifier that names the applied transformation."""
    if transformation is MorphologyTransformation.CUSTOM_STYLE:
        if style_temperature is None:
            raise ValueError("style_temperature is required for a custom-style variant")
        reformat_suffix = "__reformatted" if reformat_with_style else ""
        return (
            f"{source_doc_id}__style-{style_slug(style)}"
            f"__t{style_temperature_slug(style_temperature)}{reformat_suffix}"
            f"__v{VARIANT_VERSION:02d}"
        )
    return f"{source_doc_id}__morph-{transformation.value}__v{VARIANT_VERSION:02d}"


__all__ = [
    "build_variant_id",
    "discover_morphology_sources",
    "protect_document_text",
    "reconstruct_morphology_variant",
]
