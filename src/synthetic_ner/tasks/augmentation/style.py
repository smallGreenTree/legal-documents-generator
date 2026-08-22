"""Validation and naming for user-requested style augmentation."""

import math
import re
import unicodedata

from src.synthetic_ner.tasks.augmentation.constants import (
    MAX_CUSTOM_STYLE_CHARS,
    MAX_STYLE_SLUG_CHARS,
    MAX_STYLE_TEMPERATURE,
    MIN_STYLE_TEMPERATURE,
)
from src.synthetic_ner.types.augmentation import MorphologyError


def normalize_style(style: str | None) -> str:
    normalized = " ".join((style or "").split())
    if not normalized:
        raise MorphologyError("Custom style must be provided")
    if len(normalized) > MAX_CUSTOM_STYLE_CHARS:
        raise MorphologyError(f"Custom style must not exceed {MAX_CUSTOM_STYLE_CHARS} characters")
    return normalized


def style_slug(style: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_style(style))
    ascii_style = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_style).strip("-")
    if not slug:
        slug = "custom"
    return slug[:MAX_STYLE_SLUG_CHARS].rstrip("-")


def normalize_style_temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MorphologyError("Style temperature must be a number")
    temperature = float(value)
    if not MIN_STYLE_TEMPERATURE <= temperature <= MAX_STYLE_TEMPERATURE:
        raise MorphologyError(
            f"Style temperature must be between {MIN_STYLE_TEMPERATURE:.1f} "
            f"and {MAX_STYLE_TEMPERATURE:.1f}"
        )
    rounded = round(temperature, 1)
    if not math.isclose(temperature, rounded, abs_tol=1e-9):
        raise MorphologyError("Style temperature must use increments of 0.1")
    return rounded


def style_temperature_slug(value: float) -> str:
    temperature = normalize_style_temperature(value)
    return f"{temperature:.1f}".replace(".", "p")


def style_reformatting_instruction(enabled: bool) -> str:
    if enabled:
        return (
            "Reformatting is enabled. Within each short source paragraph, you may change "
            "line breaks, sentence boundaries and indentation to make the requested style "
            "unmistakable. Preserve source paragraph boundaries, headings and structural "
            "labels exactly."
        )
    return (
        "Reformatting is disabled. Preserve the source paragraph boundaries, line-break "
        "pattern, headings and structural labels while restyling the prose."
    )


def custom_style_instruction(style: str | None) -> str:
    normalized = normalize_style(style)
    return (
        "Rewrite every eligible prose sentence in this requested style: "
        f'"{normalized}". Preserve every factual proposition and protected value. '
        "Keep headings and structural labels unchanged. A sentence that cannot be "
        "restyled safely may remain unchanged."
    )
