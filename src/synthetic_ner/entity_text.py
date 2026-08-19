"""Shared entity-text patterns and normalization helpers."""

from __future__ import annotations

import re
from typing import Any

SURFACE_EDGE_PUNCTUATION = ".,;:()[]{}"

AMOUNT_RE = re.compile(
    r"(?:£|€|\b(?:GBP|EUR)\s*)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|m|thousand|k))?",
    re.IGNORECASE,
)


def strip_surface_punctuation(value: Any) -> str:
    """Strip punctuation allowed around an entity surface without changing its interior."""
    if not isinstance(value, str):
        return ""
    return value.strip().strip(SURFACE_EDGE_PUNCTUATION)


def normalize_phrase(value: Any) -> str:
    """Normalize phrase whitespace and surrounding punctuation."""
    return strip_surface_punctuation(" ".join(str(value).strip().split()))
