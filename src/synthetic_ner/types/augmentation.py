"""Types shared by morphological augmentation tasks and flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from prefect.input import RunInput
from pydantic import Field

from src.synthetic_ner.tasks.groundtruth.models import MentionAnnotation


class MorphologyError(ValueError):
    """Raised when a morphology variant cannot be created safely."""


class MorphologyTransformation(StrEnum):
    ACTIVE_TO_PASSIVE = "active-to-passive"
    VERBAL_TO_NOMINAL = "verbal-to-nominal"
    POSSESSIVE_REFRAME = "possessive-reframe"
    INTENTIONAL_TYPOS = "intentional-typos"
    RANDOM_LAYOUT = "random-layout"
    CUSTOM_STYLE = "custom-style"


@dataclass(frozen=True, slots=True)
class MorphologySource:
    doc_id: str
    package_dir: Path
    document_path: Path
    document_inputs_path: Path | None
    groundtruth_path: Path
    text: str
    annotations: tuple[MentionAnnotation, ...]
    entity_references: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ProtectedMention:
    token: str
    text: str
    source_start: int
    source_end: int
    annotations: tuple[MentionAnnotation, ...]

    @property
    def annotation(self) -> MentionAnnotation:
        """Return the first annotation for single-span callers."""
        return self.annotations[0]


@dataclass(frozen=True, slots=True)
class ProtectedLiteral:
    token: str
    text: str


@dataclass(frozen=True, slots=True)
class ProtectedDocument:
    text: str
    mentions: tuple[ProtectedMention, ...]
    literals: tuple[ProtectedLiteral, ...]


@dataclass(frozen=True, slots=True)
class MorphologyVariant:
    doc_id: str
    transformation: MorphologyTransformation
    text: str
    annotations: tuple[MentionAnnotation, ...]
    change_ratio: float
    style: str | None = None
    style_temperature: float | None = None
    reformat_with_style: bool = False
    style_fallback_chunk_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class MorphologyPromptsConfig:
    system: str
    user: str
    retry: str
    style_system: str
    style_user: str
    style_retry: str


@dataclass(frozen=True, slots=True)
class MorphologyWorkflowConfig:
    temperature: float
    style_temperature: float
    style_retry_temperature: float
    style_maximum_change_ratio: float
    style_max_chunk_chars: int
    style_max_protected_tokens: int
    style_max_sentences_per_chunk: int
    max_output_tokens: int
    max_chunk_chars: int
    max_chunk_attempts: int
    minimum_change_ratio: float
    maximum_change_ratio: float
    prompts: MorphologyPromptsConfig
    deterministic_minimum_change_ratio: float = 0.0001
    typo_rate: float = 0.005
    max_typos: int = 10
    layout_widths: tuple[int, ...] = (72, 88, 100)


class MorphologyReviewInput(RunInput):
    """Prefect form rendered as path/style fields and transformation checkboxes."""

    input_path: str = Field(
        default="",
        title="Input file or folder",
        description="Local .txt file, document package, or folder of document packages.",
        json_schema_extra={"position": 0},
    )
    style: str = Field(
        default="",
        title="CUSTOM STYLE REQUEST",
        description=(
            "Primary creative instruction for an additional style variant. Be explicit, "
            "for example: gritty rap verse with punchy cadence, internal rhyme, and "
            "end-rhyming couplets."
        ),
        max_length=200,
        json_schema_extra={"position": 1},
    )
    style_temperature: float = Field(
        default=0.8,
        title="Style temperature (0.0-1.5)",
        description=(
            "Creativity for the custom-style rewrite only: lower is steadier; "
            "higher is more adventurous."
        ),
        ge=0.0,
        le=1.5,
        multiple_of=0.1,
        json_schema_extra={"position": 2},
    )
    reformat_with_style: bool = Field(
        default=True,
        title="Reformat layout with the style",
        description=(
            "Allow the style variant to change line breaks, sentence boundaries, and "
            "indentation within each paragraph while preserving paragraph boundaries, "
            "headings, and facts."
        ),
        json_schema_extra={"position": 3},
    )
    active_to_passive: bool = Field(default=True, json_schema_extra={"position": 4})
    verbal_to_nominal: bool = Field(default=True, json_schema_extra={"position": 5})
    possessive_reframe: bool = Field(default=True, json_schema_extra={"position": 6})
    intentional_typos: bool = Field(default=False, json_schema_extra={"position": 7})
    random_layout: bool = Field(default=False, json_schema_extra={"position": 8})
