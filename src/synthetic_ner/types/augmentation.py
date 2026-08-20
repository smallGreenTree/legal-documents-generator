"""Types shared by morphological augmentation tasks and flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from prefect.input import RunInput

from src.synthetic_ner.tasks.groundtruth.models import MentionAnnotation


class MorphologyError(ValueError):
    """Raised when a morphology variant cannot be created safely."""


class MorphologyTransformation(StrEnum):
    ACTIVE_TO_PASSIVE = "active-to-passive"
    VERBAL_TO_NOMINAL = "verbal-to-nominal"
    POSSESSIVE_REFRAME = "possessive-reframe"
    INTENTIONAL_TYPOS = "intentional-typos"
    RANDOM_LAYOUT = "random-layout"


@dataclass(frozen=True, slots=True)
class MorphologySource:
    doc_id: str
    package_dir: Path
    document_path: Path
    document_inputs_path: Path
    groundtruth_path: Path
    text: str
    annotations: tuple[MentionAnnotation, ...]


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


@dataclass(frozen=True, slots=True)
class MorphologyPromptsConfig:
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class MorphologyWorkflowConfig:
    temperature: float
    max_output_tokens: int
    max_chunk_chars: int
    minimum_change_ratio: float
    maximum_change_ratio: float
    prompts: MorphologyPromptsConfig
    deterministic_minimum_change_ratio: float = 0.0001
    typo_rate: float = 0.005
    max_typos: int = 10
    layout_widths: tuple[int, ...] = (72, 88, 100)


class MorphologyReviewInput(RunInput):
    """Prefect form rendered as one path field and five checkboxes."""

    input_path: str = ""
    active_to_passive: bool = True
    verbal_to_nominal: bool = True
    possessive_reframe: bool = True
    intentional_typos: bool = False
    random_layout: bool = False
