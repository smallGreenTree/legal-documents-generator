"""Shared ground-truth data types and file names."""

from dataclasses import dataclass
from pathlib import Path

CONTRACT_NAME = "ner_groundtruth_mentions"
CONTRACT_VERSION = "1.0.0"
GROUNDTRUTH_HEADER = (
    "annotation_id",
    "doc_id",
    "entity_text",
    "label",
    "start_char",
    "end_char",
)
GROUNDTRUTH_FILENAME = "groundtruth.tsv"
GROUNDTRUTH_MANIFEST_FILENAME = "groundtruth_manifest.json"
GROUNDTRUTH_ERRORS_FILENAME = "groundtruth_validation_errors.json"


class GroundTruthError(ValueError):
    """Raised when ground truth cannot be safely generated or published."""

    def __init__(self, doc_id: str, issues: list[str]) -> None:
        self.doc_id = doc_id
        self.issues = issues
        super().__init__(f"Ground-truth validation failed for {doc_id}: {'; '.join(issues)}")


@dataclass(frozen=True, slots=True)
class GroundTruthContract:
    name: str
    version: str
    columns: tuple[str, ...]
    allowed_labels: frozenset[str]
    nested_same_labels: frozenset[str]
    prefer_longest_same_label: bool
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MentionAnnotation:
    annotation_id: str
    doc_id: str
    entity_text: str
    label: str
    start_char: int
    end_char: int
