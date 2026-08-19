"""Public ground-truth API.

Implementation is split by responsibility so callers do not depend on internal
module layout.
"""

from src.synthetic_ner.tasks.groundtruth.annotations import (
    build_mention_annotations,
    read_groundtruth_tsv,
    validate_mention_annotations,
)
from src.synthetic_ner.tasks.groundtruth.contract import load_groundtruth_contract
from src.synthetic_ner.tasks.groundtruth.models import (
    GROUNDTRUTH_ERRORS_FILENAME,
    GROUNDTRUTH_FILENAME,
    GROUNDTRUTH_HEADER,
    GROUNDTRUTH_MANIFEST_FILENAME,
    GroundTruthContract,
    GroundTruthError,
    MentionAnnotation,
)
from src.synthetic_ner.tasks.groundtruth.publication import (
    calculate_groundtruth_offsets,
    discover_document_packages,
    generate_groundtruth_for_document,
    groundtruth_failure_boundary,
    load_groundtruth_source,
    record_groundtruth_failure,
    require_completed_groundtruth,
    select_used_initial_entities,
    validate_and_publish_groundtruth,
)
from src.synthetic_ner.tasks.groundtruth.references import build_entity_references

__all__ = [
    "GROUNDTRUTH_ERRORS_FILENAME",
    "GROUNDTRUTH_FILENAME",
    "GROUNDTRUTH_HEADER",
    "GROUNDTRUTH_MANIFEST_FILENAME",
    "GroundTruthContract",
    "GroundTruthError",
    "MentionAnnotation",
    "build_mention_annotations",
    "build_entity_references",
    "calculate_groundtruth_offsets",
    "discover_document_packages",
    "generate_groundtruth_for_document",
    "groundtruth_failure_boundary",
    "load_groundtruth_contract",
    "load_groundtruth_source",
    "read_groundtruth_tsv",
    "record_groundtruth_failure",
    "require_completed_groundtruth",
    "select_used_initial_entities",
    "validate_and_publish_groundtruth",
    "validate_mention_annotations",
]
