"""Prefect flows for synthetic NER generation and quality."""

from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.groundtruth import (
    generate_document_groundtruth,
    generate_groundtruth_directory,
)
from src.synthetic_ner.prefect_flows.quality import score_existing_document

__all__ = [
    "generate_dataset",
    "generate_document_groundtruth",
    "generate_groundtruth_directory",
    "score_existing_document",
]
