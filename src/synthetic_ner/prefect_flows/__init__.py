"""Prefect flows for synthetic NER document and ground-truth generation."""

from src.synthetic_ner.prefect_flows.augmentation import (
    generate_document_morphological_variations,
    generate_morphological_variations,
)
from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.groundtruth import (
    generate_document_groundtruth,
    generate_groundtruth_directory,
)

__all__ = [
    "generate_dataset",
    "generate_document_morphological_variations",
    "generate_morphological_variations",
    "generate_document_groundtruth",
    "generate_groundtruth_directory",
]
