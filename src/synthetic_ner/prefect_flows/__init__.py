"""Prefect flows for synthetic NER document generation and quality scoring."""

from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.quality import score_existing_document

__all__ = ["generate_dataset", "score_existing_document"]
