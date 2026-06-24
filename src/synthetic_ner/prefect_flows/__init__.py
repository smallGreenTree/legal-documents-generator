"""Prefect flows for synthetic NER generation, quality, and evaluation."""

from src.synthetic_ner.prefect_flows.evaluation import evaluate_existing_document
from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.quality import score_existing_document

__all__ = ["evaluate_existing_document", "generate_dataset", "score_existing_document"]
