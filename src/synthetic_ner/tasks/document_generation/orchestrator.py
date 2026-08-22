"""Stable CLI and Prefect entry points for document generation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from src.synthetic_ner.case_generation.identifiers import make_doc_id, next_counter
from src.synthetic_ner.document.engine import (
    build_runtime_context,
    build_size_label,
    resolve_document_inputs,
)
from src.synthetic_ner.tasks.document_generation.orchestration.graph import (
    DocumentWorkflow,
    build_document_graph,
)
from src.synthetic_ner.tasks.document_generation.orchestration.runner import run_document_graph

__all__ = [
    "DocumentWorkflow",
    "build_document_graph",
    "run_document_graph",
    "run_langgraph_workflow",
]


def run_langgraph_workflow(args: Namespace, project_root: Path) -> None:
    context = build_runtime_context(args, project_root)
    size_label = build_size_label(context)

    for document_index in range(context.documents):
        print(
            f"\n[{document_index + 1}/{context.documents}] Generating "
            f"{context.doc_type} / {context.fraud_type} / {size_label} via langgraph …"
        )

        document = resolve_document_inputs(context)
        counter = next_counter(context.output_dir, context.doc_type, context.fraud_type)
        doc_id = make_doc_id(context.doc_type, context.fraud_type, counter)
        run_document_graph(
            context=context,
            document=document,
            doc_id=doc_id,
        )

    print("\nDone.")
