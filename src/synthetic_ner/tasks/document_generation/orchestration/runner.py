"""Run one fully configured document-generation graph."""

from __future__ import annotations

from src.synthetic_ner.tasks.document_generation.orchestration.components import (
    build_generation_components,
)
from src.synthetic_ner.tasks.document_generation.orchestration.graph import build_document_graph
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext


def run_document_graph(
    *,
    context: RuntimeContext,
    document: DocumentInputs,
    doc_id: str,
    workflow_run_id: str | None = None,
    prefect_flow_run_id: str | None = None,
) -> None:
    if not context.workflow_cfg.writer.active:
        raise ValueError("workflow.writer.active must be true for document generation.")

    components = build_generation_components(
        context=context,
        document=document,
        doc_id=doc_id,
        workflow_run_id=workflow_run_id,
        prefect_flow_run_id=prefect_flow_run_id,
    )
    trace_info = components.trace_store.start_document_run(
        doc_id=doc_id,
        input_payload={
            "doc_id": doc_id,
            "doc_type": context.doc_type,
            "fraud_type": context.fraud_type,
            "section_order": list(context.section_word_targets),
        },
        metadata={
            "doc_id": doc_id,
            "doc_type": context.doc_type,
            "fraud_type": context.fraud_type,
            "case_number": document.metadata["case_number"],
            "writer_active": context.workflow_cfg.writer.active,
            "polisher_active": context.workflow_cfg.polisher.active,
            "critic_active": context.workflow_cfg.critic.active,
        },
    )
    if trace_info.trace_url:
        print(f"  Trace   : {trace_info.trace_url}")

    final_state = None
    try:
        seed_memory_text = components.memory_manager.read_memory(components.memory_path)
        graph = build_document_graph(
            context=context,
            document=document,
            doc_id=doc_id,
            components=components,
        )
        final_state = graph.invoke(
            {
                "doc_id": doc_id,
                "memory_path": components.memory_path,
                "memory_text": seed_memory_text,
                "section_order": list(context.section_word_targets),
                "section_outputs": {},
                "section_contracts": {},
                "section_reviews": {},
            }
        )
    finally:
        components.trace_store.end_document_run(
            output_payload={
                "doc_id": doc_id,
                "rendered": bool(final_state and final_state.get("final_text")),
            }
        )
