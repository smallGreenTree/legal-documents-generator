"""Construct the runtime components used by document generation."""

from __future__ import annotations

from src.synthetic_ner.model_providers.factory import build_model_client, describe_stage_route
from src.synthetic_ner.tasks.document_generation.context.memory import CaseMemoryManager
from src.synthetic_ner.tasks.document_generation.observability.tracer import TraceStore
from src.synthetic_ner.tasks.document_generation.stages.critic import SectionCritic
from src.synthetic_ner.tasks.document_generation.stages.polisher import SectionPolisher
from src.synthetic_ner.tasks.document_generation.stages.writer import SectionWriter
from src.synthetic_ner.types.document_generation import GenerationComponents
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext


def build_generation_components(
    *,
    context: RuntimeContext,
    document: DocumentInputs,
    doc_id: str,
    workflow_run_id: str | None,
    prefect_flow_run_id: str | None,
) -> GenerationComponents:
    trace_store = TraceStore(
        context.mlflow_cfg,
        run_metadata={
            "doc_id": doc_id,
            "workflow_run_id": workflow_run_id or doc_id,
            "prefect_flow_run_id": prefect_flow_run_id,
        },
    )
    memory_manager = CaseMemoryManager(
        context.memory_dir,
        summary_chars=context.workflow_cfg.memory_summary_chars,
    )
    memory_path = memory_manager.create_initial_memory(
        doc_id=doc_id,
        doc_type=context.doc_type,
        fraud_type=context.fraud_type,
        document=document,
        section_order=list(context.section_word_targets),
    )

    writer_client = build_model_client(
        stage="writer",
        routing=context.model_routing_cfg,
        tracer=trace_store,
    )
    polisher_client = None
    if context.workflow_cfg.polisher.active:
        polisher_client = build_model_client(
            stage="polisher",
            routing=context.model_routing_cfg,
            tracer=trace_store,
        )
    critic_client = None
    if context.workflow_cfg.critic.active:
        critic_client = build_model_client(
            stage="critic",
            routing=context.model_routing_cfg,
            tracer=trace_store,
        )

    resolved_prompts = trace_store.resolve_workflow_prompts(context.workflow_cfg.prompts)
    prompts = resolved_prompts.prompts
    print(f"  Prompts : {resolved_prompts.sync_summary}")
    print(
        "  Models  : "
        + ", ".join(
            describe_stage_route(stage=stage, routing=context.model_routing_cfg)
            for stage in active_model_stages(context)
        )
    )

    writer = SectionWriter(
        client=writer_client,
        prompts=prompts,
        chunk_words=context.workflow_cfg.writer.chunk_words,
        context_tail_chars=context.workflow_cfg.writer.context_tail_chars,
        writer_temperature=context.workflow_cfg.writer.temperature,
        max_output_tokens=context.workflow_cfg.writer.max_output_tokens,
        min_output_tokens=context.workflow_cfg.writer.min_output_tokens,
        output_token_multiplier=context.workflow_cfg.writer.output_token_multiplier,
        min_completion_ratio=context.workflow_cfg.writer.min_completion_ratio,
        prompt_clients=resolved_prompts.prompt_clients,
        partial_output_dir=context.output_dir / "_partial",
    )
    polisher = None
    if polisher_client is not None:
        polisher = SectionPolisher(
            client=polisher_client,
            prompts=prompts,
            temperature=context.workflow_cfg.polisher.temperature,
            max_output_tokens=context.workflow_cfg.polisher.max_output_tokens,
            prompt_clients=resolved_prompts.prompt_clients,
            partial_output_dir=context.output_dir / "_partial",
        )
    critic = None
    if critic_client is not None:
        critic = SectionCritic(
            client=critic_client,
            prompts=prompts,
            critic_temperature=context.workflow_cfg.critic.temperature,
            acceptance_threshold=context.workflow_cfg.critic.acceptance_threshold,
            max_output_tokens=context.workflow_cfg.critic.max_output_tokens,
            memory_char_limit=context.workflow_cfg.critic.memory_char_limit,
            section_text_char_limit=context.workflow_cfg.critic.section_text_char_limit,
            rubrics=context.workflow_cfg.critic.rubrics,
            prompt_clients=resolved_prompts.prompt_clients,
        )

    return GenerationComponents(
        trace_store=trace_store,
        memory_manager=memory_manager,
        memory_path=memory_path,
        writer=writer,
        polisher=polisher,
        critic=critic,
    )


def active_model_stages(context: RuntimeContext) -> tuple[str, ...]:
    stages = ["writer"]
    if context.workflow_cfg.critic.active:
        stages.append("critic")
    if context.workflow_cfg.polisher.active:
        stages.append("polisher")
    return tuple(stages)
