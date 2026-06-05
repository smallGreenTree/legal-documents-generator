"""LangGraph orchestration for planner/writer/critic workflows."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.synthetic_ner.engine import (
    build_runtime_context,
    build_size_label,
    collect_section_output_problems,
    render_document_text,
    resolve_document_inputs,
    resolve_schema_for_document,
    save_document_artifacts,
)
from src.synthetic_ner.models.factory import build_model_client, describe_stage_route
from src.synthetic_ner.tasks.document_generation.critic import SectionCritic
from src.synthetic_ner.tasks.document_generation.generation_report import write_generation_report
from src.synthetic_ner.tasks.document_generation.memory_manager import CaseMemoryManager
from src.synthetic_ner.tasks.document_generation.planner import Planner
from src.synthetic_ner.tasks.document_generation.prompt_context import build_section_contract
from src.synthetic_ner.tasks.document_generation.tracer import TraceStore
from src.synthetic_ner.tasks.document_generation.validators import (
    clean_generated_section_text,
    validate_section_text,
)
from src.synthetic_ner.tasks.document_generation.writer import SectionWriter
from src.synthetic_ner.types.orchesterator import SectionWorkflowResult, WorkflowState


def run_langgraph_workflow(args: Namespace, project_root: Path) -> None:
    context = build_runtime_context(args, project_root)
    size_label = build_size_label(context)

    for document_index in range(context.documents):
        print(
            f"\n[{document_index + 1}/{context.documents}] Generating "
            f"{context.doc_type} / {context.fraud_type} / {size_label} via langgraph …"
        )

        document = resolve_document_inputs(context)
        doc_id, schema = resolve_schema_for_document(context, document, document_index)
        run_document_graph(
            context=context,
            document=document,
            schema=schema,
            doc_id=doc_id,
        )

    print("\nDone.")


def run_document_graph(
    *,
    context,
    document,
    schema: dict,
    doc_id: str,
    workflow_run_id: str | None = None,
    prefect_flow_run_id: str | None = None,
) -> None:
    if not context.workflow_cfg.writer.active:
        raise ValueError("workflow.writer.active must be true for document generation.")

    trace_store = TraceStore(
        context.langfuse_cfg,
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
        schema=schema,
        section_order=list(context.section_word_targets.keys()),
    )

    planner_client = None
    if context.workflow_cfg.planner.active:
        planner_client = build_model_client(
            stage="planner",
            routing=context.model_routing_cfg,
            tracer=trace_store,
        )
    writer_client = build_model_client(
        stage="writer",
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
            for stage in _active_model_stages(context.workflow_cfg)
        )
    )
    planner = None
    if planner_client is not None:
        planner = Planner(
            client=planner_client,
            prompts=prompts,
            planner_temperature=context.workflow_cfg.planner.temperature,
            document_max_output_tokens=(
                context.workflow_cfg.planner.document_max_output_tokens
            ),
            section_max_output_tokens=(
                context.workflow_cfg.planner.section_max_output_tokens
            ),
            prompt_clients=resolved_prompts.prompt_clients,
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

    trace_info = trace_store.start_document_run(
        doc_id=doc_id,
        name="document-workflow",
        input_payload={
            "doc_id": doc_id,
            "doc_type": context.doc_type,
            "fraud_type": context.fraud_type,
            "section_order": list(context.section_word_targets.keys()),
        },
        metadata={
            "doc_id": doc_id,
            "doc_type": context.doc_type,
            "fraud_type": context.fraud_type,
            "case_number": document.metadata["case_number"],
            "planner_active": context.workflow_cfg.planner.active,
            "writer_active": context.workflow_cfg.writer.active,
            "critic_active": context.workflow_cfg.critic.active,
        },
    )
    if trace_info.trace_url:
        print(f"  Trace   : {trace_info.trace_url}")

    final_state = None
    try:
        seed_memory_text = memory_manager.read_memory(memory_path)
        graph = build_document_graph(
            context=context,
            document=document,
            schema=schema,
            doc_id=doc_id,
            memory_path=memory_path,
            memory_manager=memory_manager,
            planner=planner,
            writer=writer,
            critic=critic,
            trace_store=trace_store,
        )
        final_state = graph.invoke(
            {
                "doc_id": doc_id,
                "memory_path": memory_path,
                "memory_text": seed_memory_text,
                "section_order": list(context.section_word_targets.keys()),
                "section_index": 0,
                "section_outputs": {},
                "section_plans": {},
                "section_contracts": {},
                "section_reviews": {},
                "instruction_channel": {},
                "review_channel": {},
                "content_channel": {},
            }
        )
    finally:
        trace_store.end_document_run(
            output_payload={
                "doc_id": doc_id,
                "rendered": bool(final_state and final_state.get("final_text")),
            }
        )


def _active_model_stages(workflow_cfg) -> tuple[str, ...]:
    stages = ["writer"]
    if workflow_cfg.planner.active:
        stages.insert(0, "planner")
    if workflow_cfg.critic.active:
        stages.append("critic")
    return tuple(stages)


def build_document_graph(
    *,
    context,
    document,
    schema: dict,
    doc_id: str,
    memory_path: Path,
    memory_manager: CaseMemoryManager,
    planner: Planner | None,
    writer: SectionWriter,
    critic: SectionCritic | None,
    trace_store: TraceStore,
) ->CompiledStateGraph:
    workflow = DocumentWorkflow(
        context=context,
        document=document,
        schema=schema,
        doc_id=doc_id,
        memory_path=memory_path,
        memory_manager=memory_manager,
        planner=planner,
        writer=writer,
        critic=critic,
        trace_store=trace_store,
    )
    return workflow.build_graph()


class DocumentWorkflow:
    def __init__(
        self,
        *,
        context,
        document,
        schema: dict,
        doc_id: str,
        memory_path: Path,
        memory_manager: CaseMemoryManager,
        planner: Planner | None,
        writer: SectionWriter,
        critic: SectionCritic | None,
        trace_store: TraceStore,
    ) -> None:
        self.context = context
        self.document = document
        self.schema = schema
        self.doc_id = doc_id
        self.memory_path = memory_path
        self.memory_manager = memory_manager
        self.planner = planner
        self.writer = writer
        self.critic = critic
        self.trace_store = trace_store

    def build_graph(self):
        builder = StateGraph(WorkflowState)
        self._register_nodes(builder)
        self._register_edges(builder)
        return builder.compile()

    def _register_nodes(self, builder: StateGraph) -> None:
        if self.planner is not None:
            builder.add_node(
                "document_planner",
                self._trace_node(
                    "document_planner",
                    self.document_planner_node,
                    next_node="process_sections",
                ),
            )
        builder.add_node(
            "process_sections",
            self._trace_node(
                "process_sections",
                self.process_sections_node,
                next_node="render_document",
            ),
        )
        builder.add_node(
            "render_document",
            self._trace_node(
                "render_document",
                self.render_document_node,
                next_node="END",
            ),
        )

    def _trace_node(
        self,
        node_name: str,
        handler: Callable[[WorkflowState], WorkflowState],
        *,
        next_node: str | None = None,
        next_node_resolver: Callable[[WorkflowState], str | None] | None = None,
    ) -> Callable[[WorkflowState], WorkflowState]:
        resolver = next_node_resolver
        if resolver is None and next_node is not None:
            def resolve_fixed_next_node(_state: WorkflowState) -> str | None:
                return next_node

            resolver = resolve_fixed_next_node

        @wraps(handler)
        def wrapped(state: WorkflowState) -> WorkflowState:
            return self.trace_store.run_langgraph_node(
                doc_id=self.doc_id,
                node_name=node_name,
                state=state,
                runner=lambda: handler(state),
                next_node_resolver=resolver,
            )

        return wrapped

    def _register_edges(self, builder: StateGraph) -> None:
        if self.planner is None:
            builder.add_edge(START, "process_sections")
        else:
            builder.add_edge(START, "document_planner")
            builder.add_edge("document_planner", "process_sections")
        builder.add_edge("process_sections", "render_document")
        builder.add_edge("render_document", END)

    def document_planner_node(self, state: WorkflowState) -> WorkflowState:
        if self.planner is None:
            return {"document_plan": "", "instruction_channel": {}}
        document_plan = self.planner.plan_document(
            doc_id=self.doc_id,
            parent_task_id=None,
            memory_text=state["memory_text"],
            doc_type=self.context.doc_type,
            fraud_type=self.context.fraud_type,
            case_number=self.document.metadata["case_number"],
            section_order=state["section_order"],
        )
        self.memory_manager.append_document_plan(self.memory_path, document_plan)
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["document_plan"] = document_plan
        return {
            "document_plan": document_plan,
            "instruction_channel": instruction_channel,
        }

    def process_sections_node(self, state: WorkflowState) -> WorkflowState:
        section_order = state["section_order"]
        section_outputs = dict(state.get("section_outputs", {}))
        section_plans = dict(state.get("section_plans", {}))
        section_contracts = dict(state.get("section_contracts", {}))
        section_reviews = dict(state.get("section_reviews", {}))

        for group in _parallel_section_groups(section_order):
            if len(group) == 1:
                results = [self._run_section_workflow(state, group[0])]
            else:
                results_by_section: dict[str, SectionWorkflowResult] = {}
                with ThreadPoolExecutor(
                    max_workers=len(group),
                    thread_name_prefix="section-workflow",
                ) as executor:
                    futures = {
                        executor.submit(
                            self._run_section_workflow,
                            state,
                            section_name,
                        ): section_name
                        for section_name in group
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        results_by_section[result.section_name] = result
                results = [results_by_section[section_name] for section_name in group]

            for result in results:
                section_outputs[result.section_name] = result.section_text
                section_plans[result.section_name] = result.section_plan
                section_contracts[result.section_name] = result.section_contract
                section_reviews[result.section_name] = result.issues
                self.memory_manager.append_section_result(
                    self.memory_path,
                    section_name=result.section_name,
                    section_plan=result.section_plan,
                    section_text=result.section_text,
                    issues=result.issues,
                )

        return {
            "section_outputs": section_outputs,
            "section_plans": section_plans,
            "section_contracts": section_contracts,
            "section_reviews": section_reviews,
            "section_index": len(section_order),
        }

    def _run_section_workflow(
        self,
        state: WorkflowState,
        section_name: str,
    ) -> SectionWorkflowResult:
        section_contract = build_section_contract(section_name)
        document_plan = state.get("document_plan", "")
        section_plan = ""
        writer_parent_task_id = None
        if self.planner is not None:
            section_plan = self.planner.plan_section(
                doc_id=self.doc_id,
                parent_task_id="planner_document",
                memory_text=state["memory_text"],
                document_plan=document_plan,
                doc_type=self.context.doc_type,
                section_name=section_name,
                word_target=self.context.section_word_targets[section_name],
            )
            writer_parent_task_id = f"planner_{section_name}"
        section_text = self.writer.write_section(
            doc_id=self.doc_id,
            parent_task_id=writer_parent_task_id,
            memory_text=state["memory_text"],
            document_plan=document_plan,
            section_name=section_name,
            section_plan=section_plan,
            case_number=self.document.metadata["case_number"],
            word_target=self.context.section_word_targets[section_name],
        )
        section_text = clean_generated_section_text(section_text)
        issues: list[str] = []

        if self.critic is not None:
            review = self.critic.review_section(
                doc_id=self.doc_id,
                parent_task_id=f"writer_{section_name}",
                memory_text=state["memory_text"],
                section_name=section_name,
                section_plan=section_plan,
                section_text=section_text,
                revision_round=0,
            )
            issues = list(review.issues)
            if review.blocking and not issues:
                issues.append("Critic marked the section as blocking.")

        validator_issues = validate_section_text(
            section_name=section_name,
            section_text=section_text,
            memory_text=state["memory_text"],
            word_target=self.context.section_word_targets[section_name],
            min_completion_ratio=self.context.workflow_cfg.writer.min_completion_ratio,
            enabled_validators=self.context.workflow_cfg.validators,
        )
        for issue in validator_issues:
            if issue not in issues:
                issues.append(issue)

        final_text, final_issues = self._finalize_section_text(
            section_name=section_name,
            section_plan=section_plan,
            section_text=section_text,
            issues=issues,
        )
        return SectionWorkflowResult(
            section_name=section_name,
            section_plan=section_plan,
            section_contract=section_contract,
            section_text=final_text,
            issues=final_issues,
        )

    def _finalize_section_text(
        self,
        *,
        section_name: str,
        section_plan: str,
        section_text: str,
        issues: list[str],
    ) -> tuple[str, list[str]]:
        del section_plan
        word_target = self.context.section_word_targets[section_name]
        memory_text = self.memory_manager.read_memory(self.memory_path)
        final_text = clean_generated_section_text(section_text)

        final_issues = validate_section_text(
            section_name=section_name,
            section_text=final_text,
            memory_text=memory_text,
            word_target=word_target,
            min_completion_ratio=self.context.workflow_cfg.writer.min_completion_ratio,
            enabled_validators=self.context.workflow_cfg.validators,
        )
        if final_issues:
            print(
                "  Warning : section output still has validation issues "
                f"({section_name}): {'; '.join(final_issues)}"
            )
        combined_issues = list(issues)
        for issue in final_issues:
            if issue not in combined_issues:
                combined_issues.append(issue)
        return final_text, combined_issues

    def render_document_node(self, state: WorkflowState) -> WorkflowState:
        ordered_sections = [
            state.get("section_outputs", {}).get(section_name, "[missing section]")
            for section_name in state["section_order"]
        ]
        problems = collect_section_output_problems(
            self.context.section_word_targets,
            ordered_sections,
            min_completion_ratio=self.context.workflow_cfg.writer.min_completion_ratio,
        )
        if not self.context.workflow_cfg.validators.get("minimum_length", True):
            problems = [
                problem
                for problem in problems
                if " is too short for its target " not in problem
            ]
        if problems:
            raise RuntimeError(
                "Document render aborted because one or more sections are incomplete: "
                + "; ".join(problems)
            )

        rendered_text = render_document_text(self.context, self.document, ordered_sections)
        save_document_artifacts(
            self.context,
            self.document,
            self.doc_id,
            self.schema,
            rendered_text,
        )
        write_generation_report(
            context=self.context,
            doc_id=self.doc_id,
            memory_path=self.memory_path,
            document_plan=state.get("document_plan", ""),
            section_contracts=state.get("section_contracts", {}),
            section_plans=state.get("section_plans", {}),
            section_reviews=state.get("section_reviews", {}),
            trace_store=self.trace_store,
        )
        return {"final_text": rendered_text}

def _parallel_section_groups(section_order: list[str]) -> list[list[str]]:
    dependencies = {
        "facts": {"history", "charges"},
        "evidence": {"facts"},
        "assessment": {"facts"},
    }
    remaining = list(section_order)
    completed: set[str] = set()
    groups: list[list[str]] = []

    while remaining:
        ready = [
            section_name
            for section_name in remaining
            if dependencies.get(section_name, set()).issubset(completed)
        ]
        if not ready:
            ready = [remaining[0]]
        groups.append(ready)
        completed.update(ready)
        remaining = [section_name for section_name in remaining if section_name not in ready]

    return groups
