"""LangGraph orchestration for planner/writer/critic workflows."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from src.synthetic_ner.engine import (
    build_runtime_context,
    build_size_label,
    collect_section_output_problems,
    render_document_text,
    resolve_document_inputs,
    resolve_schema_for_document,
    save_document_artifacts,
)
from src.synthetic_ner.models.ollama_client import TracedOllamaClient
from src.synthetic_ner.tasks.critic import SectionCritic
from src.synthetic_ner.tasks.memory_manager import CaseMemoryManager
from src.synthetic_ner.tasks.planner import Planner
from src.synthetic_ner.tasks.tracer import TraceStore
from src.synthetic_ner.tasks.validators import (
    build_deterministic_fallback_section,
    clean_generated_section_text,
    repair_section_text,
    validate_section_text,
)
from src.synthetic_ner.tasks.writer import SectionWriter


class WorkflowState(TypedDict, total=False):
    doc_id: str
    memory_path: Path
    memory_text: str
    document_plan: str
    section_order: list[str]
    section_index: int
    current_section: str
    current_section_plan: str
    current_section_text: str
    current_section_issues: list[str]
    current_revision_instruction: str
    revision_count: int
    repair_attempts: int
    section_outputs: dict[str, str]
    section_plans: dict[str, str]
    section_reviews: dict[str, list[str]]
    instruction_channel: dict[str, Any]
    review_channel: dict[str, Any]
    content_channel: dict[str, Any]
    final_text: str


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


def run_document_graph(*, context, document, schema: dict, doc_id: str) -> None:
    trace_store = TraceStore(context.langfuse_cfg)
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

    client = TracedOllamaClient(
        config=context.ollama_cfg,
        tracer=trace_store,
    )
    resolved_prompts = trace_store.resolve_workflow_prompts(context.workflow_cfg.prompts)
    prompts = resolved_prompts.prompts
    print(f"  Prompts : {resolved_prompts.sync_summary}")
    planner = Planner(
        client=client,
        prompts=prompts,
        planner_temperature=context.workflow_cfg.planner.temperature,
        document_max_output_tokens=(
            context.workflow_cfg.planner.document_max_output_tokens
        ),
        section_max_output_tokens=context.workflow_cfg.planner.section_max_output_tokens,
        prompt_clients=resolved_prompts.prompt_clients,
    )
    writer = SectionWriter(
        client=client,
        prompts=prompts,
        chunk_words=context.workflow_cfg.writer.chunk_words,
        context_tail_chars=context.workflow_cfg.writer.context_tail_chars,
        writer_temperature=context.workflow_cfg.writer.temperature,
        max_output_tokens=context.workflow_cfg.writer.max_output_tokens,
        min_output_tokens=context.workflow_cfg.writer.min_output_tokens,
        output_token_multiplier=context.workflow_cfg.writer.output_token_multiplier,
        prompt_clients=resolved_prompts.prompt_clients,
    )
    critic = SectionCritic(
        client=client,
        prompts=prompts,
        critic_temperature=context.workflow_cfg.critic.temperature,
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
                "section_reviews": {},
                "instruction_channel": {},
                "review_channel": {},
                "content_channel": {},
                "revision_count": 0,
                "repair_attempts": 0,
            }
        )
    finally:
        trace_store.end_document_run(
            output_payload={
                "doc_id": doc_id,
                "rendered": bool(final_state and final_state.get("final_text")),
            }
        )


def build_document_graph(
    *,
    context,
    document,
    schema: dict,
    doc_id: str,
    memory_path: Path,
    memory_manager: CaseMemoryManager,
    planner: Planner,
    writer: SectionWriter,
    critic: SectionCritic,
    trace_store: TraceStore,
):
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
        planner: Planner,
        writer: SectionWriter,
        critic: SectionCritic,
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
        builder.add_node(
            "document_planner",
            self._trace_node(
                "document_planner",
                self.document_planner_node,
                next_node="prepare_section",
            ),
        )
        builder.add_node(
            "prepare_section",
            self._trace_node(
                "prepare_section",
                self.prepare_section_node,
                next_node_resolver=route_after_prepare,
            ),
        )
        builder.add_node(
            "plan_section",
            self._trace_node(
                "plan_section",
                self.plan_section_node,
                next_node="write_section",
            ),
        )
        builder.add_node(
            "write_section",
            self._trace_node(
                "write_section",
                self.write_section_node,
                next_node="critique_section",
            ),
        )
        builder.add_node(
            "critique_section",
            self._trace_node(
                "critique_section",
                self.critique_section_node,
                next_node="validate_section",
            ),
        )
        builder.add_node(
            "validate_section",
            self._trace_node(
                "validate_section",
                self.validate_section_node,
                next_node_resolver=self.route_after_validation,
            ),
        )
        builder.add_node(
            "revise_section",
            self._trace_node(
                "revise_section",
                self.revise_section_node,
                next_node="critique_section",
            ),
        )
        builder.add_node(
            "repair_section",
            self._trace_node(
                "repair_section",
                self.repair_section_node,
                next_node="validate_section",
            ),
        )
        builder.add_node(
            "store_section",
            self._trace_node(
                "store_section",
                self.store_section_node,
                next_node="prepare_section",
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
        builder.add_edge(START, "document_planner")
        builder.add_edge("document_planner", "prepare_section")
        builder.add_conditional_edges(
            "prepare_section",
            route_after_prepare,
            {
                "plan_section": "plan_section",
                "render_document": "render_document",
            },
        )
        builder.add_edge("plan_section", "write_section")
        builder.add_edge("write_section", "critique_section")
        builder.add_edge("critique_section", "validate_section")
        builder.add_conditional_edges(
            "validate_section",
            self.route_after_validation,
            {
                "revise_section": "revise_section",
                "repair_section": "repair_section",
                "store_section": "store_section",
            },
        )
        builder.add_edge("revise_section", "critique_section")
        builder.add_edge("repair_section", "validate_section")
        builder.add_edge("store_section", "prepare_section")
        builder.add_edge("render_document", END)

    def document_planner_node(self, state: WorkflowState) -> WorkflowState:
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

    def prepare_section_node(self, state: WorkflowState) -> WorkflowState:
        section_index = state.get("section_index", 0)
        if section_index >= len(state["section_order"]):
            return {"current_section": ""}
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["current_revision_instruction"] = ""
        review_channel = dict(state.get("review_channel", {}))
        review_channel["current_section_issues"] = []
        review_channel["critic_rubrics"] = {}
        review_channel["critic_approved"] = False
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = ""
        return {
            "current_section": state["section_order"][section_index],
            "current_section_plan": "",
            "current_section_text": "",
            "current_section_issues": [],
            "current_revision_instruction": "",
            "instruction_channel": instruction_channel,
            "review_channel": review_channel,
            "content_channel": content_channel,
            "revision_count": 0,
            "repair_attempts": 0,
        }

    def plan_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        section_plan = self.planner.plan_section(
            doc_id=self.doc_id,
            parent_task_id="planner_document",
            memory_text=state["memory_text"],
            document_plan=state["document_plan"],
            doc_type=self.context.doc_type,
            section_name=section_name,
            word_target=self.context.section_word_targets[section_name],
        )
        section_plans = dict(state.get("section_plans", {}))
        section_plans[section_name] = section_plan
        return {
            "current_section_plan": section_plan,
            "section_plans": section_plans,
        }

    def write_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        section_text = self.writer.write_section(
            doc_id=self.doc_id,
            parent_task_id=f"planner_{section_name}",
            memory_text=state["memory_text"],
            document_plan=state["document_plan"],
            section_name=section_name,
            section_plan=state["current_section_plan"],
            case_number=self.document.metadata["case_number"],
            word_target=self.context.section_word_targets[section_name],
        )
        clean_text = clean_generated_section_text(section_text)
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = clean_text
        return {
            "current_section_text": clean_text,
            "content_channel": content_channel,
        }

    def critique_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        current_text = _current_section_text(state)
        review = self.critic.review_section(
            doc_id=self.doc_id,
            parent_task_id=f"writer_{section_name}",
            memory_text=state["memory_text"],
            section_name=section_name,
            section_plan=state["current_section_plan"],
            section_text=current_text,
            revision_round=state.get("revision_count", 0),
        )
        review_channel = dict(state.get("review_channel", {}))
        review_channel["current_section_issues"] = list(review.issues)
        review_channel["critic_rubrics"] = dict(review.rubrics)
        review_channel["critic_approved"] = bool(review.approved)
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["current_revision_instruction"] = review.revision_instruction
        return {
            "current_section_issues": review.issues,
            "current_revision_instruction": review.revision_instruction,
            "review_channel": review_channel,
            "instruction_channel": instruction_channel,
        }

    def validate_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        current_text = clean_generated_section_text(_current_section_text(state))
        issues = list(_current_section_issues(state))
        validator_issues = validate_section_text(
            section_name=section_name,
            section_text=current_text,
            memory_text=state["memory_text"],
            word_target=self.context.section_word_targets[section_name],
        )
        for issue in validator_issues:
            if issue not in issues:
                issues.append(issue)

        revision_instruction = _current_revision_instruction(state)
        if validator_issues:
            validator_lines = "\n".join(f"- {issue}" for issue in validator_issues)
            validator_block = (
                "Fix the deterministic validation issues before approving:\n"
                f"{validator_lines}"
            )
            if revision_instruction:
                revision_instruction = f"{revision_instruction}\n\n{validator_block}"
            else:
                revision_instruction = validator_block
        review_channel = dict(state.get("review_channel", {}))
        review_channel["current_section_issues"] = list(issues)
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["current_revision_instruction"] = revision_instruction
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = current_text
        return {
            "current_section_text": current_text,
            "current_section_issues": issues,
            "current_revision_instruction": revision_instruction,
            "review_channel": review_channel,
            "instruction_channel": instruction_channel,
            "content_channel": content_channel,
        }

    def revise_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        revision_count = state.get("revision_count", 0) + 1
        revised_text = self.writer.write_section(
            doc_id=self.doc_id,
            parent_task_id=f"critic_{section_name}",
            memory_text=state["memory_text"],
            document_plan=state["document_plan"],
            section_name=section_name,
            section_plan=state["current_section_plan"],
            case_number=self.document.metadata["case_number"],
            word_target=self.context.section_word_targets[section_name],
            revision_instruction=_current_revision_instruction(state),
            revision_round=revision_count,
        )
        clean_text = clean_generated_section_text(revised_text)
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = clean_text
        return {
            "current_section_text": clean_text,
            "content_channel": content_channel,
            "revision_count": revision_count,
        }

    def repair_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        issues = list(_current_section_issues(state))
        current_text = _current_section_text(state)
        repaired_text = repair_section_text(
            section_text=current_text,
            issues=issues,
            memory_text=state["memory_text"],
        )
        repaired_text = clean_generated_section_text(repaired_text)
        repair_attempts = state.get("repair_attempts", 0) + 1
        if repaired_text and repaired_text != current_text:
            content_channel = dict(state.get("content_channel", {}))
            content_channel["current_section_text"] = repaired_text
            return {
                "current_section_text": repaired_text,
                "content_channel": content_channel,
                "repair_attempts": repair_attempts,
            }

        revision_count = state.get("revision_count", 0) + 1
        forced_instruction = (
            "Rewrite the section using only entities, organisations, dates, case references, "
            "and VAT/reference numbers listed in CASE_MEMORY. "
            "If a value is not listed, omit it. "
            f"Resolve these issues:\n{chr(10).join(f'- {issue}' for issue in issues) or '- none'}"
        )
        rewritten_text = self.writer.write_section(
            doc_id=self.doc_id,
            parent_task_id=f"repair_{section_name}",
            memory_text=state["memory_text"],
            document_plan=state["document_plan"],
            section_name=section_name,
            section_plan=state["current_section_plan"],
            case_number=self.document.metadata["case_number"],
            word_target=self.context.section_word_targets[section_name],
            revision_instruction=forced_instruction,
            revision_round=revision_count,
        )
        clean_rewritten_text = clean_generated_section_text(rewritten_text)
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = clean_rewritten_text
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["current_revision_instruction"] = forced_instruction
        return {
            "current_section_text": clean_rewritten_text,
            "current_revision_instruction": forced_instruction,
            "revision_count": revision_count,
            "repair_attempts": repair_attempts,
            "content_channel": content_channel,
            "instruction_channel": instruction_channel,
        }

    def store_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        word_target = self.context.section_word_targets[section_name]
        issues = list(_current_section_issues(state))
        current_text = clean_generated_section_text(_current_section_text(state))

        final_text = current_text
        if issues:
            repaired_text = repair_section_text(
                section_text=current_text,
                issues=issues,
                memory_text=state["memory_text"],
            )
            final_text = clean_generated_section_text(repaired_text) or current_text

        final_issues = validate_section_text(
            section_name=section_name,
            section_text=final_text,
            memory_text=state["memory_text"],
            word_target=word_target,
        )

        if final_issues:
            fallback_text = build_deterministic_fallback_section(
                section_name=section_name,
                memory_text=state["memory_text"],
                word_target=word_target,
            )
            fallback_text = clean_generated_section_text(fallback_text)
            fallback_issues = validate_section_text(
                section_name=section_name,
                section_text=fallback_text,
                memory_text=state["memory_text"],
                word_target=word_target,
            )
            if fallback_text:
                final_text = fallback_text
                final_issues = fallback_issues
            print(
                "  Warning : replaced invalid section output with deterministic fallback "
                f"({section_name}). Remaining issues: "
                f"{'; '.join(final_issues) if final_issues else 'none'}"
            )

        section_outputs = dict(state.get("section_outputs", {}))
        section_outputs[section_name] = final_text
        section_reviews = dict(state.get("section_reviews", {}))
        section_reviews[section_name] = final_issues
        self.memory_manager.append_section_result(
            self.memory_path,
            section_name=section_name,
            section_plan=state["current_section_plan"],
            section_text=final_text,
            issues=final_issues,
        )
        review_channel = dict(state.get("review_channel", {}))
        review_channel["current_section_issues"] = list(final_issues)
        content_channel = dict(state.get("content_channel", {}))
        content_channel["current_section_text"] = final_text
        instruction_channel = dict(state.get("instruction_channel", {}))
        instruction_channel["current_revision_instruction"] = ""
        return {
            "section_outputs": section_outputs,
            "section_reviews": section_reviews,
            "section_index": state.get("section_index", 0) + 1,
            "current_section_text": final_text,
            "current_section_issues": final_issues,
            "current_revision_instruction": "",
            "review_channel": review_channel,
            "content_channel": content_channel,
            "instruction_channel": instruction_channel,
            "repair_attempts": 0,
        }

    def render_document_node(self, state: WorkflowState) -> WorkflowState:
        ordered_sections = [
            state.get("section_outputs", {}).get(section_name, "[missing section]")
            for section_name in state["section_order"]
        ]
        problems = collect_section_output_problems(
            self.context.section_word_targets,
            ordered_sections,
        )
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
            section_plans=state.get("section_plans", {}),
            section_reviews=state.get("section_reviews", {}),
            trace_store=self.trace_store,
        )
        return {"final_text": rendered_text}

    def route_after_validation(self, state: WorkflowState) -> str:
        return route_after_validation(
            state,
            self.context.workflow_cfg.max_revisions,
        )


def route_after_prepare(state: WorkflowState) -> str:
    if state.get("current_section"):
        return "plan_section"
    return "render_document"


def route_after_validation(state: WorkflowState, max_revisions: int) -> str:
    issues = _current_section_issues(state)
    revision_count = state.get("revision_count", 0)
    repair_attempts = state.get("repair_attempts", 0)
    if issues and revision_count < max_revisions:
        return "revise_section"
    if issues and repair_attempts < 1:
        return "repair_section"
    return "store_section"


def _current_section_text(state: Mapping[str, Any]) -> str:
    content_channel = state.get("content_channel")
    if isinstance(content_channel, Mapping):
        value = content_channel.get("current_section_text")
        if isinstance(value, str):
            return value
    fallback = state.get("current_section_text")
    return fallback if isinstance(fallback, str) else ""


def _current_section_issues(state: Mapping[str, Any]) -> list[str]:
    review_channel = state.get("review_channel")
    if isinstance(review_channel, Mapping):
        value = review_channel.get("current_section_issues")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
    fallback = state.get("current_section_issues")
    if isinstance(fallback, list):
        return [item for item in fallback if isinstance(item, str)]
    return []


def _current_revision_instruction(state: Mapping[str, Any]) -> str:
    instruction_channel = state.get("instruction_channel")
    if isinstance(instruction_channel, Mapping):
        value = instruction_channel.get("current_revision_instruction")
        if isinstance(value, str):
            return value
    fallback = state.get("current_revision_instruction")
    return fallback if isinstance(fallback, str) else ""


def write_generation_report(
    *,
    context,
    doc_id: str,
    memory_path: Path,
    document_plan: str,
    section_plans: dict[str, str],
    section_reviews: dict[str, list[str]],
    trace_store: TraceStore,
) -> Path:
    trace_info = trace_store.get_trace_info()
    node_summary = trace_store.get_langgraph_node_summary()
    llm_summary = trace_store.get_llm_run_summary()
    llm_calls = trace_store.get_llm_call_records()
    report_path = context.output_dir / doc_id / "generation_report.md"
    lines = [
        f"# Generation Report: {doc_id}",
        "",
        f"- Workflow mode: {context.workflow_cfg.mode}",
        f"- Memory file: {memory_path}",
        f"- Langfuse enabled: {str(trace_info.enabled).lower()}",
        f"- Langfuse trace id: {trace_info.trace_id or 'n/a'}",
        f"- Langfuse trace url: {trace_info.trace_url or 'n/a'}",
        f"- Total LLM calls: {llm_summary['total_llm_calls']}",
        f"- Total LLM latency ms: {llm_summary['total_latency_ms']}",
        f"- Empty LLM responses: {llm_summary['empty_responses']}",
        f"- Truncated LLM calls: {llm_summary['truncated_calls']}",
        "",
        "## Document Plan",
        document_plan or "- none",
        "",
        "## Section Results",
    ]
    for section_name in section_plans:
        issues = section_reviews.get(section_name, [])
        lines.extend(
            [
                f"### {section_name}",
                "",
                "Plan:",
                section_plans[section_name],
                "",
                "Issues:",
                *([f"- {issue}" for issue in issues] or ["- none"]),
                "",
            ]
        )
    if node_summary:
        lines.extend(
            [
                "## LangGraph Node Analytics",
                "",
                "Each node below also appears as its own child span in Langfuse.",
                "",
                "| Node | Executions | Avg ms | Max ms | Errors | Next Nodes |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in node_summary:
            next_nodes = ", ".join(row["next_nodes"]) if row["next_nodes"] else "-"
            lines.append(
                "| "
                f"{row['node_name']} | "
                f"{row['executions']} | "
                f"{row['avg_latency_ms']} | "
                f"{row['max_latency_ms']} | "
                f"{row['errors']} | "
                f"{next_nodes} |"
            )
        lines.append("")
    if llm_calls:
        lines.extend(_format_llm_analytics(llm_summary, llm_calls))
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _format_llm_analytics(
    llm_summary: dict[str, Any],
    llm_calls: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## LLM Run Analytics",
        "",
        "### Stage Totals",
        "",
        "| Stage | Calls | Total ms | Avg ms | Prompt Tokens | Response Tokens | Empty | Truncated | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in llm_summary["by_stage"]:
        lines.append(
            "| "
            f"{row['stage']} | "
            f"{row['calls']} | "
            f"{row['total_latency_ms']} | "
            f"{row['avg_latency_ms']} | "
            f"{row['prompt_tokens']} | "
            f"{row['response_tokens']} | "
            f"{row['empty_responses']} | "
            f"{row['truncated_calls']} | "
            f"{row['errors']} |"
        )

    revised_sections = llm_summary["sections_with_revisions"]
    lines.extend(
        [
            "",
            "### Bottleneck Candidates",
            "",
            f"- Slowest call: {_format_call_summary(llm_summary['slowest_call'], 'latency_ms')}",
            f"- Largest prompt: {_format_call_summary(llm_summary['largest_prompt'], 'prompt_chars')}",
            f"- Largest response: {_format_call_summary(llm_summary['largest_response'], 'response_chars')}",
            "- Sections with revisions: "
            + (", ".join(revised_sections) if revised_sections else "none"),
            "",
            "### LLM Calls",
            "",
            "| Task | Stage | Section | Rev | Prompt chars | Response chars | Prompt tokens | Response tokens | Budget | ms | Done | Empty |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for call in llm_calls:
        lines.append(
            "| "
            f"{call.get('task_id') or '-'} | "
            f"{call.get('stage') or '-'} | "
            f"{call.get('section_name') or '-'} | "
            f"{_format_optional_int(call.get('revision_round'))} | "
            f"{_format_optional_int(call.get('prompt_chars'))} | "
            f"{_format_optional_int(call.get('response_chars'))} | "
            f"{_format_optional_int(call.get('tokens_prompt'))} | "
            f"{_format_optional_int(call.get('tokens_response'))} | "
            f"{_format_optional_int(call.get('output_budget'))} | "
            f"{_format_optional_int(call.get('latency_ms'))} | "
            f"{call.get('done_reason') or '-'} | "
            f"{str(call.get('response_empty')).lower()} |"
        )
    lines.append("")
    return lines


def _format_optional_int(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    return str(value) if isinstance(value, int) else "-"


def _format_call_summary(call: dict[str, Any] | None, field_name: str) -> str:
    if not call:
        return "n/a"
    task_id = call.get("task_id") or "-"
    stage = call.get("stage") or "-"
    section = call.get("section_name") or "-"
    value = call.get(field_name)
    return f"{task_id} ({stage}/{section}, {field_name}={value})"
