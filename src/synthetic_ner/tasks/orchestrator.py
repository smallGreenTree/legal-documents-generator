"""LangGraph orchestration for planner/writer/critic workflows."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TypedDict

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
from src.synthetic_ner.tasks.validators import validate_section_text
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
    section_outputs: dict[str, str]
    section_plans: dict[str, str]
    section_reviews: dict[str, list[str]]
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
    trace_store = TraceStore(context.trace_dir, context.langfuse_cfg)
    memory_manager = CaseMemoryManager(
        context.memory_dir,
        summary_chars=context.workflow_cfg.get("memory_summary_chars", 500),
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
        base_url=context.ollama_cfg["base_url"],
        model=context.ollama_cfg["model"],
        timeout=context.ollama_cfg.get("timeout", 180),
        tracer=trace_store,
    )
    prompts = context.workflow_cfg.get("prompts", {})
    planner = Planner(
        client=client,
        prompts=prompts,
        planner_temperature=context.workflow_cfg.get("planner", {}).get("temperature", 0.2),
    )
    writer = SectionWriter(
        client=client,
        prompts=prompts,
        chunk_words=context.workflow_cfg.get("writer", {}).get("chunk_words", 700),
        context_tail_chars=context.workflow_cfg.get("writer", {}).get(
            "context_tail_chars", 600
        ),
        writer_temperature=context.workflow_cfg.get("writer", {}).get("temperature", 0.7),
    )
    critic = SectionCritic(
        client=client,
        prompts=prompts,
        critic_temperature=context.workflow_cfg.get("critic", {}).get("temperature", 0.2),
    )

    trace_store.start_document_run(
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
    final_state = None
    try:
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
                "memory_text": memory_manager.read_memory(memory_path),
                "section_order": list(context.section_word_targets.keys()),
                "section_index": 0,
                "section_outputs": {},
                "section_plans": {},
                "section_reviews": {},
                "revision_count": 0,
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
        builder.add_node("document_planner", self.document_planner_node)
        builder.add_node("prepare_section", self.prepare_section_node)
        builder.add_node("plan_section", self.plan_section_node)
        builder.add_node("write_section", self.write_section_node)
        builder.add_node("critique_section", self.critique_section_node)
        builder.add_node("validate_section", self.validate_section_node)
        builder.add_node("revise_section", self.revise_section_node)
        builder.add_node("store_section", self.store_section_node)
        builder.add_node("render_document", self.render_document_node)

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
                "store_section": "store_section",
            },
        )
        builder.add_edge("revise_section", "critique_section")
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
        return {
            "document_plan": document_plan,
            "memory_text": self.memory_manager.read_memory(self.memory_path),
        }

    def prepare_section_node(self, state: WorkflowState) -> WorkflowState:
        section_index = state.get("section_index", 0)
        if section_index >= len(state["section_order"]):
            return {"current_section": ""}
        return {
            "current_section": state["section_order"][section_index],
            "current_section_plan": "",
            "current_section_text": "",
            "current_section_issues": [],
            "current_revision_instruction": "",
            "revision_count": 0,
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
        return {"current_section_text": section_text}

    def critique_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        review = self.critic.review_section(
            doc_id=self.doc_id,
            parent_task_id=f"writer_{section_name}",
            memory_text=state["memory_text"],
            section_name=section_name,
            section_plan=state["current_section_plan"],
            section_text=state["current_section_text"],
            revision_round=state.get("revision_count", 0),
        )
        return {
            "current_section_issues": review.issues,
            "current_revision_instruction": review.revision_instruction,
        }

    def validate_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        issues = list(state.get("current_section_issues", []))
        validator_issues = validate_section_text(
            section_name=section_name,
            section_text=state["current_section_text"],
            document=self.document,
            word_target=self.context.section_word_targets[section_name],
        )
        for issue in validator_issues:
            if issue not in issues:
                issues.append(issue)

        revision_instruction = state.get("current_revision_instruction", "")
        if validator_issues and not revision_instruction:
            validator_lines = "\n".join(f"- {issue}" for issue in validator_issues)
            revision_instruction = (
                "Fix the deterministic validation issues before approving:\n"
                f"{validator_lines}"
            )
        return {
            "current_section_issues": issues,
            "current_revision_instruction": revision_instruction,
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
            revision_instruction=state.get("current_revision_instruction", ""),
            revision_round=revision_count,
        )
        return {
            "current_section_text": revised_text,
            "revision_count": revision_count,
        }

    def store_section_node(self, state: WorkflowState) -> WorkflowState:
        section_name = state["current_section"]
        issues = list(state.get("current_section_issues", []))
        if issues:
            raise RuntimeError(
                f"Section '{section_name}' failed validation: {'; '.join(issues)}"
            )

        section_outputs = dict(state.get("section_outputs", {}))
        section_outputs[section_name] = state["current_section_text"]
        section_reviews = dict(state.get("section_reviews", {}))
        section_reviews[section_name] = issues
        self.memory_manager.append_section_result(
            self.memory_path,
            section_name=section_name,
            section_plan=state["current_section_plan"],
            section_text=state["current_section_text"],
            issues=section_reviews[section_name],
        )
        return {
            "section_outputs": section_outputs,
            "section_reviews": section_reviews,
            "section_index": state.get("section_index", 0) + 1,
            "memory_text": self.memory_manager.read_memory(self.memory_path),
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
            self.context.workflow_cfg.get("max_revisions", 2),
        )


def route_after_prepare(state: WorkflowState) -> str:
    if state.get("current_section"):
        return "plan_section"
    return "render_document"


def route_after_validation(state: WorkflowState, max_revisions: int) -> str:
    issues = state.get("current_section_issues", [])
    revision_count = state.get("revision_count", 0)
    if issues and revision_count < max_revisions:
        return "revise_section"
    return "store_section"


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
    trace_index_path = trace_store.write_trace_index(doc_id)
    trace_info = trace_store.get_trace_info(doc_id)
    report_path = context.output_dir / doc_id / "generation_report.md"
    lines = [
        f"# Generation Report: {doc_id}",
        "",
        f"- Workflow mode: {context.workflow_cfg.get('mode', 'langgraph')}",
        f"- Memory file: {memory_path}",
        f"- Trace directory: {context.trace_dir / doc_id}",
        f"- Langfuse enabled: {str(trace_info.enabled).lower()}",
        f"- Langfuse trace id: {trace_info.trace_id or 'n/a'}",
        f"- Langfuse trace url: {trace_info.trace_url or 'n/a'}",
        f"- Trace index: {trace_index_path}",
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
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path
