"""LangGraph definition for composing and rendering document sections."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import wraps

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.synthetic_ner.document.engine import (
    collect_section_output_problems,
    render_document_text,
    save_document_artifacts,
)
from src.synthetic_ner.tasks.document_generation.artifacts.generation_report import (
    write_generation_report,
)
from src.synthetic_ner.tasks.document_generation.constants import SECTION_DEPENDENCIES
from src.synthetic_ner.tasks.document_generation.orchestration.section import (
    SectionWorkflowRunner,
)
from src.synthetic_ner.types.document_generation import (
    GenerationComponents,
    SectionWorkflowResult,
    WorkflowState,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext


def build_document_graph(
    *,
    context: RuntimeContext,
    document: DocumentInputs,
    doc_id: str,
    components: GenerationComponents,
) -> CompiledStateGraph:
    workflow = DocumentWorkflow(
        context=context,
        document=document,
        doc_id=doc_id,
        components=components,
    )
    return workflow.build_graph()


class DocumentWorkflow:
    def __init__(
        self,
        *,
        context: RuntimeContext,
        document: DocumentInputs,
        doc_id: str,
        components: GenerationComponents,
    ) -> None:
        self.context = context
        self.document = document
        self.doc_id = doc_id
        self.components = components
        self.section_runner = SectionWorkflowRunner(
            context=context,
            document=document,
            doc_id=doc_id,
            memory_path=components.memory_path,
            memory_manager=components.memory_manager,
            writer=components.writer,
            polisher=components.polisher,
            critic=components.critic,
        )

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(WorkflowState)
        self._register_nodes(builder)
        builder.add_edge(START, "process_sections")
        builder.add_edge("process_sections", "render_document")
        builder.add_edge("render_document", END)
        return builder.compile()

    def _register_nodes(self, builder: StateGraph) -> None:
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
        next_node: str,
    ) -> Callable[[WorkflowState], WorkflowState]:
        @wraps(handler)
        def wrapped(state: WorkflowState) -> WorkflowState:
            return self.components.trace_store.run_langgraph_node(
                doc_id=self.doc_id,
                node_name=node_name,
                state=state,
                runner=lambda: handler(state),
                next_node_resolver=lambda _state: next_node,
            )

        return wrapped

    def process_sections_node(self, state: WorkflowState) -> WorkflowState:
        section_order = state["section_order"]
        section_outputs = dict(state.get("section_outputs", {}))
        section_contracts = dict(state.get("section_contracts", {}))
        section_reviews = dict(state.get("section_reviews", {}))

        for group in parallel_section_groups(section_order):
            results = self._process_section_group(
                group=group,
                memory_text=state["memory_text"],
            )
            for result in results:
                section_outputs[result.section_name] = result.section_text
                section_contracts[result.section_name] = result.section_contract
                section_reviews[result.section_name] = result.issues
                self.components.memory_manager.append_section_result(
                    self.components.memory_path,
                    section_name=result.section_name,
                    section_text=result.section_text,
                    issues=result.issues,
                )

        return {
            "section_outputs": section_outputs,
            "section_contracts": section_contracts,
            "section_reviews": section_reviews,
        }

    def _process_section_group(
        self,
        *,
        group: list[str],
        memory_text: str,
    ) -> list[SectionWorkflowResult]:
        if len(group) == 1:
            return [
                self.section_runner.run(
                    memory_text=memory_text,
                    section_name=group[0],
                )
            ]

        results_by_section: dict[str, SectionWorkflowResult] = {}
        with ThreadPoolExecutor(
            max_workers=len(group),
            thread_name_prefix="section-workflow",
        ) as executor:
            futures = {
                executor.submit(
                    copy_context().run,
                    self.section_runner.run,
                    memory_text=memory_text,
                    section_name=section_name,
                ): section_name
                for section_name in group
            }
            for future in as_completed(futures):
                result = future.result()
                results_by_section[result.section_name] = result
        return [results_by_section[section_name] for section_name in group]

    def render_document_node(self, state: WorkflowState) -> WorkflowState:
        ordered_sections = [
            state.get("section_outputs", {}).get(section_name, "[missing section]")
            for section_name in state["section_order"]
        ]
        problems = collect_section_output_problems(
            self.context.section_order,
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
            rendered_text,
        )
        write_generation_report(
            context=self.context,
            doc_id=self.doc_id,
            memory_path=self.components.memory_path,
            section_contracts=state.get("section_contracts", {}),
            section_reviews=state.get("section_reviews", {}),
            trace_store=self.components.trace_store,
        )
        return {"final_text": rendered_text}


def parallel_section_groups(section_order: list[str]) -> list[list[str]]:
    remaining = list(section_order)
    completed: set[str] = set()
    groups: list[list[str]] = []

    while remaining:
        ready = [
            section_name
            for section_name in remaining
            if SECTION_DEPENDENCIES.get(section_name, frozenset()).issubset(completed)
        ]
        if not ready:
            ready = [remaining[0]]
        groups.append(ready)
        completed.update(ready)
        remaining = [section_name for section_name in remaining if section_name not in ready]

    return groups
