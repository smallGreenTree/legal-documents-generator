from src.synthetic_ner.tasks.document_generation.orchestration.graph import (
    parallel_section_groups,
)
from src.synthetic_ner.tasks.document_generation.orchestration.runner import run_document_graph
from src.synthetic_ner.tasks.document_generation.orchestration.section import (
    combine_revision_instruction,
)
from src.synthetic_ner.tasks.document_generation.orchestrator import (
    run_document_graph as facade_run_document_graph,
)


def test_section_groups_respect_document_dependencies():
    section_order = [
        "persons",
        "companies",
        "history",
        "charges",
        "facts",
        "evidence",
        "assessment",
    ]

    assert parallel_section_groups(section_order) == [
        ["persons", "companies", "history", "charges"],
        ["facts"],
        ["evidence", "assessment"],
    ]


def test_revision_instruction_contains_every_issue_and_critic_proposal():
    instruction = combine_revision_instruction(
        critic_instruction="Replace the unsupported conclusion.",
        issues=[f"issue {index}" for index in range(1, 9)],
    )

    assert all(f"- issue {index}" in instruction for index in range(1, 9))
    assert "Replace the unsupported conclusion." in instruction


def test_orchestrator_keeps_stable_document_runner_entry_point():
    assert facade_run_document_graph is run_document_graph
