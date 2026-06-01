from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    doc_id: str
    memory_path: Path
    memory_text: str
    document_plan: str
    section_order: list[str]
    section_index: int
    current_section: str
    current_section_contract: str
    current_section_plan: str
    current_section_text: str
    current_section_issues: list[str]
    current_revision_instruction: str
    revision_count: int
    repair_attempts: int
    section_outputs: dict[str, str]
    section_plans: dict[str, str]
    section_contracts: dict[str, str]
    section_reviews: dict[str, list[str]]
    revision_counts: dict[str, int]
    instruction_channel: dict[str, Any]
    review_channel: dict[str, Any]
    content_channel: dict[str, Any]
    final_text: str


@dataclass(slots=True)
class SectionWorkflowResult:
    section_name: str
    section_plan: str
    section_contract: str
    section_text: str
    issues: list[str]
    revisions: int
