from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    doc_id: str
    memory_path: Path
    memory_text: str
    section_order: list[str]
    section_outputs: dict[str, str]
    section_contracts: dict[str, str]
    section_reviews: dict[str, list[str]]
    final_text: str


@dataclass(slots=True)
class SectionWorkflowResult:
    section_name: str
    section_contract: str
    section_text: str
    issues: list[str]
