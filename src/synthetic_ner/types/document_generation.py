from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from src.synthetic_ner.tasks.document_generation.context.memory import CaseMemoryManager
    from src.synthetic_ner.tasks.document_generation.observability.tracer import TraceStore
    from src.synthetic_ner.tasks.document_generation.stages.critic import SectionCritic
    from src.synthetic_ner.tasks.document_generation.stages.polisher import SectionPolisher
    from src.synthetic_ner.tasks.document_generation.stages.writer import SectionWriter


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


@dataclass(slots=True)
class GenerationComponents:
    trace_store: TraceStore
    memory_manager: CaseMemoryManager
    memory_path: Path
    writer: SectionWriter
    polisher: SectionPolisher | None
    critic: SectionCritic | None


@dataclass(slots=True)
class WriterPacket:
    content: str
    facts_used: list[str]
    tone: str
    legal_risks: list[str]
    raw_text: str
    valid_json: bool
    parse_error: str | None = None


@dataclass(slots=True)
class AllowedFacts:
    person_surface_forms: set[str]
    titled_people: set[str]
    initials: set[str]
    org_names: set[str]
    vat_numbers: set[str]
    amounts: set[str]
    case_refs: set[str]
    dates: set[str]
