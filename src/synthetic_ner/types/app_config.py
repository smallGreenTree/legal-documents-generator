from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str
    schema_dir: str
    memory_dir: str
    trace_dir: str


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    timeout: int


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool
    base_url: str
    public_key_env: str
    secret_key_env: str


@dataclass(frozen=True)
class GenerationConfig:
    words_per_page: int


@dataclass(frozen=True)
class PlannerConfig:
    temperature: float


@dataclass(frozen=True)
class WriterConfig:
    chunk_words: int
    context_tail_chars: int
    temperature: float


@dataclass(frozen=True)
class CriticConfig:
    temperature: float


@dataclass(frozen=True)
class WorkflowPromptsConfig:
    document_planner_system: str
    document_planner_user: str
    section_planner_system: str
    section_planner_user: str
    writer_system: str
    writer_user: str
    critic_system: str
    critic_user: str


@dataclass(frozen=True)
class WorkflowConfig:
    mode: str
    max_revisions: int
    memory_summary_chars: int
    planner: PlannerConfig
    writer: WriterConfig
    critic: CriticConfig
    prompts: WorkflowPromptsConfig


@dataclass(frozen=True)
class ProfileConfig:
    doc_type: str
    fraud_type: str
    documents: int
    section_words: dict[str, int]


@dataclass(frozen=True)
class OffencePeriodConfig:
    start: str
    end: str


@dataclass(frozen=True)
class CaseMetadataConfig:
    court: str
    case_number: str
    cross_ref: str
    filing_date: str
    offence_period: OffencePeriodConfig


@dataclass(frozen=True)
class PersonSpecConfig:
    nationality: str
    title: str
    surface_forms: int


@dataclass(frozen=True)
class CaseCastConfig:
    defendants: list[PersonSpecConfig]
    collateral: list[PersonSpecConfig]
    charged_orgs: int
    associated_orgs: int


@dataclass(frozen=True)
class CountConfig:
    offence: str
    statute: str
    particulars: str


@dataclass(frozen=True)
class CaseConfig:
    metadata: CaseMetadataConfig
    cast: CaseCastConfig
    defendants: str | list[dict[str, Any]]
    collateral: str | list[dict[str, Any]]
    charged_orgs: str | list[dict[str, Any]]
    associated_orgs: str | list[dict[str, Any]]
    schema: str | dict[str, Any]
    prose: dict[str, str]
    counts: str | list[CountConfig]


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    ollama: OllamaConfig
    langfuse: LangfuseConfig
    generation: GenerationConfig
    workflow: WorkflowConfig
    profile: ProfileConfig
    case: CaseConfig
    nationality_locales: dict[str, str]
    vat_prefixes: dict[str, str]
    fraud_statutes: dict[str, list[CountConfig]]
