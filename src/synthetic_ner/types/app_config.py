from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str
    schema_dir: str
    memory_dir: str


@dataclass(frozen=True)
class OllamaRecoveryConfig:
    max_generate_attempts: int
    retry_backoff_seconds: float
    controlled_empty_section: str


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    timeout: int
    recovery: OllamaRecoveryConfig
    num_ctx: int | None = None
    think: bool | None = None


@dataclass(frozen=True)
class ModelProviderConfig:
    provider: str
    model: str
    timeout: int
    base_url: str | None = None
    api_key_env: str | None = None
    thinking_budget: int | None = None
    num_ctx: int | None = None
    think: bool | None = None
    max_generate_attempts: int = 1
    retry_backoff_seconds: float = 0.0
    min_interval_seconds: float = 0.0


@dataclass(frozen=True)
class ModelRoutingConfig:
    default: ModelProviderConfig
    stages: dict[str, ModelProviderConfig]


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool
    host: str
    public_key_env: str
    secret_key_env: str


@dataclass(frozen=True)
class GenerationConfig:
    words_per_page: int


@dataclass(frozen=True)
class PersonVariantGenerationConfig:
    enabled: bool
    nickname_variants: int
    misspelling_variants: int
    locale_aware: bool


@dataclass(frozen=True)
class PersonVariantEligibilityConfig:
    nickname: bool
    misspelling: bool


@dataclass(frozen=True)
class EntityVariantsConfig:
    persons: PersonVariantGenerationConfig


@dataclass(frozen=True)
class PlannerConfig:
    temperature: float
    document_max_output_tokens: int
    section_max_output_tokens: int


@dataclass(frozen=True)
class WriterConfig:
    chunk_words: int
    context_tail_chars: int
    temperature: float
    max_output_tokens: int
    min_output_tokens: int
    output_token_multiplier: float


@dataclass(frozen=True)
class CriticConfig:
    temperature: float
    max_output_tokens: int
    memory_char_limit: int
    section_text_char_limit: int
    rubrics: dict[str, str]


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
    variants: PersonVariantEligibilityConfig


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
    model_routing: ModelRoutingConfig
    langfuse: LangfuseConfig
    generation: GenerationConfig
    entity_variants: EntityVariantsConfig
    workflow: WorkflowConfig
    profile: ProfileConfig
    case: CaseConfig
    nationality_locales: dict[str, str]
    vat_prefixes: dict[str, str]
    fraud_statutes: dict[str, list[CountConfig]]
