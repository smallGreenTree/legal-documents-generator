from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str
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
    top_p: float | None = None


@dataclass(frozen=True)
class ModelProviderConfig:
    provider: str
    model: str
    timeout: int
    base_url: str
    num_ctx: int | None = None
    think: bool | None = None
    top_p: float | None = None
    max_generate_attempts: int = 1
    retry_backoff_seconds: float = 0.0
    controlled_empty_section: str = "[section not generated]"


@dataclass(frozen=True)
class ModelRoutingConfig:
    stages: dict[str, ModelProviderConfig]


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    service_name: str
    pipeline_stage: str
    trace_name: str
    prompt_name_prefix: str
    prompt_alias: str


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
class WriterConfig:
    active: bool
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True)
class PolisherConfig:
    active: bool
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True)
class CriticConfig:
    active: bool
    acceptance_threshold: float
    temperature: float
    max_output_tokens: int
    memory_char_limit: int
    section_text_char_limit: int
    rubrics: dict[str, str]


@dataclass(frozen=True)
class WorkflowPromptsConfig:
    writer_system: str
    writer_user: str
    polisher_system: str
    polisher_user: str
    critic_system: str
    critic_user: str


@dataclass(frozen=True)
class WorkflowConfig:
    mode: str
    max_revisions: int
    memory_summary_chars: int
    validators: dict[str, bool]
    writer: WriterConfig
    polisher: PolisherConfig
    critic: CriticConfig
    prompts: WorkflowPromptsConfig


@dataclass(frozen=True)
class ProfileConfig:
    doc_type: str
    fraud_type: str
    documents: int
    sections: list[str]


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
    role: str = ""


@dataclass(frozen=True)
class OrganisationSpecConfig:
    group: str
    country: str
    role: str = ""


@dataclass(frozen=True)
class CaseCastConfig:
    defendants: list[PersonSpecConfig]
    collateral: list[PersonSpecConfig]
    charged_orgs: int
    associated_orgs: int
    organisation_specs: list[OrganisationSpecConfig] = field(default_factory=list)
    address_surface_forms: int = 3


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
    evidence_categories: list[str]
    prose: dict[str, str]
    counts: str | list[CountConfig]
    scenario_brief: dict[str, Any]


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    model_routing: ModelRoutingConfig
    mlflow: MlflowConfig
    generation: GenerationConfig
    entity_variants: EntityVariantsConfig
    workflow: WorkflowConfig
    profile: ProfileConfig
    case: CaseConfig
    nationality_locales: dict[str, str]
    vat_prefixes: dict[str, str]
    fraud_statutes: dict[str, list[CountConfig]]
