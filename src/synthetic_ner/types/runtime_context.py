from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment
from src.synthetic_ner.types.app_config import (
    AppConfig,
    CaseConfig,
    GenerationConfig,
    LangfuseConfig,
    ModelRoutingConfig,
    PathsConfig,
    ProfileConfig,
    WorkflowConfig,
)


@dataclass
class RuntimeContext:
    project_root: Path
    app_config: AppConfig
    paths: PathsConfig
    generation_cfg: GenerationConfig
    profile: ProfileConfig
    case_cfg: CaseConfig
    langfuse_cfg: LangfuseConfig
    model_routing_cfg: ModelRoutingConfig
    workflow_cfg: WorkflowConfig
    nat_locales: dict[str, str]
    vat_prefixes: dict[str, str]
    doc_type: str
    fraud_type: str
    output_dir: Path
    schema_dir: Path
    memory_dir: Path
    template_path: Path
    template_env: Environment
    template_name: str
    sections: dict
    labels: dict
    section_word_targets: dict[str, int]
    documents: int
    prose_overrides: dict[str, str]
    schema_source_path: Path | None
