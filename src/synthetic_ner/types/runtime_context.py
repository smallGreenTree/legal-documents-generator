from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment
from src.synthetic_ner.config import GenerationConfig


@dataclass
class RuntimeContext:
    cfg: dict
    generation_cfg: GenerationConfig
    profile: dict
    case_cfg: dict
    langfuse_cfg: dict
    ollama_cfg: dict
    workflow_cfg: dict
    nat_locales: dict
    vat_prefixes: dict
    doc_type: str
    fraud_type: str
    output_dir: Path
    schema_dir: Path
    memory_dir: Path
    trace_dir: Path
    template_env: Environment
    sections: dict
    labels: dict
    section_word_targets: dict[str, int]
    documents: int
    prose_overrides: dict[str, str]
    schema_source_path: Path | None
