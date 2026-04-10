"""Typed config parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_ner.types.app_config import (
    AppConfig,
    CaseCastConfig,
    CaseConfig,
    CaseMetadataConfig,
    CountConfig,
    CriticConfig,
    GenerationConfig,
    LangfuseConfig,
    OffencePeriodConfig,
    OllamaConfig,
    PathsConfig,
    PersonSpecConfig,
    PlannerConfig,
    ProfileConfig,
    WorkflowConfig,
    WorkflowPromptsConfig,
    WriterConfig,
)
from src.synthetic_ner.utils import load_config


def load_app_config(path: Path | str) -> AppConfig:
    raw = load_config(path)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must load into a top-level mapping")
    return build_app_config(raw)


def build_app_config(cfg: dict[str, Any]) -> AppConfig:
    return AppConfig(
        paths=_build_paths_config(_require_mapping(cfg["paths"], "paths")),
        ollama=_build_ollama_config(_require_mapping(cfg["ollama"], "ollama")),
        langfuse=_build_langfuse_config(_require_mapping(cfg["langfuse"], "langfuse")),
        generation=_build_generation_config(
            _require_mapping(cfg["generation"], "generation")
        ),
        workflow=_build_workflow_config(_require_mapping(cfg["workflow"], "workflow")),
        profile=_build_profile_config(_require_mapping(cfg["profile"], "profile")),
        case=_build_case_config(_require_mapping(cfg["case"], "case")),
        nationality_locales=_build_string_mapping(
            _require_mapping(cfg["nationality_locales"], "nationality_locales"),
            "nationality_locales",
        ),
        vat_prefixes=_build_string_mapping(
            _require_mapping(cfg["vat_prefixes"], "vat_prefixes"),
            "vat_prefixes",
        ),
        fraud_statutes=_build_statute_mapping(
            _require_mapping(cfg["fraud_statutes"], "fraud_statutes"),
            "fraud_statutes",
        ),
    )


def resolve_doc_types(generation_cfg: GenerationConfig) -> tuple[str, ...]:
    return tuple(generation_cfg.section_weights.keys())


def resolve_section_order(generation_cfg: GenerationConfig, doc_type: str) -> list[str]:
    section_weights = generation_cfg.section_weights.get(doc_type)
    if section_weights is None:
        available = ", ".join(sorted(generation_cfg.section_weights))
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Available configured doc types: {available}"
        )
    return list(section_weights.keys())


def _build_paths_config(raw: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        output_dir=_require_string(raw["output_dir"], "paths.output_dir"),
        schema_dir=_require_string(raw["schema_dir"], "paths.schema_dir"),
        memory_dir=_require_string(raw["memory_dir"], "paths.memory_dir"),
        trace_dir=_require_string(raw["trace_dir"], "paths.trace_dir"),
    )


def _build_ollama_config(raw: dict[str, Any]) -> OllamaConfig:
    return OllamaConfig(
        base_url=_require_string(raw["base_url"], "ollama.base_url"),
        model=_require_string(raw["model"], "ollama.model"),
        timeout=_require_positive_int(raw["timeout"], "ollama.timeout"),
    )


def _build_langfuse_config(raw: dict[str, Any]) -> LangfuseConfig:
    return LangfuseConfig(
        enabled=_require_bool(raw["enabled"], "langfuse.enabled"),
        base_url=_require_string(raw["base_url"], "langfuse.base_url"),
        public_key_env=_require_string(
            raw["public_key_env"],
            "langfuse.public_key_env",
        ),
        secret_key_env=_require_string(
            raw["secret_key_env"],
            "langfuse.secret_key_env",
        ),
    )


def _build_generation_config(raw: dict[str, Any]) -> GenerationConfig:
    words_per_page = _require_positive_int(
        raw["words_per_page"],
        "generation.words_per_page",
    )
    section_weights_raw = _require_mapping(
        raw["section_weights"],
        "generation.section_weights",
    )

    section_weights: dict[str, dict[str, float]] = {}
    for doc_type, section_map in section_weights_raw.items():
        section_path = f"generation.section_weights.{doc_type}"
        section_weights[doc_type] = _build_float_mapping(
            _require_mapping(section_map, section_path),
            section_path,
        )

    return GenerationConfig(
        words_per_page=words_per_page,
        section_weights=section_weights,
    )


def _build_workflow_config(raw: dict[str, Any]) -> WorkflowConfig:
    prompts = _require_mapping(raw["prompts"], "workflow.prompts")

    return WorkflowConfig(
        mode=_require_string(raw["mode"], "workflow.mode"),
        max_revisions=_require_non_negative_int(
            raw["max_revisions"],
            "workflow.max_revisions",
        ),
        memory_summary_chars=_require_positive_int(
            raw["memory_summary_chars"],
            "workflow.memory_summary_chars",
        ),
        planner=PlannerConfig(
            temperature=_require_number(
                _require_mapping(raw["planner"], "workflow.planner")["temperature"],
                "workflow.planner.temperature",
            )
        ),
        writer=WriterConfig(
            chunk_words=_require_positive_int(
                _require_mapping(raw["writer"], "workflow.writer")["chunk_words"],
                "workflow.writer.chunk_words",
            ),
            context_tail_chars=_require_positive_int(
                _require_mapping(raw["writer"], "workflow.writer")["context_tail_chars"],
                "workflow.writer.context_tail_chars",
            ),
            temperature=_require_number(
                _require_mapping(raw["writer"], "workflow.writer")["temperature"],
                "workflow.writer.temperature",
            ),
        ),
        critic=CriticConfig(
            temperature=_require_number(
                _require_mapping(raw["critic"], "workflow.critic")["temperature"],
                "workflow.critic.temperature",
            )
        ),
        prompts=WorkflowPromptsConfig(
            document_planner_system=_require_string(
                prompts["document_planner_system"],
                "workflow.prompts.document_planner_system",
            ),
            document_planner_user=_require_string(
                prompts["document_planner_user"],
                "workflow.prompts.document_planner_user",
            ),
            section_planner_system=_require_string(
                prompts["section_planner_system"],
                "workflow.prompts.section_planner_system",
            ),
            section_planner_user=_require_string(
                prompts["section_planner_user"],
                "workflow.prompts.section_planner_user",
            ),
            writer_system=_require_string(
                prompts["writer_system"],
                "workflow.prompts.writer_system",
            ),
            writer_user=_require_string(
                prompts["writer_user"],
                "workflow.prompts.writer_user",
            ),
            critic_system=_require_string(
                prompts["critic_system"],
                "workflow.prompts.critic_system",
            ),
            critic_user=_require_string(
                prompts["critic_user"],
                "workflow.prompts.critic_user",
            ),
        ),
    )


def _build_profile_config(raw: dict[str, Any]) -> ProfileConfig:
    section_words = _build_optional_int_mapping(
        raw["section_words"],
        "profile.section_words",
    )
    pages = _require_optional_positive_int(raw["pages"], "profile.pages")
    return ProfileConfig(
        doc_type=_require_string(raw["doc_type"], "profile.doc_type"),
        fraud_type=_require_string(raw["fraud_type"], "profile.fraud_type"),
        documents=_require_positive_int(raw["documents"], "profile.documents"),
        pages=pages,
        section_words=section_words,
    )


def _build_case_config(raw: dict[str, Any]) -> CaseConfig:
    metadata = _require_mapping(raw["metadata"], "case.metadata")
    cast = _require_mapping(raw["cast"], "case.cast")

    return CaseConfig(
        metadata=CaseMetadataConfig(
            court=_require_string(metadata["court"], "case.metadata.court", allow_auto=True),
            case_number=_require_string(
                metadata["case_number"],
                "case.metadata.case_number",
                allow_auto=True,
            ),
            cross_ref=_require_string(
                metadata["cross_ref"],
                "case.metadata.cross_ref",
                allow_auto=True,
            ),
            filing_date=_require_string(
                metadata["filing_date"],
                "case.metadata.filing_date",
                allow_auto=True,
            ),
            offence_period=OffencePeriodConfig(
                start=_require_string(
                    _require_mapping(
                        metadata["offence_period"],
                        "case.metadata.offence_period",
                    )["start"],
                    "case.metadata.offence_period.start",
                    allow_auto=True,
                ),
                end=_require_string(
                    _require_mapping(
                        metadata["offence_period"],
                        "case.metadata.offence_period",
                    )["end"],
                    "case.metadata.offence_period.end",
                    allow_auto=True,
                ),
            ),
        ),
        cast=CaseCastConfig(
            defendants=_build_person_specs(
                _require_list(cast["defendants"], "case.cast.defendants"),
                "case.cast.defendants",
            ),
            collateral=_build_person_specs(
                _require_list(cast["collateral"], "case.cast.collateral"),
                "case.cast.collateral",
            ),
            charged_orgs=_require_non_negative_int(
                cast["charged_orgs"],
                "case.cast.charged_orgs",
            ),
            associated_orgs=_require_non_negative_int(
                cast["associated_orgs"],
                "case.cast.associated_orgs",
            ),
        ),
        defendants=_build_auto_or_list(raw["defendants"], "case.defendants"),
        collateral=_build_auto_or_list(raw["collateral"], "case.collateral"),
        charged_orgs=_build_auto_or_list(raw["charged_orgs"], "case.charged_orgs"),
        associated_orgs=_build_auto_or_list(
            raw["associated_orgs"],
            "case.associated_orgs",
        ),
        schema=_build_auto_or_mapping(raw["schema"], "case.schema"),
        prose=_build_string_mapping(
            _require_mapping(raw["prose"], "case.prose"),
            "case.prose",
            allow_auto=True,
        ),
        counts=_build_auto_or_statute_list(raw["counts"], "case.counts"),
    )


def _build_person_specs(raw: list[Any], path: str) -> list[PersonSpecConfig]:
    specs = []
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        mapping = _require_mapping(item, item_path)
        specs.append(
            PersonSpecConfig(
                nationality=_require_string(
                    mapping["nationality"],
                    f"{item_path}.nationality",
                ),
                title=_require_string(
                    mapping["title"],
                    f"{item_path}.title",
                    allow_empty=True,
                ),
                surface_forms=_require_positive_int(
                    mapping["surface_forms"],
                    f"{item_path}.surface_forms",
                ),
            )
        )
    return specs


def _build_statute_mapping(
    raw: dict[str, Any],
    path: str,
) -> dict[str, list[CountConfig]]:
    statutes: dict[str, list[CountConfig]] = {}
    for fraud_type, items in raw.items():
        statutes[fraud_type] = _build_statute_list(
            _require_list(items, f"{path}.{fraud_type}"),
            f"{path}.{fraud_type}",
        )
    return statutes


def _build_auto_or_statute_list(value: Any, path: str) -> str | list[CountConfig]:
    if value == "auto":
        return "auto"
    return _build_statute_list(_require_list(value, path), path)


def _build_statute_list(raw: list[Any], path: str) -> list[CountConfig]:
    statutes = []
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        mapping = _require_mapping(item, item_path)
        statutes.append(
            CountConfig(
                offence=_require_string(mapping["offence"], f"{item_path}.offence"),
                statute=_require_string(mapping["statute"], f"{item_path}.statute"),
                particulars=_require_string(
                    mapping["particulars"],
                    f"{item_path}.particulars",
                ),
            )
        )
    return statutes


def _build_optional_int_mapping(
    value: Any,
    path: str,
) -> dict[str, int] | None:
    if value is None:
        return None
    raw = _require_mapping(value, path)
    return {
        key: _require_positive_int(item, f"{path}.{key}")
        for key, item in raw.items()
    }


def _build_float_mapping(raw: dict[str, Any], path: str) -> dict[str, float]:
    return {
        key: _require_positive_number(value, f"{path}.{key}")
        for key, value in raw.items()
    }


def _build_string_mapping(
    raw: dict[str, Any],
    path: str,
    *,
    allow_auto: bool = False,
) -> dict[str, str]:
    return {
        key: _require_string(value, f"{path}.{key}", allow_auto=allow_auto)
        for key, value in raw.items()
    }


def _build_auto_or_list(value: Any, path: str) -> str | list[dict[str, Any]]:
    if value == "auto":
        return "auto"
    raw = _require_list(value, path)
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{path} must be 'auto' or a list of mappings")
    return raw


def _build_auto_or_mapping(value: Any, path: str) -> str | dict[str, Any]:
    if value == "auto":
        return "auto"
    return _require_mapping(value, path)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_string(
    value: Any,
    path: str,
    *,
    allow_empty: bool = False,
    allow_auto: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if allow_auto and value == "auto":
        return value
    if allow_empty or value.strip():
        return value
    raise ValueError(f"{path} must be a non-empty string")


def _require_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _require_optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, path)


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _require_positive_number(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be a positive number")
    return number
