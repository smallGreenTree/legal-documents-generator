"""Typed config parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_ner.constants import PROSE_SECTION_ORDER
from src.synthetic_ner.types.app_config import (
    AppConfig,
    CaseCastConfig,
    CaseConfig,
    CaseMetadataConfig,
    CountConfig,
    CriticConfig,
    EntityVariantsConfig,
    GenerationConfig,
    LangfuseConfig,
    ModelProviderConfig,
    ModelRoutingConfig,
    OffencePeriodConfig,
    OrganisationSpecConfig,
    PathsConfig,
    PersonSpecConfig,
    PersonVariantEligibilityConfig,
    PersonVariantGenerationConfig,
    PlannerConfig,
    ProfileConfig,
    WorkflowConfig,
    WorkflowPromptsConfig,
    WriterConfig,
)
from src.synthetic_ner.utils import load_config, resolve_project_path

DEFAULT_WORKFLOW_VALIDATORS = {
    "empty_section": True,
    "placeholder_text": True,
    "hidden_reasoning_markup": True,
    "placeholder_markers": True,
    "review_metadata": True,
    "meta_summary_style": True,
    "markdown_formatting": True,
    "incomplete_date_range": True,
    "dangling_between_phrase": True,
    "unresolved_timeline_placeholder": True,
    "partial_vat_identifier": True,
    "repeated_long_sentences": True,
    "repeated_sentence_fragments": True,
    "truncated_sentence": True,
    "minimum_length": True,
    "required_person_facts": True,
    "required_company_facts": True,
    "known_entity_presence": True,
    "unknown_case_references": True,
    "unknown_dates": True,
    "unknown_amounts": True,
    "unknown_vat_numbers": True,
    "unknown_organisations": True,
    "unknown_titled_people": True,
    "unknown_initials": True,
    "facts_contract": True,
}


def load_app_config(
    path: Path | str,
    case_config_path: Path | str | None = None,
) -> AppConfig:
    config_path = Path(path)
    raw = load_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must load into a top-level mapping")
    resolved_case_config_path = (
        Path(case_config_path)
        if case_config_path is not None
        else config_path.parent / "config_case" / "case_1.yaml"
    )
    case_raw = load_config(resolved_case_config_path)
    if not isinstance(case_raw, dict):
        raise ValueError(
            f"{resolved_case_config_path} must load into a top-level mapping"
        )
    return build_app_config(
        raw,
        case_cfg=case_raw,
        config_path=config_path,
    )


def build_app_config(
    cfg: dict[str, Any],
    *,
    case_cfg: dict[str, Any],
    config_path: Path | None = None,
) -> AppConfig:
    profile_cfg = _require_mapping(case_cfg["profile"], "profile")
    scenario_cfg = _require_mapping(case_cfg.get("scenario", {}), "scenario")
    return AppConfig(
        paths=_build_paths_config(_require_mapping(cfg["paths"], "paths")),
        model_routing=_build_model_routing_config(
            _require_mapping(cfg["model_routing"], "model_routing"),
        ),
        langfuse=_build_langfuse_config(_require_mapping(cfg["langfuse"], "langfuse")),
        generation=_build_generation_config(
            _require_mapping(cfg["generation"], "generation")
        ),
        entity_variants=_build_entity_variants_config(
            _require_mapping(cfg["entity_variants"], "entity_variants")
        ),
        workflow=_build_workflow_config(
            _require_mapping(cfg["workflow"], "workflow"),
            config_path=config_path,
        ),
        profile=_build_profile_config(
            profile_cfg,
            scenario_id=_optional_scenario_id(scenario_cfg),
        ),
        case=_build_case_config(_require_mapping(case_cfg["case"], "case")),
        nationality_locales=_build_string_mapping(
            _require_mapping(case_cfg["nationality_locales"], "nationality_locales"),
            "nationality_locales",
        ),
        vat_prefixes=_build_string_mapping(
            _require_mapping(case_cfg["vat_prefixes"], "vat_prefixes"),
            "vat_prefixes",
        ),
        fraud_statutes=_build_case_statute_mapping(case_cfg),
    )


def resolve_section_order(doc_type: str) -> list[str]:
    section_order = PROSE_SECTION_ORDER.get(doc_type)
    if section_order is None:
        available = ", ".join(sorted(PROSE_SECTION_ORDER))
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Available configured doc types: {available}"
        )
    return list(section_order)


def _build_paths_config(raw: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        output_dir=_require_string(raw["output_dir"], "paths.output_dir"),
        schema_dir=_require_string(raw["schema_dir"], "paths.schema_dir"),
        memory_dir=_require_string(raw["memory_dir"], "paths.memory_dir"),
    )


def _build_model_routing_config(raw: dict[str, Any]) -> ModelRoutingConfig:
    stages_raw = _require_mapping(raw["stages"], "model_routing.stages")
    required_stages = ("planner", "writer", "critic")
    missing_stages = [
        stage_name for stage_name in required_stages if stage_name not in stages_raw
    ]
    if missing_stages:
        raise ValueError(
            "model_routing.stages must explicitly configure: "
            + ", ".join(missing_stages)
        )
    stages = {
        stage_name: _build_model_provider_config(
            _require_mapping(stage_raw, f"model_routing.stages.{stage_name}"),
            f"model_routing.stages.{stage_name}",
        )
        for stage_name, stage_raw in stages_raw.items()
    }
    return ModelRoutingConfig(stages=stages)


def _build_model_provider_config(
    raw: dict[str, Any],
    path: str,
) -> ModelProviderConfig:
    provider = _require_string(raw["provider"], f"{path}.provider")
    model = _require_string(raw["model"], f"{path}.model")
    timeout = _require_positive_int(raw["timeout"], f"{path}.timeout")
    base_url = _require_string(raw["base_url"], f"{path}.base_url")
    num_ctx_value = raw.get("num_ctx")
    num_ctx = (
        _require_positive_int(num_ctx_value, f"{path}.num_ctx")
        if num_ctx_value is not None
        else None
    )
    think_value = raw.get("think")
    think = (
        _require_bool(think_value, f"{path}.think")
        if think_value is not None
        else None
    )
    top_p_value = raw.get("top_p")
    top_p = (
        _require_ratio(top_p_value, f"{path}.top_p")
        if top_p_value is not None
        else None
    )
    recovery = _require_mapping(raw["recovery"], f"{path}.recovery")
    max_generate_attempts = _require_positive_int(
        recovery["max_generate_attempts"],
        f"{path}.recovery.max_generate_attempts",
    )
    retry_backoff_seconds = _require_positive_number(
        recovery["retry_backoff_seconds"],
        f"{path}.recovery.retry_backoff_seconds",
    )
    controlled_empty_section = _require_string(
        recovery["controlled_empty_section"],
        f"{path}.recovery.controlled_empty_section",
    )
    if provider != "ollama":
        raise ValueError(f"{path}.provider must be ollama")
    return ModelProviderConfig(
        provider=provider,
        model=model,
        timeout=timeout,
        base_url=base_url,
        num_ctx=num_ctx,
        think=think,
        top_p=top_p,
        max_generate_attempts=max_generate_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        controlled_empty_section=controlled_empty_section,
    )


def _build_langfuse_config(raw: dict[str, Any]) -> LangfuseConfig:
    return LangfuseConfig(
        enabled=_require_bool(raw["enabled"], "langfuse.enabled"),
        host=_require_string(raw["host"], "langfuse.host"),
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
    return GenerationConfig(words_per_page=words_per_page)


def _build_entity_variants_config(raw: dict[str, Any]) -> EntityVariantsConfig:
    persons = _require_mapping(raw["persons"], "entity_variants.persons")
    generation = _require_mapping(
        persons["generation"],
        "entity_variants.persons.generation",
    )
    enabled = _require_bool(persons["enabled"], "entity_variants.persons.enabled")
    nickname_variants = _require_non_negative_int(
        generation["nickname_variants"],
        "entity_variants.persons.generation.nickname_variants",
    )
    misspelling_variants = _require_non_negative_int(
        generation["misspelling_variants"],
        "entity_variants.persons.generation.misspelling_variants",
    )
    if not enabled:
        nickname_variants = 0
        misspelling_variants = 0
    return EntityVariantsConfig(
        persons=PersonVariantGenerationConfig(
            enabled=enabled,
            nickname_variants=nickname_variants,
            misspelling_variants=misspelling_variants,
            locale_aware=_require_bool(
                generation["locale_aware"],
                "entity_variants.persons.generation.locale_aware",
            ),
        )
    )


def _build_workflow_config(
    raw: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> WorkflowConfig:
    prompts = _resolve_workflow_prompts(raw, config_path=config_path)
    planner_cfg = _build_planner_config(
        _require_mapping(raw["planner"], "workflow.planner")
    )
    critic_cfg = _build_critic_config(
        _require_mapping(raw["critic"], "workflow.critic")
    )
    writer = _require_mapping(raw["writer"], "workflow.writer")
    writer_max_output_tokens = _require_positive_int(
        writer["max_output_tokens"],
        "workflow.writer.max_output_tokens",
    )
    writer_min_output_tokens = _require_positive_int(
        writer["min_output_tokens"],
        "workflow.writer.min_output_tokens",
    )
    if writer_min_output_tokens > writer_max_output_tokens:
        raise ValueError(
            "workflow.writer.min_output_tokens must be less than or equal to "
            "workflow.writer.max_output_tokens"
        )

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
        validators=_build_validator_config(raw.get("validators", {})),
        planner=planner_cfg,
        writer=WriterConfig(
            active=_require_bool(
                writer.get("active", True),
                "workflow.writer.active",
            ),
            chunk_words=_require_positive_int(
                writer["chunk_words"],
                "workflow.writer.chunk_words",
            ),
            context_tail_chars=_require_positive_int(
                writer["context_tail_chars"],
                "workflow.writer.context_tail_chars",
            ),
            temperature=_require_number(
                writer["temperature"],
                "workflow.writer.temperature",
            ),
            max_output_tokens=writer_max_output_tokens,
            min_output_tokens=writer_min_output_tokens,
            output_token_multiplier=_require_positive_number(
                writer["output_token_multiplier"],
                "workflow.writer.output_token_multiplier",
            ),
            min_completion_ratio=_require_ratio(
                writer["min_completion_ratio"],
                "workflow.writer.min_completion_ratio",
            ),
        ),
        critic=critic_cfg,
        prompts=_build_workflow_prompts_config(
            prompts,
            planner_active=planner_cfg.active,
        ),
    )


def _build_workflow_prompts_config(
    prompts: dict[str, Any],
    *,
    planner_active: bool,
) -> WorkflowPromptsConfig:
    planner_path = "workflow.prompts"
    if planner_active:
        document_planner_system = _require_prompt(
            prompts,
            "document_planner_system",
            planner_path,
        )
        document_planner_user = _require_prompt(
            prompts,
            "document_planner_user",
            planner_path,
        )
        section_planner_system = _require_prompt(
            prompts,
            "section_planner_system",
            planner_path,
        )
        section_planner_user = _require_prompt(
            prompts,
            "section_planner_user",
            planner_path,
        )
    else:
        document_planner_system = _optional_prompt_string(
            prompts.get("document_planner_system"),
            f"{planner_path}.document_planner_system",
        )
        document_planner_user = _optional_prompt_string(
            prompts.get("document_planner_user"),
            f"{planner_path}.document_planner_user",
        )
        section_planner_system = _optional_prompt_string(
            prompts.get("section_planner_system"),
            f"{planner_path}.section_planner_system",
        )
        section_planner_user = _optional_prompt_string(
            prompts.get("section_planner_user"),
            f"{planner_path}.section_planner_user",
        )

    return WorkflowPromptsConfig(
        writer_system=_require_prompt(
            prompts,
            "writer_system",
            planner_path,
        ),
        writer_user=_require_prompt(
            prompts,
            "writer_user",
            planner_path,
        ),
        polisher_system=_require_prompt(
            prompts,
            "polisher_system",
            planner_path,
        ),
        polisher_user=_require_prompt(
            prompts,
            "polisher_user",
            planner_path,
        ),
        critic_system=_require_prompt(
            prompts,
            "critic_system",
            planner_path,
        ),
        critic_user=_require_prompt(
            prompts,
            "critic_user",
            planner_path,
        ),
        document_planner_system=document_planner_system,
        document_planner_user=document_planner_user,
        section_planner_system=section_planner_system,
        section_planner_user=section_planner_user,
    )


def _build_validator_config(raw: Any) -> dict[str, bool]:
    validators = dict(DEFAULT_WORKFLOW_VALIDATORS)
    if raw is None:
        return validators
    configured = _require_mapping(raw, "workflow.validators")
    unknown = sorted(set(configured) - set(DEFAULT_WORKFLOW_VALIDATORS))
    if unknown:
        raise ValueError(
            "Unknown workflow.validators keys: "
            f"{', '.join(unknown)}. Available validators: "
            f"{', '.join(sorted(DEFAULT_WORKFLOW_VALIDATORS))}"
        )
    for key, value in configured.items():
        validators[key] = _require_bool(value, f"workflow.validators.{key}")
    return validators


def _resolve_workflow_prompts(
    raw: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    prompts_path = raw.get("prompts_config_path")
    if prompts_path is None:
        return _require_mapping(raw["prompts"], "workflow.prompts")

    if config_path is None:
        raise ValueError(
            "workflow.prompts_config_path requires load_app_config to be called with a file path"
        )

    prompts_config_path = resolve_project_path(
        config_path.resolve().parent,
        _require_string(prompts_path, "workflow.prompts_config_path"),
    )
    prompts_raw = load_config(prompts_config_path)
    if not isinstance(prompts_raw, dict):
        raise ValueError(
            f"{prompts_config_path} must contain a top-level mapping"
        )
    if "prompts" in prompts_raw:
        return _require_mapping(
            prompts_raw["prompts"],
            f"{prompts_config_path}.prompts",
        )
    return prompts_raw


def _build_planner_config(raw: dict[str, Any]) -> PlannerConfig:
    max_output_tokens = _require_mapping(
        raw["max_output_tokens"],
        "workflow.planner.max_output_tokens",
    )
    return PlannerConfig(
        active=_require_bool(
            raw.get("active", True),
            "workflow.planner.active",
        ),
        temperature=_require_number(
            raw["temperature"],
            "workflow.planner.temperature",
        ),
        document_max_output_tokens=_require_positive_int(
            max_output_tokens["document"],
            "workflow.planner.max_output_tokens.document",
        ),
        section_max_output_tokens=_require_positive_int(
            max_output_tokens["section"],
            "workflow.planner.max_output_tokens.section",
        ),
    )


def _build_critic_config(raw: dict[str, Any]) -> CriticConfig:
    return CriticConfig(
        active=_require_bool(
            raw.get("active", True),
            "workflow.critic.active",
        ),
        acceptance_threshold=_require_ratio_score(
            raw.get("acceptance_threshold", 3.5),
            "workflow.critic.acceptance_threshold",
        ),
        temperature=_require_number(
            raw["temperature"],
            "workflow.critic.temperature",
        ),
        max_output_tokens=_require_positive_int(
            raw["max_output_tokens"],
            "workflow.critic.max_output_tokens",
        ),
        memory_char_limit=_require_positive_int(
            raw["memory_char_limit"],
            "workflow.critic.memory_char_limit",
        ),
        section_text_char_limit=_require_positive_int(
            raw["section_text_char_limit"],
            "workflow.critic.section_text_char_limit",
        ),
        rubrics=_build_string_mapping(
            _require_mapping(raw["rubrics"], "workflow.critic.rubrics"),
            "workflow.critic.rubrics",
        ),
    )


def _build_profile_config(raw: dict[str, Any], *, scenario_id: str = "") -> ProfileConfig:
    section_words = _build_required_int_mapping(
        raw["section_words"],
        "profile.section_words",
    )
    profile_fraud_type = str(raw.get("fraud_type") or "").strip()
    if scenario_id and profile_fraud_type and scenario_id != profile_fraud_type:
        raise ValueError(
            "profile.fraud_type duplicates scenario.id and the values differ; "
            "remove profile.fraud_type or make it match scenario.id"
        )
    fraud_type = scenario_id or profile_fraud_type
    if not fraud_type:
        raise ValueError("scenario.id is required when profile.fraud_type is omitted")
    return ProfileConfig(
        doc_type=_require_string(raw["doc_type"], "profile.doc_type"),
        fraud_type=fraud_type,
        documents=_require_positive_int(raw["documents"], "profile.documents"),
        section_words=section_words,
    )


def _optional_scenario_id(raw: dict[str, Any]) -> str:
    if "id" not in raw:
        return ""
    return _require_string(raw["id"], "scenario.id")


def _build_case_config(raw: dict[str, Any]) -> CaseConfig:
    metadata = _require_mapping(raw["metadata"], "case.metadata")
    cast = _require_mapping(raw["cast"], "case.cast")

    return CaseConfig(
        metadata=CaseMetadataConfig(
            court=_require_string(metadata["court"], "case.metadata.court"),
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
            organisation_specs=_build_organisation_specs(
                cast.get("organisation_specs", []),
                "case.cast.organisation_specs",
            ),
            address_surface_forms=_require_positive_int(
                cast.get("address_surface_forms", 3),
                "case.cast.address_surface_forms",
            ),
        ),
        defendants=_build_auto_or_list(raw.get("defendants", "auto"), "case.defendants"),
        collateral=_build_auto_or_list(raw.get("collateral", "auto"), "case.collateral"),
        charged_orgs=_build_auto_or_list(
            raw.get("charged_orgs", "auto"),
            "case.charged_orgs",
        ),
        associated_orgs=_build_auto_or_list(
            raw.get("associated_orgs", "auto"),
            "case.associated_orgs",
        ),
        schema=_build_auto_or_mapping(raw.get("schema", "auto"), "case.schema"),
        evidence_categories=_build_optional_string_list(
            raw.get("evidence_categories", []),
            "case.evidence_categories",
        ),
        prose=_build_string_mapping(
            _require_mapping(raw.get("prose", {}), "case.prose"),
            "case.prose",
            allow_auto=True,
        ),
        counts=_build_auto_or_statute_list(raw.get("counts", "auto"), "case.counts"),
        scenario_brief=_build_optional_mapping(
            raw.get("scenario_brief", {}),
            "case.scenario_brief",
        ),
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
                variants=_build_person_variant_eligibility(
                    mapping.get("variants"),
                    f"{item_path}.variants",
                ),
                role=_require_string(
                    mapping.get("role", ""),
                    f"{item_path}.role",
                    allow_empty=True,
                ),
            )
        )
    return specs


def _build_organisation_specs(raw: Any, path: str) -> list[OrganisationSpecConfig]:
    specs = []
    for index, item in enumerate(_require_list(raw, path)):
        item_path = f"{path}[{index}]"
        mapping = _require_mapping(item, item_path)
        specs.append(
            OrganisationSpecConfig(
                group=_require_string(mapping.get("group", "charged"), f"{item_path}.group"),
                country=_require_string(mapping["country"], f"{item_path}.country"),
                role=_require_string(
                    mapping.get("role", ""),
                    f"{item_path}.role",
                    allow_empty=True,
                ),
            )
        )
    return specs


def _build_person_variant_eligibility(
    raw: Any,
    path: str,
) -> PersonVariantEligibilityConfig:
    if raw is None:
        return PersonVariantEligibilityConfig(nickname=True, misspelling=True)

    mapping = _require_mapping(raw, path)
    nickname = mapping["nickname"] if "nickname" in mapping else True
    misspelling = mapping["misspelling"] if "misspelling" in mapping else True
    return PersonVariantEligibilityConfig(
        nickname=_require_bool(nickname, f"{path}.nickname"),
        misspelling=_require_bool(misspelling, f"{path}.misspelling"),
    )


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


def _build_case_statute_mapping(case_cfg: dict[str, Any]) -> dict[str, list[CountConfig]]:
    if "fraud_statutes" in case_cfg:
        return _build_statute_mapping(
            _require_mapping(case_cfg["fraud_statutes"], "fraud_statutes"),
            "fraud_statutes",
        )

    scenario = _require_mapping(case_cfg.get("scenario", {}), "scenario")
    if "counts" not in scenario:
        return {}
    scenario_id = _require_string(scenario.get("id"), "scenario.id")
    return {
        scenario_id: _build_statute_list(
            _require_list(scenario.get("counts"), "scenario.counts"),
            "scenario.counts",
        )
    }


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


def _build_required_int_mapping(
    value: Any,
    path: str,
) -> dict[str, int]:
    raw = _require_mapping(value, path)
    return {
        key: _require_positive_int(item, f"{path}.{key}")
        for key, item in raw.items()
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


def _build_optional_string_list(value: Any, path: str) -> list[str]:
    return [
        _require_string(item, f"{path}[{index}]")
        for index, item in enumerate(_require_list(value, path))
    ]


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


def _build_optional_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
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


def _require_prompt(prompts: dict[str, Any], key: str, path: str) -> str:
    if key not in prompts:
        raise ValueError(f"{path}.{key} is required")
    return _require_string(prompts[key], f"{path}.{key}")


def _optional_prompt_string(value: Any, path: str) -> str:
    if value is None:
        return ""
    return _require_string(value, path, allow_empty=True)


def _require_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value




def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _require_positive_number(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be a positive number")
    return number


def _require_ratio(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number <= 0 or number > 1:
        raise ValueError(f"{path} must be greater than 0 and less than or equal to 1")
    return number


def _require_ratio_score(value: Any, path: str) -> float:
    number = _require_number(value, path)
    if number < 1 or number > 5:
        raise ValueError(f"{path} must be between 1 and 5")
    return number
