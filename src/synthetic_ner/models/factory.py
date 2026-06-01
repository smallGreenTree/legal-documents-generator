"""Model client factory for stage-level routing."""

from __future__ import annotations

from src.synthetic_ner.models.ollama_client import TracedOllamaClient
from src.synthetic_ner.tasks.document_generation.tracer import TraceStore
from src.synthetic_ner.types.app_config import (
    ModelProviderConfig,
    ModelRoutingConfig,
    OllamaConfig,
    OllamaRecoveryConfig,
)


def build_model_client(
    *,
    stage: str,
    routing: ModelRoutingConfig,
    tracer: TraceStore,
):
    provider_cfg = _stage_config(routing, stage)
    if provider_cfg.provider == "ollama":
        return TracedOllamaClient(
            config=ollama_config_from_provider(provider_cfg),
            tracer=tracer,
        )
    raise ValueError(f"Unsupported model provider: {provider_cfg.provider}")


def describe_stage_route(
    *,
    stage: str,
    routing: ModelRoutingConfig,
) -> str:
    provider_cfg = _stage_config(routing, stage)
    return f"{stage}={provider_cfg.provider}:{provider_cfg.model}"


def ollama_config_from_provider(provider_cfg: ModelProviderConfig) -> OllamaConfig:
    return OllamaConfig(
        base_url=provider_cfg.base_url,
        model=provider_cfg.model,
        timeout=provider_cfg.timeout,
        num_ctx=provider_cfg.num_ctx,
        think=provider_cfg.think,
        recovery=OllamaRecoveryConfig(
            max_generate_attempts=provider_cfg.max_generate_attempts,
            retry_backoff_seconds=provider_cfg.retry_backoff_seconds,
            controlled_empty_section=provider_cfg.controlled_empty_section,
        ),
    )


def _stage_config(routing: ModelRoutingConfig, stage: str) -> ModelProviderConfig:
    try:
        return routing.stages[stage]
    except KeyError as exc:
        raise ValueError(f"model_routing.stages.{stage} is required") from exc
