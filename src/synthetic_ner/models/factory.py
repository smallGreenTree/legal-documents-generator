"""Model client factory for stage-level routing."""

from __future__ import annotations

from src.synthetic_ner.models.gemini_client import TracedGeminiClient
from src.synthetic_ner.models.ollama_client import TracedOllamaClient
from src.synthetic_ner.tasks.tracer import TraceStore
from src.synthetic_ner.types.app_config import (
    ModelProviderConfig,
    ModelRoutingConfig,
    OllamaConfig,
)


def build_model_client(
    *,
    stage: str,
    routing: ModelRoutingConfig,
    fallback_ollama: OllamaConfig,
    tracer: TraceStore,
):
    provider_cfg = routing.stages.get(stage, routing.default)
    if provider_cfg.provider == "ollama":
        return TracedOllamaClient(
            config=OllamaConfig(
                base_url=provider_cfg.base_url or fallback_ollama.base_url,
                model=provider_cfg.model,
                timeout=provider_cfg.timeout,
                num_ctx=provider_cfg.num_ctx,
                think=provider_cfg.think,
                recovery=fallback_ollama.recovery,
            ),
            tracer=tracer,
        )
    if provider_cfg.provider == "gemini":
        return TracedGeminiClient(config=provider_cfg, tracer=tracer)
    raise ValueError(f"Unsupported model provider: {provider_cfg.provider}")


def describe_stage_route(
    *,
    stage: str,
    routing: ModelRoutingConfig,
) -> str:
    provider_cfg: ModelProviderConfig = routing.stages.get(stage, routing.default)
    return f"{stage}={provider_cfg.provider}:{provider_cfg.model}"
