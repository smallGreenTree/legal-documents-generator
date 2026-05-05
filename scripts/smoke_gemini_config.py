"""Smoke-test Gemini routing from config to API call.

This script is intentionally small and temporary. It loads the same .env files and
config as the app, builds the routed critic client, and sends one tiny request.
It prints no API key and should be safe to paste into logs.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.config import load_app_config
from src.synthetic_ner.models.factory import build_model_client, describe_stage_route
from src.synthetic_ner.tasks.tracer import TraceStore


def main() -> None:
    project_root = PROJECT_ROOT
    load_env_files(project_root)
    app_config = load_app_config(
        project_root / "config.yaml",
        project_root / "config_case" / "case_1.yaml",
    )

    route = app_config.model_routing.stages.get(
        "critic",
        app_config.model_routing.default,
    )
    print(f"critic route: {describe_stage_route(stage='critic', routing=app_config.model_routing)}")
    print(f"base_url: {route.base_url}")
    print(f"api_key_env: {route.api_key_env}")

    tracer = TraceStore(replace(app_config.langfuse, enabled=False))
    client = build_model_client(
        stage="critic",
        routing=app_config.model_routing,
        fallback_ollama=app_config.ollama,
        tracer=tracer,
    )
    result = client.invoke(
        doc_id="smoke_gemini_config",
        task_id="critic_smoke_r0",
        stage="critic",
        system_prompt="Reply with exactly: OK",
        user_prompt="This is a smoke test. Reply with exactly: OK",
        temperature=0.0,
        max_output_tokens=200,
    )
    print(f"response: {result.text!r}")
    print(f"metadata: provider={result.metadata.get('provider')}, model={result.metadata.get('model')}, done_reason={result.metadata.get('done_reason')}")


if __name__ == "__main__":
    main()
