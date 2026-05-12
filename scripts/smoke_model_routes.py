"""Smoke-test configured planner, writer, and critic model routes.

Loads the same config and env files as the app, builds each routed model client,
and sends one tiny request to each stage. This is intended for quick local
diagnostics after changing model routing, provider credentials, or local model
availability.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_ner.cli import load_env_files  # noqa: E402
from src.synthetic_ner.config import load_app_config  # noqa: E402
from src.synthetic_ner.models.factory import (  # noqa: E402
    build_model_client,
    describe_stage_route,
)
from src.synthetic_ner.tasks.tracer import TraceStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test planner, writer, and critic model routes."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="App config path relative to project root.",
    )
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case config path relative to project root.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Maximum response tokens per smoke call.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_env_files(PROJECT_ROOT)
    app_config = load_app_config(
        PROJECT_ROOT / args.config,
        PROJECT_ROOT / args.case_config,
    )
    tracer = TraceStore(replace(app_config.langfuse, enabled=False))

    print(f"config: {args.config}")
    print(f"case_config: {args.case_config}")
    for stage in ("planner", "writer", "critic"):
        route = app_config.model_routing.stages.get(
            stage,
            app_config.model_routing.default,
        )
        print(f"\n[{stage}] {describe_stage_route(stage=stage, routing=app_config.model_routing)}")
        print(f"base_url: {route.base_url or app_config.ollama.base_url}")

        client = build_model_client(
            stage=stage,
            routing=app_config.model_routing,
            fallback_ollama=app_config.ollama,
            tracer=tracer,
        )
        result = client.invoke(
            doc_id="smoke_model_routes",
            task_id=f"{stage}_smoke",
            stage=stage,
            system_prompt="Reply with exactly: OK",
            user_prompt="Smoke test. Reply with exactly: OK",
            temperature=0.0,
            max_output_tokens=args.max_output_tokens,
        )
        text = result.text.strip()
        print(f"response: {text!r}")
        print(
            "metadata: "
            f"provider={result.metadata.get('provider')}, "
            f"model={result.metadata.get('model')}, "
            f"done_reason={result.metadata.get('done_reason')}"
        )


if __name__ == "__main__":
    main()
