#!/usr/bin/env python
"""Smoke-test configured Ollama model routes.

This is intentionally cheaper than a document generation run. It verifies that
the configured models are present/reachable and that each selected stage can
return a short response with the same Ollama options used by the app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_ner.config import load_app_config


DEFAULT_STAGES = ("writer", "critic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test configured Ollama routes.")
    parser.add_argument("--config", default="config.yaml", help="Root config path.")
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case config path used to load the full app config.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("planner", "writer", "critic"),
        help="Stage to test. May be repeated. Defaults to writer and critic.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Return exactly this JSON object and nothing else: "
            "{\"ok\":true,\"stage\":\"smoke\"}"
        ),
        help="Tiny prompt used for each smoke generation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds for smoke calls.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=64,
        help="Ollama num_predict for smoke calls.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    config_path = _resolve_path(project_root, args.config)
    case_config_path = _resolve_path(project_root, args.case_config)
    app_config = load_app_config(config_path, case_config_path)
    stages = tuple(args.stage or DEFAULT_STAGES)

    failures: list[str] = []
    for stage in stages:
        provider = app_config.model_routing.stages[stage]
        if provider.provider != "ollama":
            failures.append(f"{stage}: unsupported provider {provider.provider!r}")
            continue
        result = _smoke_ollama_stage(
            stage=stage,
            base_url=provider.base_url,
            model=provider.model,
            prompt=args.prompt,
            num_ctx=provider.num_ctx,
            top_p=provider.top_p,
            think=provider.think,
            timeout=min(args.timeout, provider.timeout),
            max_output_tokens=args.max_output_tokens,
        )
        if result["ok"]:
            print(
                f"ok {stage}: model={provider.model} "
                f"tokens={result.get('tokens_response')} "
                f"done={result.get('done_reason') or 'n/a'}"
            )
        else:
            message = f"{stage}: {result['error']}"
            failures.append(message)
            print(f"FAIL {message}", file=sys.stderr)

    if failures:
        print("Smoke test failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Smoke test passed.")
    return 0


def _smoke_ollama_stage(
    *,
    stage: str,
    base_url: str,
    model: str,
    prompt: str,
    num_ctx: int | None,
    top_p: float | None,
    think: bool | None,
    timeout: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": 0.0 if stage in {"critic", "planner"} else 0.2,
        "num_predict": max_output_tokens,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if top_p is not None:
        options["top_p"] = top_p

    request_json: dict[str, Any] = {
        "model": model,
        "prompt": f"[SYSTEM]\nYou are a smoke-test model route.\n\n[USER]\n{prompt}\n",
        "stream": False,
        "options": options,
    }
    if think is not None:
        request_json["think"] = think

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=request_json,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}

    payload = response.json()
    text = str(payload.get("response") or "").strip()
    if not text:
        return {"ok": False, "error": "empty response"}
    return {
        "ok": True,
        "tokens_response": payload.get("eval_count"),
        "done_reason": payload.get("done_reason"),
        "response_preview": text[:160],
    }


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


if __name__ == "__main__":
    raise SystemExit(main())
