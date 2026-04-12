"""Sync workflow prompts to Langfuse Prompt Management."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from langfuse import Langfuse
from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.config import load_app_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upsert workflow prompts into Langfuse Prompt Management.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root that contains config.yaml and .env.langfuse (default: .)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Relative path to app config from project root (default: config.yaml)",
    )
    parser.add_argument(
        "--label",
        default="production",
        help="Prompt label to target in Langfuse (default: production)",
    )
    parser.add_argument(
        "--name-prefix",
        default="synthetic_ner",
        help="Prompt name prefix in Langfuse (default: synthetic_ner)",
    )
    parser.add_argument(
        "--commit-message",
        default="Sync prompts from workflow prompt config",
        help="Commit message for new prompt versions in Langfuse",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config

    load_env_files(project_root)
    app_config = load_app_config(config_path)
    langfuse_cfg = app_config.langfuse
    if not langfuse_cfg.enabled:
        raise SystemExit("Langfuse is disabled in config.")

    public_key = os.getenv(langfuse_cfg.public_key_env)
    secret_key = os.getenv(langfuse_cfg.secret_key_env)
    if not public_key or not secret_key:
        raise SystemExit(
            "Langfuse credentials are missing. "
            "Expected env vars "
            f"'{langfuse_cfg.public_key_env}' and '{langfuse_cfg.secret_key_env}'."
        )

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=langfuse_cfg.host,
    )

    prompt_templates = asdict(app_config.workflow.prompts)
    created = 0
    updated = 0
    unchanged = 0
    errors = 0

    for key, template in prompt_templates.items():
        name = f"{args.name_prefix}.{key}" if args.name_prefix else key
        existing_prompt = None
        try:
            existing_prompt = client.get_prompt(
                name=name,
                label=args.label,
                type="text",
                cache_ttl_seconds=0,
                fetch_timeout_seconds=5,
                max_retries=1,
            )
        except Exception:
            existing_prompt = None

        if existing_prompt is not None and getattr(existing_prompt, "prompt", None) == template:
            unchanged += 1
            print(f"[unchanged] {name} label={args.label}")
            continue

        try:
            result = client.create_prompt(
                name=name,
                prompt=template,
                labels=[args.label],
                type="text",
                commit_message=args.commit_message,
            )
            version = getattr(result, "version", "?")
            if existing_prompt is None:
                created += 1
                print(f"[created] {name} v{version} label={args.label}")
            else:
                updated += 1
                print(f"[updated] {name} v{version} label={args.label}")
        except Exception as exc:
            errors += 1
            print(f"[error] {name}: {exc}")

    print(
        "Summary: "
        f"created={created}, updated={updated}, unchanged={unchanged}, errors={errors}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
