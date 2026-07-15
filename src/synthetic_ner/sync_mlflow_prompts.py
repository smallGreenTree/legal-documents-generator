"""Synchronize workflow prompts to the MLflow Prompt Registry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mlflow

from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.config import load_app_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upsert workflow prompts into the MLflow Prompt Registry.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--case-config", default="config_case/case_1.yaml")
    parser.add_argument("--alias", default=None)
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument(
        "--commit-message",
        default="Sync prompts from workflow prompt config",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    load_env_files(project_root)
    app_config = load_app_config(
        project_root / args.config,
        project_root / args.case_config,
    )
    cfg = app_config.mlflow
    if not cfg.enabled:
        raise SystemExit("MLflow is disabled in config.")

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    alias = args.alias or cfg.prompt_alias
    name_prefix = args.name_prefix or cfg.prompt_name_prefix
    templates = {
        key: value
        for key, value in asdict(app_config.workflow.prompts).items()
        if isinstance(value, str) and value.strip()
    }
    created = updated = unchanged = errors = 0

    for key, template in templates.items():
        name = f"{name_prefix}.{key}" if name_prefix else key
        existing = None
        try:
            existing = mlflow.genai.load_prompt(
                f"prompts:/{name}@{alias}",
                cache_ttl_seconds=0,
                link_to_model=False,
            )
        except Exception:
            pass

        if existing is not None and getattr(existing, "template", None) == template:
            unchanged += 1
            print(f"[unchanged] {name} alias={alias}")
            continue

        try:
            prompt = mlflow.genai.register_prompt(
                name=name,
                template=template,
                commit_message=args.commit_message,
                tags={"source": "synthetic-ner"},
            )
            mlflow.genai.set_prompt_alias(name, alias, prompt.version)
            if existing is None:
                created += 1
                action = "created"
            else:
                updated += 1
                action = "updated"
            print(f"[{action}] {name} v{prompt.version} alias={alias}")
        except Exception as exc:
            errors += 1
            print(f"[error] {name}: {exc}")

    print(f"Summary: created={created}, updated={updated}, unchanged={unchanged}, errors={errors}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
