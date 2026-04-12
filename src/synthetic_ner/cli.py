"""CLI entrypoint for the generator."""

import argparse
import os
from pathlib import Path

from src.synthetic_ner.config import load_app_config, resolve_doc_types


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    app_config = load_app_config(project_root / "config.yaml")
    doc_type_choices = sorted(resolve_doc_types(app_config.generation))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents",
        "--count",
        dest="documents",
        type=int,
        default=None,
        help="Override profile.documents",
    )
    parser.add_argument(
        "--doc-type",
        choices=doc_type_choices,
        metavar="TYPE",
        help="Override profile.doc_type",
    )
    parser.add_argument(
        "--fraud-type",
        metavar="TYPE",
        help="Override profile.fraud_type",
    )
    parser.add_argument(
        "--from-schema",
        metavar="PATH",
        help="Load an existing case schema instead of auto-generating one",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=("classic", "langgraph"),
        default=None,
        help="Override workflow.mode",
    )
    return parser


def resolve_workflow_mode(project_root: Path, args: argparse.Namespace) -> str:
    if args.workflow_mode is not None:
        return args.workflow_mode

    app_config = load_app_config(project_root / "config.yaml")
    return app_config.workflow.mode


def load_env_files(project_root: Path) -> None:
    for env_name in (".env", ".env.langfuse"):
        _load_env_file(project_root / env_name)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def main(project_root: Path | None = None) -> None:
    resolved_project_root = project_root or Path(__file__).resolve().parents[2]
    load_env_files(resolved_project_root)
    args = build_parser(resolved_project_root).parse_args()
    workflow_mode = resolve_workflow_mode(resolved_project_root, args)

    if workflow_mode == "langgraph":
        from src.synthetic_ner.tasks.orchestrator import run_langgraph_workflow

        run_langgraph_workflow(args, resolved_project_root)
        return

    from src.synthetic_ner.engine import run_generation

    run_generation(args, resolved_project_root)
