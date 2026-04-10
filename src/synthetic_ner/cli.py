"""CLI entrypoint for the generator."""

import argparse
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


def main(project_root: Path | None = None) -> None:
    resolved_project_root = project_root or Path(__file__).resolve().parents[2]
    args = build_parser(resolved_project_root).parse_args()
    workflow_mode = resolve_workflow_mode(resolved_project_root, args)

    if workflow_mode == "langgraph":
        from src.synthetic_ner.tasks.orchestrator import run_langgraph_workflow

        run_langgraph_workflow(args, resolved_project_root)
        return

    from src.synthetic_ner.engine import run_generation

    run_generation(args, resolved_project_root)
