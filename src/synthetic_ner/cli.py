"""CLI entrypoint for the generator."""

import argparse
from pathlib import Path

from .constants import SECTION_WEIGHTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Ollama; insert placeholders (for testing)",
    )
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
        choices=sorted(SECTION_WEIGHTS.keys()),
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
    return parser


def main(project_root: Path | None = None) -> None:
    args = build_parser().parse_args()
    resolved_project_root = project_root or Path(__file__).resolve().parents[2]
    from .engine import run_generation

    run_generation(args, resolved_project_root)
