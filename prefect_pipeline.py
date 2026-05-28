"""Compatibility entrypoint for Prefect orchestration commands."""

from __future__ import annotations

import argparse

from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.quality import score_existing_document
from src.synthetic_ner.tasks.quality_report import DEFAULT_QUALITY_CONFIG_PATH

__all__ = ["generate_dataset", "score_existing_document"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic NER generation through Prefect.")
    parser.add_argument(
        "--score-doc-id",
        nargs="?",
        const="",
        default=None,
        help=(
            "Score an existing document id instead of running generation. "
            "Pass without a value to pause and select the document in Prefect."
        ),
    )
    parser.add_argument(
        "--score-document-quality",
        action="store_true",
        help="Run the quality scoring flow and pause for document selection.",
    )
    parser.add_argument(
        "--no-document-selection-pause",
        action="store_true",
        help="Score the supplied --score-doc-id directly without the Prefect selection pause.",
    )
    parser.add_argument(
        "--quality-config",
        default=DEFAULT_QUALITY_CONFIG_PATH,
        help="Quality scoring config path relative to project root.",
    )
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case recipe config path relative to project root.",
    )
    parser.add_argument("--documents", "--count", dest="documents", type=int, default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--fraud-type", default=None)
    parser.add_argument("--from-schema", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--review-scenario",
        action="store_true",
        help="Pause after scenario selection so a human can approve or alter inputs.",
    )
    parser.add_argument(
        "--review-entities",
        action="store_true",
        help="Pause after entity resolution so a human can approve or edit document inputs.",
    )
    parser.add_argument(
        "--review-timeout-seconds",
        type=int,
        default=3600,
        help="How long a human review pause waits for input before continuing unchanged.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.score_doc_id is not None or args.score_document_quality:
        score_existing_document(
            doc_id=args.score_doc_id or None,
            case_config=args.case_config,
            quality_config=args.quality_config,
            doc_type=args.doc_type,
            fraud_type=args.fraud_type,
            project_root=args.project_root,
            review_scenario=args.review_scenario,
            review_document_selection=not args.no_document_selection_pause,
            review_timeout_seconds=args.review_timeout_seconds,
        )
        return
    generate_dataset(
        case_config=args.case_config,
        documents=args.documents,
        doc_type=args.doc_type,
        fraud_type=args.fraud_type,
        from_schema=args.from_schema,
        project_root=args.project_root,
        review_scenario=args.review_scenario,
        review_entities=args.review_entities,
        review_timeout_seconds=args.review_timeout_seconds,
    )


if __name__ == "__main__":
    main()
