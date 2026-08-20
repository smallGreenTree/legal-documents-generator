"""Compatibility entrypoint for Prefect orchestration commands."""

from __future__ import annotations

import argparse

from src.synthetic_ner.prefect_flows.augmentation import generate_morphological_variations
from src.synthetic_ner.prefect_flows.generation import generate_dataset
from src.synthetic_ner.prefect_flows.groundtruth import generate_groundtruth_directory

__all__ = [
    "generate_dataset",
    "generate_groundtruth_directory",
    "generate_morphological_variations",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic NER generation through Prefect.")
    parser.add_argument(
        "--groundtruth-directory",
        default=None,
        help="Generate validated ground truth for every document package in this directory.",
    )
    parser.add_argument(
        "--groundtruth-contract",
        default="groundtruth_contract.yaml",
        help="Ground-truth contract path relative to project root.",
    )
    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Run the morphological augmentation flow.",
    )
    parser.add_argument(
        "--morphology-input",
        default="",
        help="A .txt file, document package, or parent package folder.",
    )
    parser.add_argument(
        "--review-morphology",
        action="store_true",
        help="Pause with the input-path field and transformation checkboxes.",
    )
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case recipe config path relative to project root.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Jinja template path relative to project root.",
    )
    parser.add_argument("--documents", "--count", dest="documents", type=int, default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--fraud-type", default=None)
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
    if args.morphology:
        generate_morphological_variations(
            input_path=args.morphology_input,
            project_root=args.project_root,
            review=args.review_morphology,
            review_timeout_seconds=args.review_timeout_seconds,
        )
        return
    if args.groundtruth_directory is not None:
        generate_groundtruth_directory(
            input_directory=args.groundtruth_directory,
            project_root=args.project_root,
            contract_path=args.groundtruth_contract,
        )
        return
    generate_dataset(
        case_config=args.case_config,
        template=args.template,
        documents=args.documents,
        doc_type=args.doc_type,
        fraud_type=args.fraud_type,
        project_root=args.project_root,
        review_scenario=args.review_scenario,
        review_entities=args.review_entities,
        review_timeout_seconds=args.review_timeout_seconds,
    )


if __name__ == "__main__":
    main()
