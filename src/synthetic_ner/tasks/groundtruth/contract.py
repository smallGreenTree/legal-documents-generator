"""Loading and validation for the ground-truth TSV contract."""

from pathlib import Path

import yaml

from src.synthetic_ner.tasks.groundtruth.files import sha256_file
from src.synthetic_ner.tasks.groundtruth.models import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    GROUNDTRUTH_HEADER,
    GroundTruthContract,
)


def load_groundtruth_contract(path: Path | str) -> GroundTruthContract:
    contract_path = Path(path)
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Ground-truth contract must be a mapping: {contract_path}")

    column_names, allowed_labels = _contract_columns(raw)
    nested_labels, prefer_longest = _overlap_policy(raw)
    _validate_contract_metadata(raw)
    return GroundTruthContract(
        name=raw["contract_name"],
        version=raw["contract_version"],
        columns=column_names,
        allowed_labels=frozenset(allowed_labels),
        nested_same_labels=frozenset(str(label) for label in nested_labels),
        prefer_longest_same_label=prefer_longest,
        path=contract_path,
        sha256=sha256_file(contract_path),
    )


def _contract_columns(raw: dict) -> tuple[tuple[str, ...], list[str]]:
    columns = raw.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("Ground-truth contract columns must be a mapping")
    column_names = tuple(columns)
    if column_names != GROUNDTRUTH_HEADER:
        raise ValueError(
            "Ground-truth contract columns must be exactly: " + ", ".join(GROUNDTRUTH_HEADER)
        )
    label_config = columns.get("label")
    allowed_labels = label_config.get("allowed_values") if isinstance(label_config, dict) else None
    if not isinstance(allowed_labels, list) or not all(
        isinstance(label, str) and label for label in allowed_labels
    ):
        raise ValueError("Ground-truth contract label.allowed_values must be a string list")
    return column_names, allowed_labels


def _overlap_policy(raw: dict) -> tuple[list, bool]:
    overlap_policy = raw.get("overlap_policy") or {}
    nested_labels = (
        overlap_policy.get("allow_nested_same_label", {})
        if isinstance(overlap_policy, dict)
        else {}
    )
    if not isinstance(nested_labels, list):
        raise ValueError("overlap_policy.allow_nested_same_label must be a list")
    prefer_longest = (
        overlap_policy.get("prefer_longest_same_label")
        if isinstance(overlap_policy, dict)
        else None
    )
    if not isinstance(prefer_longest, bool):
        raise ValueError("overlap_policy.prefer_longest_same_label must be a boolean")
    return nested_labels, prefer_longest


def _validate_contract_metadata(raw: dict) -> None:
    expected = {
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "format": "TSV",
        "encoding": "UTF-8",
        "delimiter": "tab",
        "header_required": True,
        "line_endings": "LF",
        "primary_key": ["annotation_id"],
        "matching_key": ["doc_id", "start_char", "end_char", "label"],
    }
    issues = [
        f"{key} must be {value!r}" for key, value in expected.items() if raw.get(key) != value
    ]
    if issues:
        raise ValueError("Invalid ground-truth contract: " + "; ".join(issues))
