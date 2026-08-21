"""Validate, serialize, and load resolved document inputs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from src.synthetic_ner.types.document_inputs import DOCUMENT_INPUTS_FILENAME, DocumentInputs

_LIST_FIELDS = (
    "defendants",
    "collateral",
    "charged_orgs",
    "associated_orgs",
    "counts_list",
)
_MAPPING_FIELDS = ("metadata", "amounts")
ENTITY_REFERENCES_FIELD = "entity_references"


def document_inputs_from_payload(
    payload: Any,
    *,
    source: str = DOCUMENT_INPUTS_FILENAME,
) -> DocumentInputs:
    """Validate a payload and return the canonical document-input model."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must be a JSON object")
    for field_name in _LIST_FIELDS:
        if not isinstance(payload.get(field_name), list):
            raise ValueError(f"{source}.{field_name} must be a list")
    for field_name in _MAPPING_FIELDS:
        if not isinstance(payload.get(field_name), dict):
            raise ValueError(f"{source}.{field_name} must be an object")

    evidence_categories = payload.get("evidence_categories", [])
    scenario_brief = payload.get("scenario_brief", {})
    if not isinstance(evidence_categories, list) or not all(
        isinstance(value, str) for value in evidence_categories
    ):
        raise ValueError(f"{source}.evidence_categories must be a string list")
    if not isinstance(scenario_brief, dict):
        raise ValueError(f"{source}.scenario_brief must be an object")

    return DocumentInputs(
        defendants=payload["defendants"],
        collateral=payload["collateral"],
        charged_orgs=payload["charged_orgs"],
        associated_orgs=payload["associated_orgs"],
        metadata=payload["metadata"],
        amounts=payload["amounts"],
        counts_list=payload["counts_list"],
        evidence_categories=evidence_categories,
        scenario_brief=scenario_brief,
    )


def document_inputs_payload(document: DocumentInputs) -> dict[str, Any]:
    """Return the canonical JSON payload for resolved document inputs."""
    return asdict(document)


def load_document_inputs(path: Path | str) -> DocumentInputs:
    """Load and validate a saved document-input file."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return document_inputs_from_payload(payload, source=input_path.name)


def entity_references_from_payload(
    payload: Mapping[str, Any],
    *,
    source: str = DOCUMENT_INPUTS_FILENAME,
) -> list[dict[str, Any]]:
    """Validate optional direct entity references used by imported corpora."""
    raw = payload.get(ENTITY_REFERENCES_FIELD, [])
    if not isinstance(raw, list):
        raise ValueError(f"{source}.{ENTITY_REFERENCES_FIELD} must be a list")
    references: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}.{ENTITY_REFERENCES_FIELD}[{index}] must be an object")
        entity_text = item.get("entity_text")
        label = item.get("label")
        if not isinstance(entity_text, str) or not entity_text.strip():
            raise ValueError(
                f"{source}.{ENTITY_REFERENCES_FIELD}[{index}].entity_text must be non-empty"
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{source}.{ENTITY_REFERENCES_FIELD}[{index}].label must be non-empty")
        references.append(dict(item))
    return references


def write_document_inputs(path: Path | str, document: DocumentInputs) -> Path:
    """Atomically write resolved inputs using the canonical serialization."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = output_path.with_name(f".{output_path.name}.pending")
    pending_path.write_text(
        json.dumps(
            document_inputs_payload(document),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(pending_path, output_path)
    return output_path
