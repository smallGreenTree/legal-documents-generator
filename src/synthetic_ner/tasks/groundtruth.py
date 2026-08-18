"""Reproducible occurrence-level ground-truth generation."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from src.synthetic_ner.constants import PROSECUTION

CONTRACT_NAME = "ner_groundtruth_mentions"
CONTRACT_VERSION = "1.0.0"
REFERENCE_SCHEMA_VERSION = "1.0.0"
DOCUMENT_MANIFEST_VERSION = "1.0.0"
GROUNDTRUTH_HEADER = (
    "annotation_id",
    "doc_id",
    "entity_text",
    "label",
    "start_char",
    "end_char",
)
ENTITY_REFERENCE_FILENAME = "entity_reference.json"
DOCUMENT_MANIFEST_FILENAME = "document_manifest.json"
GROUNDTRUTH_FILENAME = "groundtruth.tsv"
GROUNDTRUTH_MANIFEST_FILENAME = "groundtruth_manifest.json"
GROUNDTRUTH_ERRORS_FILENAME = "groundtruth_validation_errors.json"

_AMOUNT_RE = re.compile(
    r"(?:£|€|\b(?:GBP|EUR)\s*)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|m|thousand|k))?",
    re.IGNORECASE,
)


class GroundTruthError(ValueError):
    """Raised when ground truth cannot be safely generated or published."""

    def __init__(self, doc_id: str, issues: list[str]) -> None:
        self.doc_id = doc_id
        self.issues = issues
        super().__init__(f"Ground-truth validation failed for {doc_id}: {'; '.join(issues)}")


@dataclass(frozen=True, slots=True)
class GroundTruthContract:
    name: str
    version: str
    columns: tuple[str, ...]
    allowed_labels: frozenset[str]
    nested_same_labels: frozenset[str]
    prefer_longest_same_label: bool
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class MentionAnnotation:
    annotation_id: str
    doc_id: str
    entity_text: str
    label: str
    start_char: int
    end_char: int


def load_groundtruth_contract(path: Path | str) -> GroundTruthContract:
    contract_path = Path(path)
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Ground-truth contract must be a mapping: {contract_path}")

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

    overlap_policy = raw.get("overlap_policy") or {}
    nested_labels = (
        overlap_policy.get("allow_nested_same_label", {})
        if isinstance(overlap_policy, dict)
        else {}
    )
    if not isinstance(nested_labels, list):
        raise ValueError("overlap_policy.allow_nested_same_label must be a list")
    prefer_longest_same_label = (
        overlap_policy.get("prefer_longest_same_label")
        if isinstance(overlap_policy, dict)
        else None
    )
    if not isinstance(prefer_longest_same_label, bool):
        raise ValueError("overlap_policy.prefer_longest_same_label must be a boolean")

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

    return GroundTruthContract(
        name=raw["contract_name"],
        version=raw["contract_version"],
        columns=column_names,
        allowed_labels=frozenset(allowed_labels),
        nested_same_labels=frozenset(str(label) for label in nested_labels),
        prefer_longest_same_label=prefer_longest_same_label,
        path=contract_path,
        sha256=_sha256(contract_path),
    )


def write_document_reference_artifacts(
    *,
    doc_dir: Path,
    doc_id: str,
    document: Any,
    document_path: Path,
    address_surface_forms: int,
) -> tuple[Path, Path]:
    """Freeze the entity catalogue and hashes needed for later reproduction."""
    reference_path = doc_dir / ENTITY_REFERENCE_FILENAME
    reference_payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "doc_id": doc_id,
        "entities": build_entity_references(
            document,
            address_surface_forms=address_surface_forms,
        ),
    }
    _write_json_atomic(reference_path, reference_payload)

    manifest_path = doc_dir / DOCUMENT_MANIFEST_FILENAME
    manifest_payload = {
        "manifest_version": DOCUMENT_MANIFEST_VERSION,
        "doc_id": doc_id,
        "encoding": "UTF-8",
        "offset_unit": "unicode_code_points",
        "document": {
            "path": document_path.name,
            "sha256": _sha256(document_path),
        },
        "entity_reference": {
            "path": reference_path.name,
            "sha256": _sha256(reference_path),
            "schema_version": REFERENCE_SCHEMA_VERSION,
        },
    }
    _write_json_atomic(manifest_path, manifest_payload)
    return reference_path, manifest_path


def build_entity_references(
    document: Any,
    *,
    address_surface_forms: int = 3,
) -> list[dict[str, Any]]:
    """Build a de-duplicated catalogue of generator-authorized entity surfaces."""
    references: dict[tuple[str, str], dict[str, Any]] = {}
    _add_people_groups(references, document, address_surface_forms)
    _add_organisation_groups(references, document, address_surface_forms)
    _add_metadata_references(references, document.metadata)
    _add_count_amount_references(references, document.counts_list)
    _add_amount_references(references, document.amounts)
    _add_negative_control_references(references, document.metadata)
    return list(references.values())


def _add_people_groups(
    references: dict[tuple[str, str], dict[str, Any]],
    document: Any,
    address_surface_forms: int,
) -> None:
    for group_name, people in (
        ("defendants", document.defendants),
        ("collateral", document.collateral),
    ):
        for index, person in enumerate(people):
            _add_person_references(
                references,
                person,
                prefix=f"case.{group_name}[{index}]",
                group_name=group_name,
                address_surface_forms=address_surface_forms,
            )


def _add_person_references(
    references: dict[tuple[str, str], dict[str, Any]],
    person: dict[str, Any],
    *,
    prefix: str,
    group_name: str,
    address_surface_forms: int,
) -> None:
    name = person.get("name")
    initials = person.get("initials")
    title_surname = person.get("title_surname")
    short_name = person.get("short_name")
    _add_reference(references, name, "PERSON", f"{prefix}.name", f"{group_name} person")
    _add_reference(
        references,
        initials,
        "INITIAL",
        f"{prefix}.initials",
        f"initials for {name}",
    )
    surname = str(name or "").split()[-1] if name else ""
    if title_surname and title_surname != surname:
        _add_reference(
            references,
            title_surname,
            "TITLE",
            f"{prefix}.title_surname",
            f"title surface for {name}",
        )
    if short_name not in {name, initials, title_surname}:
        _add_reference(
            references,
            short_name,
            "PERSON",
            f"{prefix}.short_name",
            f"short name for {name}",
        )
    _add_person_surface_forms(
        references,
        person,
        prefix=prefix,
        excluded={name, initials, title_surname, short_name},
    )
    _add_reference(
        references,
        person.get("dob"),
        "DATE",
        f"{prefix}.dob",
        f"date of birth for {name}",
    )
    _add_address_references(
        references,
        person,
        prefix=prefix,
        notes=f"address for {name}",
        address_surface_forms=address_surface_forms,
    )


def _add_person_surface_forms(
    references: dict[tuple[str, str], dict[str, Any]],
    person: dict[str, Any],
    *,
    prefix: str,
    excluded: set[Any],
) -> None:
    name = person.get("name")
    for surface_index, surface in enumerate(person.get("surface_forms_list") or []):
        if surface in excluded:
            continue
        _add_reference(
            references,
            surface,
            "PERSON",
            f"{prefix}.surface_forms_list[{surface_index}]",
            f"configured person surface for {name}",
        )


def _add_organisation_groups(
    references: dict[tuple[str, str], dict[str, Any]],
    document: Any,
    address_surface_forms: int,
) -> None:
    for group_name, orgs in (
        ("charged_orgs", document.charged_orgs),
        ("associated_orgs", document.associated_orgs),
    ):
        for index, org in enumerate(orgs):
            prefix = f"case.{group_name}[{index}]"
            name = org.get("name")
            _add_reference(
                references,
                name,
                "ORG",
                f"{prefix}.name",
                f"{group_name} organisation",
            )
            _add_address_references(
                references,
                org,
                prefix=prefix,
                notes=f"address for {name}",
                address_surface_forms=address_surface_forms,
            )
            _add_reference(
                references,
                org.get("vat"),
                "VAT",
                f"{prefix}.vat",
                f"VAT number for {name}",
            )


def _add_metadata_references(
    references: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    for field_name in ("case_number", "cross_ref", "legal_reference"):
        _add_reference(
            references,
            metadata.get(field_name),
            "CASE_REFERENCE",
            f"case.metadata.{field_name}",
            field_name.replace("_", " "),
        )
    _add_reference(
        references,
        metadata.get("filing_date"),
        "DATE",
        "case.metadata.filing_date",
        "filing date",
    )
    offence_period = metadata.get("offence_period")
    if not offence_period:
        return
    _add_reference(
        references,
        offence_period[0],
        "DATE",
        "case.metadata.offence_period.start",
        "offence period start",
    )
    _add_reference(
        references,
        offence_period[1],
        "DATE",
        "case.metadata.offence_period.end",
        "offence period end",
    )


def _add_count_amount_references(
    references: dict[tuple[str, str], dict[str, Any]],
    counts_list: list[dict[str, Any]],
) -> None:
    for count_index, count in enumerate(counts_list):
        for amount_index, amount in enumerate(_extract_amounts(count.get("particulars", ""))):
            _add_reference(
                references,
                amount,
                "AMOUNT",
                f"case.counts[{count_index}].particulars.amounts[{amount_index}]",
                "amount in count particulars",
            )


def _add_negative_control_references(
    references: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    _add_reference(
        references,
        PROSECUTION,
        "NEGATIVE_CONTROL",
        "template.prosecution",
        "prosecution negative control",
    )
    _add_reference(
        references,
        metadata.get("court"),
        "NEGATIVE_CONTROL",
        "case.metadata.court",
        "court negative control",
    )


def generate_groundtruth_for_document(
    *,
    document_dir: Path | str,
    contract_path: Path | str,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish ground truth for one document package."""
    doc_dir = Path(document_dir)
    contract = load_groundtruth_contract(contract_path)
    manifest_path = doc_dir / DOCUMENT_MANIFEST_FILENAME
    reference_path = doc_dir / ENTITY_REFERENCE_FILENAME
    manifest = _read_json(manifest_path)
    doc_id = _required_string(manifest.get("doc_id"), "document_manifest.doc_id")
    document_info = _required_mapping(manifest.get("document"), "document_manifest.document")
    reference_info = _required_mapping(
        manifest.get("entity_reference"),
        "document_manifest.entity_reference",
    )
    document_path = doc_dir / _required_string(document_info.get("path"), "document.path")
    declared_reference_path = doc_dir / _required_string(
        reference_info.get("path"),
        "entity_reference.path",
    )
    if declared_reference_path != reference_path:
        raise GroundTruthError(doc_id, ["entity reference path must be entity_reference.json"])

    issues = _input_integrity_issues(
        document_path=document_path,
        reference_path=reference_path,
        document_info=document_info,
        reference_info=reference_info,
    )
    if issues:
        _write_validation_errors(doc_dir, doc_id, issues)
        raise GroundTruthError(doc_id, issues)

    document_text = document_path.read_text(encoding="utf-8")
    reference_payload = _read_json(reference_path)
    if reference_payload.get("doc_id") != doc_id:
        issues = ["entity_reference.doc_id does not match document_manifest.doc_id"]
        _write_validation_errors(doc_dir, doc_id, issues)
        raise GroundTruthError(doc_id, issues)
    references = reference_payload.get("entities")
    if not isinstance(references, list):
        issues = ["entity_reference.entities must be a list"]
        _write_validation_errors(doc_dir, doc_id, issues)
        raise GroundTruthError(doc_id, issues)

    try:
        existing = _reuse_existing_groundtruth(
            doc_dir=doc_dir,
            doc_id=doc_id,
            document_text=document_text,
            document_sha256=_sha256(document_path),
            reference_sha256=_sha256(reference_path),
            contract=contract,
            references=references,
        )
        if existing is not None:
            return existing

        annotations = build_mention_annotations(
            doc_id=doc_id,
            document_text=document_text,
            references=references,
            contract=contract,
        )
        validate_mention_annotations(
            doc_id=doc_id,
            document_text=document_text,
            annotations=annotations,
            references=references,
            contract=contract,
        )
        result = _publish_groundtruth(
            doc_dir=doc_dir,
            doc_id=doc_id,
            document_text=document_text,
            document_sha256=_sha256(document_path),
            reference_sha256=_sha256(reference_path),
            annotations=annotations,
            references=references,
            contract=contract,
        )
    except GroundTruthError as exc:
        _write_validation_errors(doc_dir, doc_id, exc.issues)
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        issues = [f"existing ground truth could not be validated: {exc}"]
        _write_validation_errors(doc_dir, doc_id, issues)
        raise GroundTruthError(doc_id, issues) from exc
    return result


def build_mention_annotations(
    *,
    doc_id: str,
    document_text: str,
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
) -> list[MentionAnnotation]:
    occurrences: set[tuple[int, int, str, str]] = set()
    issues: list[str] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            issues.append(f"entity_reference.entities[{index}] must be a mapping")
            continue
        entity_text = reference.get("entity_text")
        label = reference.get("label")
        if not isinstance(entity_text, str) or not entity_text:
            issues.append(f"entity_reference.entities[{index}].entity_text must be non-empty")
            continue
        if label not in contract.allowed_labels:
            issues.append(f"entity_reference.entities[{index}].label {label!r} is not approved")
            continue
        start = document_text.find(entity_text)
        while start != -1:
            end = start + len(entity_text)
            occurrences.add((start, end, entity_text, label))
            start = document_text.find(entity_text, start + 1)
    if issues:
        raise GroundTruthError(doc_id, issues)

    occurrences = _apply_occurrence_overlap_policy(occurrences, contract)
    ordered = sorted(occurrences, key=lambda item: (item[0], item[1], item[3], item[2]))
    return [
        MentionAnnotation(
            annotation_id=f"{doc_id}-{index:03d}",
            doc_id=doc_id,
            entity_text=entity_text,
            label=label,
            start_char=start,
            end_char=end,
        )
        for index, (start, end, entity_text, label) in enumerate(ordered, start=1)
    ]


def validate_mention_annotations(
    *,
    doc_id: str,
    document_text: str,
    annotations: list[MentionAnnotation],
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
) -> None:
    issues: list[str] = []
    annotation_ids: set[str] = set()
    matching_keys: set[tuple[str, int, int, str]] = set()
    span_labels: dict[tuple[str, int, int], str] = {}

    for annotation in annotations:
        issues.extend(
            _annotation_value_issues(
                annotation,
                doc_id=doc_id,
                document_text=document_text,
                allowed_labels=contract.allowed_labels,
            )
        )
        if annotation.annotation_id in annotation_ids:
            issues.append(f"annotation {annotation.annotation_id}: annotation_id is duplicated")
        annotation_ids.add(annotation.annotation_id)
        matching_key = (
            annotation.doc_id,
            annotation.start_char,
            annotation.end_char,
            annotation.label,
        )
        if matching_key in matching_keys:
            issues.append(f"annotation {annotation.annotation_id}: matching key is duplicated")
        matching_keys.add(matching_key)
        _record_span_label(annotation, span_labels, issues)

    _validate_overlaps(annotations, contract, issues)
    expected = _expected_occurrence_keys(document_text, references, contract, issues)
    actual = {
        (annotation.start_char, annotation.end_char, annotation.entity_text, annotation.label)
        for annotation in annotations
    }
    for missing in sorted(expected - actual):
        issues.append(f"missing annotation for occurrence {missing!r}")
    for unexpected in sorted(actual - expected):
        issues.append(f"unexpected annotation occurrence {unexpected!r}")
    if issues:
        raise GroundTruthError(doc_id, issues)


def _annotation_value_issues(
    annotation: MentionAnnotation,
    *,
    doc_id: str,
    document_text: str,
    allowed_labels: frozenset[str],
) -> list[str]:
    prefix = f"annotation {annotation.annotation_id}"
    issues = []
    if annotation.doc_id != doc_id:
        issues.append(f"{prefix}: doc_id does not match {doc_id}")
    if annotation.label not in allowed_labels:
        issues.append(f"{prefix}: label {annotation.label!r} is not approved")
    valid_offsets = 0 <= annotation.start_char < annotation.end_char <= len(document_text)
    if not valid_offsets:
        issues.append(
            f"{prefix}: invalid offsets {annotation.start_char}:{annotation.end_char} "
            f"for document length {len(document_text)}"
        )
    elif document_text[annotation.start_char : annotation.end_char] != annotation.entity_text:
        issues.append(f"{prefix}: document slice does not equal entity_text")
    return issues


def _record_span_label(
    annotation: MentionAnnotation,
    span_labels: dict[tuple[str, int, int], str],
    issues: list[str],
) -> None:
    span_key = (annotation.doc_id, annotation.start_char, annotation.end_char)
    existing_label = span_labels.get(span_key)
    if existing_label is not None and existing_label != annotation.label:
        issues.append(
            f"annotation {annotation.annotation_id}: span conflicts with label {existing_label!r}"
        )
    span_labels[span_key] = annotation.label


def read_groundtruth_tsv(path: Path | str) -> list[MentionAnnotation]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != GROUNDTRUTH_HEADER:
            raise ValueError("Ground-truth TSV header does not match the contract")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    MentionAnnotation(
                        annotation_id=row["annotation_id"],
                        doc_id=row["doc_id"],
                        entity_text=row["entity_text"],
                        label=row["label"],
                        start_char=int(row["start_char"]),
                        end_char=int(row["end_char"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ground-truth TSV row {row_number}: {exc}") from exc
    return rows


def require_completed_groundtruth(document_dir: Path | str, doc_id: str) -> dict[str, Any]:
    doc_dir = Path(document_dir)
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    manifest_path = doc_dir / GROUNDTRUTH_MANIFEST_FILENAME
    if not groundtruth_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Ground truth has not completed for {doc_id}")
    manifest = _read_json(manifest_path)
    if manifest.get("doc_id") != doc_id or manifest.get("status") != "completed":
        raise RuntimeError(f"Ground-truth manifest is not completed for {doc_id}")
    if manifest.get("groundtruth_sha256") != _sha256(groundtruth_path):
        raise RuntimeError(f"Ground-truth checksum mismatch for {doc_id}")
    return manifest


def discover_document_packages(input_directory: Path | str) -> list[Path]:
    root = Path(input_directory)
    if not root.is_dir():
        raise ValueError(f"Ground-truth input directory does not exist: {root}")
    if (root / DOCUMENT_MANIFEST_FILENAME).is_file():
        return [root]
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / DOCUMENT_MANIFEST_FILENAME).is_file()
    )


def _add_reference(
    references: dict[tuple[str, str], dict[str, Any]],
    value: Any,
    label: str,
    source_field: str,
    notes: str,
) -> None:
    if value is None:
        return
    entity_text = str(value).strip()
    if not entity_text:
        return
    key = (entity_text, label)
    existing = references.get(key)
    if existing is None:
        references[key] = {
            "entity_text": entity_text,
            "label": label,
            "source_fields": [source_field],
            "notes": [notes],
        }
        return
    if source_field not in existing["source_fields"]:
        existing["source_fields"].append(source_field)
    if notes not in existing["notes"]:
        existing["notes"].append(notes)


def _add_address_references(
    references: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
    *,
    prefix: str,
    notes: str,
    address_surface_forms: int,
) -> None:
    fields = ("address", "street", "city_postcode")
    for field_name in fields[:address_surface_forms]:
        _add_reference(
            references,
            record.get(field_name),
            "ADDRESS",
            f"{prefix}.{field_name}",
            notes,
        )


def _add_amount_references(
    references: dict[tuple[str, str], dict[str, Any]],
    amounts: dict[str, Any],
) -> None:
    for field_name, notes in (
        ("total_loss", "total alleged loss"),
        ("inflated_invoice_value", "inflated invoice value"),
    ):
        _add_reference(
            references,
            amounts.get(field_name),
            "AMOUNT",
            f"case.amounts.{field_name}",
            notes,
        )
    for index, transfer in enumerate(amounts.get("transfers", [])):
        if not isinstance(transfer, dict):
            continue
        _add_reference(
            references,
            transfer.get("amount"),
            "AMOUNT",
            f"case.amounts.transfers[{index}].amount",
            "transfer amount",
        )


def _extract_amounts(value: str) -> list[str]:
    amounts: list[str] = []
    for match in _AMOUNT_RE.findall(value):
        cleaned = str(match).strip().rstrip(".,;:")
        if cleaned and cleaned not in amounts:
            amounts.append(cleaned)
    return amounts


def _input_integrity_issues(
    *,
    document_path: Path,
    reference_path: Path,
    document_info: dict[str, Any],
    reference_info: dict[str, Any],
) -> list[str]:
    issues = []
    for label, path, info in (
        ("document", document_path, document_info),
        ("entity reference", reference_path, reference_info),
    ):
        if not path.is_file():
            issues.append(f"{label} file does not exist: {path}")
            continue
        expected_hash = info.get("sha256")
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            issues.append(
                f"{label} SHA-256 mismatch: expected {expected_hash!r}, got {actual_hash!r}"
            )
    return issues


def _expected_occurrence_keys(
    document_text: str,
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
    issues: list[str],
) -> set[tuple[int, int, str, str]]:
    expected: set[tuple[int, int, str, str]] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            continue
        entity_text = reference.get("entity_text")
        label = reference.get("label")
        if not isinstance(entity_text, str) or not entity_text:
            continue
        if label not in contract.allowed_labels:
            issues.append(f"entity_reference.entities[{index}].label {label!r} is not approved")
            continue
        start = document_text.find(entity_text)
        while start != -1:
            expected.add((start, start + len(entity_text), entity_text, label))
            start = document_text.find(entity_text, start + 1)
    return _apply_occurrence_overlap_policy(expected, contract)


def _apply_occurrence_overlap_policy(
    occurrences: set[tuple[int, int, str, str]],
    contract: GroundTruthContract,
) -> set[tuple[int, int, str, str]]:
    if not contract.prefer_longest_same_label:
        return occurrences
    retained = set(occurrences)
    for occurrence in occurrences:
        start, end, _entity_text, label = occurrence
        if label in contract.nested_same_labels:
            continue
        contained_by_longer = any(
            other_label == label
            and other_start <= start
            and end <= other_end
            and (other_start < start or end < other_end)
            for other_start, other_end, _other_text, other_label in occurrences
        )
        if contained_by_longer:
            retained.discard(occurrence)
    return retained


def _validate_overlaps(
    annotations: list[MentionAnnotation],
    contract: GroundTruthContract,
    issues: list[str],
) -> None:
    ordered = sorted(annotations, key=lambda row: (row.start_char, row.end_char))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if right.start_char >= left.end_char:
                break
            if left.start_char == right.start_char and left.end_char == right.end_char:
                continue
            nested = (left.start_char <= right.start_char and right.end_char <= left.end_char) or (
                right.start_char <= left.start_char and left.end_char <= right.end_char
            )
            allowed_nested = (
                nested and left.label == right.label and left.label in contract.nested_same_labels
            )
            if not allowed_nested:
                issues.append(
                    "overlap is not allowed between "
                    f"{left.annotation_id} ({left.start_char}:{left.end_char}, {left.label}) and "
                    f"{right.annotation_id} ({right.start_char}:{right.end_char}, {right.label})"
                )


def _publish_groundtruth(
    *,
    doc_dir: Path,
    doc_id: str,
    document_text: str,
    document_sha256: str,
    reference_sha256: str,
    annotations: list[MentionAnnotation],
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
) -> dict[str, Any]:
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    pending_path = doc_dir / f".{GROUNDTRUTH_FILENAME}.pending"
    with pending_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(GROUNDTRUTH_HEADER),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(annotation) for annotation in annotations)

    readback = read_groundtruth_tsv(pending_path)
    validate_mention_annotations(
        doc_id=doc_id,
        document_text=document_text,
        annotations=readback,
        references=references,
        contract=contract,
    )
    os.replace(pending_path, groundtruth_path)

    manifest_path = doc_dir / GROUNDTRUTH_MANIFEST_FILENAME
    manifest = {
        "status": "completed",
        "doc_id": doc_id,
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_sha256": contract.sha256,
        "document_sha256": document_sha256,
        "entity_reference_sha256": reference_sha256,
        "groundtruth_sha256": _sha256(groundtruth_path),
        "annotation_count": len(annotations),
        "encoding": "UTF-8",
        "delimiter": "tab",
        "line_endings": "LF",
        "offset_unit": "unicode_code_points",
        "overlap_policy": {
            "prefer_longest_same_label": contract.prefer_longest_same_label,
            "allow_nested_same_label": sorted(contract.nested_same_labels),
        },
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json_atomic(manifest_path, manifest)
    error_path = doc_dir / GROUNDTRUTH_ERRORS_FILENAME
    if error_path.exists():
        error_path.unlink()
    return _result_payload(doc_dir, manifest, reused=False)


def _reuse_existing_groundtruth(
    *,
    doc_dir: Path,
    doc_id: str,
    document_text: str,
    document_sha256: str,
    reference_sha256: str,
    contract: GroundTruthContract,
    references: list[dict[str, Any]],
) -> dict[str, Any] | None:
    groundtruth_path = doc_dir / GROUNDTRUTH_FILENAME
    manifest_path = doc_dir / GROUNDTRUTH_MANIFEST_FILENAME
    if not groundtruth_path.exists() and not manifest_path.exists():
        return None
    if not groundtruth_path.is_file() or not manifest_path.is_file():
        raise GroundTruthError(
            doc_id,
            ["existing ground truth is incomplete and will not be overwritten"],
        )
    manifest = _read_json(manifest_path)
    expected = {
        "status": "completed",
        "doc_id": doc_id,
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_sha256": contract.sha256,
        "document_sha256": document_sha256,
        "entity_reference_sha256": reference_sha256,
        "groundtruth_sha256": _sha256(groundtruth_path),
    }
    mismatches = [
        f"existing groundtruth manifest {key} mismatch"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise GroundTruthError(doc_id, mismatches)
    annotations = read_groundtruth_tsv(groundtruth_path)
    validate_mention_annotations(
        doc_id=doc_id,
        document_text=document_text,
        annotations=annotations,
        references=references,
        contract=contract,
    )
    return _result_payload(doc_dir, manifest, reused=True)


def _result_payload(doc_dir: Path, manifest: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "status": "completed",
        "doc_id": manifest["doc_id"],
        "contract_version": manifest["contract_version"],
        "annotation_count": manifest["annotation_count"],
        "document_sha256": manifest["document_sha256"],
        "groundtruth_sha256": manifest["groundtruth_sha256"],
        "groundtruth_path": str(doc_dir / GROUNDTRUTH_FILENAME),
        "manifest_path": str(doc_dir / GROUNDTRUTH_MANIFEST_FILENAME),
        "reused": reused,
    }


def _write_validation_errors(doc_dir: Path, doc_id: str, issues: list[str]) -> None:
    _write_json_atomic(
        doc_dir / GROUNDTRUTH_ERRORS_FILENAME,
        {
            "status": "failed",
            "doc_id": doc_id,
            "issues": issues,
            "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = path.with_name(f".{path.name}.pending")
    pending_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(pending_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return payload


def _required_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
