"""Occurrence discovery, annotation construction, and TSV validation."""

import csv
from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.groundtruth.models import (
    GROUNDTRUTH_HEADER,
    GroundTruthContract,
    GroundTruthError,
    MentionAnnotation,
)

Occurrence = tuple[int, int, str, str]


def build_mention_annotations(
    *,
    doc_id: str,
    document_text: str,
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
) -> list[MentionAnnotation]:
    issues: list[str] = []
    occurrences = expected_occurrences(document_text, references, contract, issues)
    if issues:
        raise GroundTruthError(doc_id, issues)
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
        issues.extend(_annotation_value_issues(annotation, doc_id, document_text, contract))
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
    expected = expected_occurrences(document_text, references, contract, issues)
    actual = {(row.start_char, row.end_char, row.entity_text, row.label) for row in annotations}
    issues.extend(f"missing annotation for occurrence {row!r}" for row in sorted(expected - actual))
    issues.extend(f"unexpected annotation occurrence {row!r}" for row in sorted(actual - expected))
    if issues:
        raise GroundTruthError(doc_id, issues)


def read_groundtruth_tsv(path: Path | str) -> list[MentionAnnotation]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != GROUNDTRUTH_HEADER:
            raise ValueError("Ground-truth TSV header does not match the contract")
        return [_annotation_from_row(row, row_number) for row_number, row in enumerate(reader, 2)]


def expected_occurrences(
    document_text: str,
    references: list[dict[str, Any]],
    contract: GroundTruthContract,
    issues: list[str],
) -> set[Occurrence]:
    expected: set[Occurrence] = set()
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            issues.append(f"entity references[{index}] must be a mapping")
            continue
        entity_text = reference.get("entity_text")
        label = reference.get("label")
        if not isinstance(entity_text, str) or not entity_text:
            issues.append(f"entity references[{index}].entity_text must be non-empty")
            continue
        if label not in contract.allowed_labels:
            issues.append(f"entity references[{index}].label {label!r} is not approved")
            continue
        start = document_text.find(entity_text)
        while start != -1:
            expected.add((start, start + len(entity_text), entity_text, label))
            start = document_text.find(entity_text, start + 1)
    return _apply_overlap_policy(expected, contract)


def _annotation_from_row(row: dict[str, str], row_number: int) -> MentionAnnotation:
    try:
        return MentionAnnotation(
            annotation_id=row["annotation_id"],
            doc_id=row["doc_id"],
            entity_text=row["entity_text"],
            label=row["label"],
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ground-truth TSV row {row_number}: {exc}") from exc


def _annotation_value_issues(
    annotation: MentionAnnotation,
    doc_id: str,
    document_text: str,
    contract: GroundTruthContract,
) -> list[str]:
    prefix = f"annotation {annotation.annotation_id}"
    issues = []
    if annotation.doc_id != doc_id:
        issues.append(f"{prefix}: doc_id does not match {doc_id}")
    if annotation.label not in contract.allowed_labels:
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


_TITLE_CONTAINED_LABELS = frozenset({"PERSON", "INITIAL"})


def _apply_overlap_policy(
    occurrences: set[Occurrence], contract: GroundTruthContract
) -> set[Occurrence]:
    if not contract.prefer_longest_same_label:
        return occurrences
    retained = set(occurrences)
    for start, end, entity_text, label in occurrences:
        if label in contract.nested_same_labels:
            continue
        contained = any(
            other_label == label
            and other_start <= start
            and end <= other_end
            and (other_start < start or end < other_end)
            for other_start, other_end, _other_text, other_label in occurrences
        )
        if contained:
            retained.discard((start, end, entity_text, label))
            continue
        if label in _TITLE_CONTAINED_LABELS and _contained_in_title(start, end, occurrences):
            retained.discard((start, end, entity_text, label))
    return retained


def _contained_in_title(start: int, end: int, occurrences: set[Occurrence]) -> bool:
    return any(
        other_label == "TITLE"
        and other_start <= start
        and end <= other_end
        and (other_start < start or end < other_end)
        for other_start, other_end, _other_text, other_label in occurrences
    )


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
            if nested and left.label == right.label and left.label in contract.nested_same_labels:
                continue
            issues.append(
                "overlap is not allowed between "
                f"{left.annotation_id} ({left.start_char}:{left.end_char}, {left.label}) and "
                f"{right.annotation_id} ({right.start_char}:{right.end_char}, {right.label})"
            )
