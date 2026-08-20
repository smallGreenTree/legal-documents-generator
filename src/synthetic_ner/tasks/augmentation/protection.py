"""Protect NER spans and reconstruct validated morphology variants."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from src.synthetic_ner.tasks.augmentation.constants import (
    LITERAL_TOKEN_TEMPLATE,
    MENTION_TOKEN_TEMPLATE,
    NUMERIC_LITERAL_PATTERN,
    PROTECTED_TOKEN_PATTERN,
)
from src.synthetic_ner.tasks.groundtruth import load_groundtruth_contract
from src.synthetic_ner.tasks.groundtruth.annotations import validate_mention_annotations
from src.synthetic_ner.tasks.groundtruth.models import MentionAnnotation
from src.synthetic_ner.types.augmentation import (
    MorphologyError,
    MorphologyTransformation,
    MorphologyVariant,
    ProtectedDocument,
    ProtectedLiteral,
    ProtectedMention,
)


def protect_document_text(
    document_text: str,
    annotations: tuple[MentionAnnotation, ...],
) -> ProtectedDocument:
    """Replace validated entity spans and unannotated numeric facts with tokens."""
    ordered = tuple(sorted(annotations, key=lambda row: (row.start_char, row.end_char)))
    _validate_source_spans(document_text, ordered)
    mentions = tuple(
        ProtectedMention(
            token=MENTION_TOKEN_TEMPLATE.format(index=index),
            text=document_text[start:end],
            source_start=start,
            source_end=end,
            annotations=component,
        )
        for index, (start, end, component) in enumerate(
            _annotation_components(ordered),
            start=1,
        )
    )
    protected_text = document_text
    for mention in reversed(mentions):
        protected_text = (
            protected_text[: mention.source_start]
            + mention.token
            + protected_text[mention.source_end :]
        )
    protected_text, literals = _protect_numeric_literals(protected_text)
    return ProtectedDocument(text=protected_text, mentions=mentions, literals=literals)


def reconstruct_morphology_variant(
    *,
    source_text: str,
    protected: ProtectedDocument,
    transformed_text: str,
    variant_doc_id: str,
    transformation: MorphologyTransformation,
    minimum_change_ratio: float,
    maximum_change_ratio: float,
    contract_path: Path | str,
) -> MorphologyVariant:
    """Restore protected values, rebuild offsets, and validate exact occurrences."""
    token_values = {
        **{mention.token: mention.text for mention in protected.mentions},
        **{literal.token: literal.text for literal in protected.literals},
    }
    _validate_token_inventory(transformed_text, token_values)
    mention_by_token = {mention.token: mention for mention in protected.mentions}
    restored_parts: list[str] = []
    provisional: list[tuple[int, int, str, str]] = []
    restored_length = 0
    cursor = 0
    for match in re.finditer(PROTECTED_TOKEN_PATTERN, transformed_text):
        prefix = transformed_text[cursor : match.start()]
        restored_parts.append(prefix)
        restored_length += len(prefix)
        token = match.group(0)
        value = token_values[token]
        start = restored_length
        restored_parts.append(value)
        restored_length += len(value)
        mention = mention_by_token.get(token)
        if mention is not None:
            for annotation in mention.annotations:
                relative_start = annotation.start_char - mention.source_start
                relative_end = annotation.end_char - mention.source_start
                provisional.append(
                    (
                        start + relative_start,
                        start + relative_end,
                        annotation.entity_text,
                        annotation.label,
                    )
                )
        cursor = match.end()
    suffix = transformed_text[cursor:]
    restored_parts.append(suffix)
    restored_text = "".join(restored_parts)
    change_ratio = 1.0 - SequenceMatcher(None, source_text, restored_text).ratio()
    if change_ratio < minimum_change_ratio:
        raise MorphologyError(
            f"Variant change ratio {change_ratio:.4f} is below minimum change ratio "
            f"{minimum_change_ratio:.4f}"
        )
    if change_ratio > maximum_change_ratio:
        raise MorphologyError(
            f"Variant change ratio {change_ratio:.4f} exceeds maximum change ratio "
            f"{maximum_change_ratio:.4f}"
        )

    annotations = tuple(
        MentionAnnotation(
            annotation_id=f"{variant_doc_id}-{index:03d}",
            doc_id=variant_doc_id,
            entity_text=entity_text,
            label=label,
            start_char=start,
            end_char=end,
        )
        for index, (start, end, entity_text, label) in enumerate(
            sorted(provisional),
            start=1,
        )
    )
    references = [
        {"entity_text": entity_text, "label": label}
        for entity_text, label in sorted(
            {
                (annotation.entity_text, annotation.label)
                for mention in protected.mentions
                for annotation in mention.annotations
            }
        )
    ]
    validate_mention_annotations(
        doc_id=variant_doc_id,
        document_text=restored_text,
        annotations=list(annotations),
        references=references,
        contract=load_groundtruth_contract(contract_path),
    )
    _validate_mention_inventory(protected, annotations)
    return MorphologyVariant(
        doc_id=variant_doc_id,
        transformation=transformation,
        text=restored_text,
        annotations=annotations,
        change_ratio=change_ratio,
    )


def _validate_source_spans(
    document_text: str,
    annotations: tuple[MentionAnnotation, ...],
) -> None:
    for annotation in annotations:
        valid_offsets = 0 <= annotation.start_char < annotation.end_char <= len(document_text)
        if (
            not valid_offsets
            or document_text[annotation.start_char : annotation.end_char] != annotation.entity_text
        ):
            raise MorphologyError(
                f"Source annotation {annotation.annotation_id} does not match document text"
            )


def _annotation_components(
    annotations: tuple[MentionAnnotation, ...],
) -> list[tuple[int, int, tuple[MentionAnnotation, ...]]]:
    components: list[tuple[int, int, tuple[MentionAnnotation, ...]]] = []
    for annotation in annotations:
        if not components or annotation.start_char >= components[-1][1]:
            components.append((annotation.start_char, annotation.end_char, (annotation,)))
            continue
        start, end, rows = components[-1]
        components[-1] = (start, max(end, annotation.end_char), (*rows, annotation))
    return components


def _protect_numeric_literals(text: str) -> tuple[str, tuple[ProtectedLiteral, ...]]:
    literals: list[ProtectedLiteral] = []
    output: list[str] = []
    for part in re.split(f"({PROTECTED_TOKEN_PATTERN})", text):
        if re.fullmatch(PROTECTED_TOKEN_PATTERN, part):
            output.append(part)
            continue

        def replace(match: re.Match[str]) -> str:
            token = LITERAL_TOKEN_TEMPLATE.format(index=len(literals) + 1)
            literals.append(ProtectedLiteral(token=token, text=match.group(0)))
            return token

        output.append(re.sub(NUMERIC_LITERAL_PATTERN, replace, part))
    return "".join(output), tuple(literals)


def _validate_token_inventory(transformed_text: str, token_values: dict[str, str]) -> None:
    observed = Counter(re.findall(PROTECTED_TOKEN_PATTERN, transformed_text))
    unknown = sorted(set(observed) - set(token_values))
    if unknown:
        raise MorphologyError(f"Unknown protected tokens returned: {', '.join(unknown)}")
    invalid = [token for token in token_values if observed[token] != 1]
    if invalid:
        raise MorphologyError(
            "Every protected token must appear exactly once; invalid tokens: " + ", ".join(invalid)
        )


def _validate_mention_inventory(
    protected: ProtectedDocument,
    annotations: tuple[MentionAnnotation, ...],
) -> None:
    expected = Counter(
        (annotation.entity_text, annotation.label)
        for mention in protected.mentions
        for annotation in mention.annotations
    )
    actual = Counter((annotation.entity_text, annotation.label) for annotation in annotations)
    if actual != expected:
        raise MorphologyError("The reconstructed entity mention inventory changed")
