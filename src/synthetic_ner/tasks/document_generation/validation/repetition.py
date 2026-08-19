"""Repeated-content checks and cleanup helpers."""

from __future__ import annotations

import re

from src.synthetic_ner.tasks.document_generation.constants import (
    SENTENCE_SPLIT_RE,
    TOKEN_RE,
)


def has_repeated_long_sentences(text: str) -> bool:
    normalized_sentences = []
    for sentence in SENTENCE_SPLIT_RE.split(" ".join(text.split())):
        normalized = sentence.strip().lower()
        if len(normalized) < 80:
            continue
        normalized = re.sub(r"\s+", " ", normalized)
        normalized_sentences.append(normalized)
    if len(normalized_sentences) < 2:
        return False
    seen: set[str] = set()
    for sentence in normalized_sentences:
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def has_repeated_sentence_fragments(text: str) -> bool:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return False
    fragment_counts: dict[str, int] = {}
    for sentence in SENTENCE_SPLIT_RE.split(normalized_text):
        tokens = [token.lower() for token in TOKEN_RE.findall(sentence)]
        if len(tokens) < 8:
            continue
        fragment_key = " ".join(tokens[:10])
        fragment_counts[fragment_key] = fragment_counts.get(fragment_key, 0) + 1
        if fragment_counts[fragment_key] >= 2:
            return True
    return False
