"""Repeated-content checks and cleanup helpers."""

from __future__ import annotations

import re

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")


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


def dedupe_repeated_content(text: str) -> str:
    cleaned = _dedupe_repeated_lines(text)
    if not cleaned:
        return cleaned
    return _dedupe_repeated_sentences(cleaned)


def _dedupe_repeated_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return text

    deduped_lines: list[str] = []
    seen_sentence_keys: set[str] = set()
    for line in lines:
        _append_line_if_new(
            deduped_lines=deduped_lines,
            seen_sentence_keys=seen_sentence_keys,
            line=line,
        )
    return "\n".join(deduped_lines).strip()


def _append_line_if_new(
    *,
    deduped_lines: list[str],
    seen_sentence_keys: set[str],
    line: str,
) -> None:
    normalized_line = " ".join(line.split()).strip()
    if not normalized_line:
        _append_blank_line(deduped_lines)
        return

    key = normalized_line.casefold()
    if _is_duplicate_line(deduped_lines, seen_sentence_keys, key):
        return

    deduped_lines.append(line)
    if len(key) >= 80:
        seen_sentence_keys.add(key)


def _append_blank_line(deduped_lines: list[str]) -> None:
    if deduped_lines and deduped_lines[-1] == "":
        return
    deduped_lines.append("")


def _is_duplicate_line(
    deduped_lines: list[str],
    seen_sentence_keys: set[str],
    key: str,
) -> bool:
    if len(key) >= 80 and key in seen_sentence_keys:
        return True
    return bool(deduped_lines and deduped_lines[-1] and deduped_lines[-1].casefold() == key)


def _dedupe_repeated_sentences(text: str) -> str:
    sentence_parts: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in SENTENCE_SPLIT_RE.split(" ".join(text.split())):
        part = sentence.strip()
        if not part:
            continue
        key = re.sub(r"\s+", " ", part).casefold()
        if len(key) >= 90 and key in seen_sentences:
            continue
        sentence_parts.append(part)
        if len(key) >= 90:
            seen_sentences.add(key)
    return " ".join(sentence_parts).strip()


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
