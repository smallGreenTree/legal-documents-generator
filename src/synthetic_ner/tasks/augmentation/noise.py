"""Deterministic typo and layout variation for protected legal text."""

from __future__ import annotations

import hashlib
import random
import re
import textwrap

from src.synthetic_ner.tasks.augmentation.constants import (
    PROTECTED_TOKEN_PATTERN,
    TYPO_EXCLUDED_WORDS,
)


def apply_intentional_typos(
    text: str,
    *,
    seed_key: str,
    typo_rate: float,
    max_typos: int,
) -> str:
    """Transpose internal letters in a bounded sample of unprotected context words."""
    rng = _rng(seed_key)
    candidates: list[tuple[int, int, str]] = []
    cursor = 0
    for token_match in re.finditer(PROTECTED_TOKEN_PATTERN, text):
        candidates.extend(_word_candidates(text[cursor : token_match.start()], cursor))
        cursor = token_match.end()
    candidates.extend(_word_candidates(text[cursor:], cursor))
    if not candidates:
        return text
    count = min(max_typos, max(1, round(len(candidates) * typo_rate)))
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    changed = text
    for start, end, word in sorted(selected, reverse=True):
        changed = changed[:start] + _transpose(word, rng) + changed[end:]
    return changed


def apply_random_layout(
    text: str,
    *,
    seed_key: str,
    widths: tuple[int, ...],
) -> str:
    """Reflow paragraphs with reproducible widths, indentation, and blank lines."""
    rng = _rng(seed_key)
    trailing_newline = text.endswith("\n")
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    rendered = []
    for paragraph in paragraphs:
        content = " ".join(paragraph.split())
        indent = rng.choice(("", "  ", "    "))
        rendered.append(
            textwrap.fill(
                content,
                width=rng.choice(widths),
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    changed = "\n\n".join(rendered)
    if trailing_newline:
        changed += "\n"
    if changed == text and changed.strip():
        changed = "  " + changed
    return changed


def _word_candidates(text: str, offset: int) -> list[tuple[int, int, str]]:
    return [
        (offset + match.start(), offset + match.end(), match.group(0))
        for match in re.finditer(r"\b[^\W\d_]{5,}\b", text, flags=re.UNICODE)
        if match.group(0).casefold() not in TYPO_EXCLUDED_WORDS
        and not match.group(0)[0].isupper()
        and _transposable_indexes(match.group(0))
    ]


def _transpose(word: str, rng: random.Random) -> str:
    indexes = _transposable_indexes(word)
    if not indexes:
        return word
    index = rng.choice(indexes)
    letters = list(word)
    letters[index], letters[index + 1] = letters[index + 1], letters[index]
    return "".join(letters)


def _transposable_indexes(word: str) -> list[int]:
    internal = [index for index in range(1, len(word) - 2) if word[index] != word[index + 1]]
    if internal:
        return internal
    return [index for index in range(len(word) - 1) if word[index] != word[index + 1]]


def _rng(seed_key: str) -> random.Random:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311
