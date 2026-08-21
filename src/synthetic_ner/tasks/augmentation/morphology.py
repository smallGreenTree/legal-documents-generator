"""LLM-backed controlled morphology transformation."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.synthetic_ner.tasks.augmentation.constants import (
    DETERMINISTIC_TRANSFORMATIONS,
    MORPHOLOGY_MODEL_STAGE,
    PROTECTED_TOKEN_PATTERN,
    TRANSFORMATION_INSTRUCTIONS,
)
from src.synthetic_ner.tasks.augmentation.noise import (
    apply_intentional_typos,
    apply_random_layout,
)
from src.synthetic_ner.tasks.augmentation.protection import (
    protect_document_text,
    reconstruct_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.style import (
    custom_style_instruction,
    normalize_style,
    normalize_style_temperature,
    style_reformatting_instruction,
)
from src.synthetic_ner.text.templates import render_prompt_template
from src.synthetic_ner.types.augmentation import (
    MorphologyError,
    MorphologySource,
    MorphologyTransformation,
    MorphologyVariant,
    MorphologyWorkflowConfig,
)


class MorphologyAugmenter:
    """Apply one named transformation while preserving protected values."""

    def __init__(self, *, client, config: MorphologyWorkflowConfig) -> None:
        self.client = client
        self.config = config
        self._style_fallback_chunk_indices: list[int] = []
        self._protected_entity_surfaces: tuple[str, ...] = ()

    @property
    def style_fallback_chunk_indices(self) -> tuple[int, ...]:
        return tuple(self._style_fallback_chunk_indices)

    def create_variant(
        self,
        *,
        source: MorphologySource,
        transformation: MorphologyTransformation,
        variant_doc_id: str,
        contract_path: Path | str,
        style: str | None = None,
        style_temperature: float | None = None,
        reformat_with_style: bool = False,
    ) -> MorphologyVariant:
        self._style_fallback_chunk_indices = []
        normalized_style = (
            normalize_style(style)
            if transformation is MorphologyTransformation.CUSTOM_STYLE
            else None
        )
        effective_style_temperature = (
            normalize_style_temperature(
                self.config.style_temperature if style_temperature is None else style_temperature
            )
            if transformation is MorphologyTransformation.CUSTOM_STYLE
            else None
        )
        effective_reformat = bool(reformat_with_style) and (
            transformation is MorphologyTransformation.CUSTOM_STYLE
        )
        protected = protect_document_text(source.text, source.annotations)
        self._protected_entity_surfaces = tuple(
            sorted(
                {
                    annotation.entity_text
                    for mention in protected.mentions
                    for annotation in mention.annotations
                },
                key=lambda value: (-len(value), value),
            )
        )
        transformed = self._create_transformed_text(
            variant_doc_id=variant_doc_id,
            protected_text=protected.text,
            transformation=transformation,
            style=normalized_style,
            style_temperature=effective_style_temperature,
            reformat_with_style=effective_reformat,
            transformation_instruction=_transformation_instruction(
                transformation,
                normalized_style,
            ),
        )
        minimum_change_ratio = (
            self.config.deterministic_minimum_change_ratio
            if transformation in DETERMINISTIC_TRANSFORMATIONS
            else self.config.minimum_change_ratio
        )
        maximum_change_ratio = (
            self.config.style_maximum_change_ratio
            if transformation is MorphologyTransformation.CUSTOM_STYLE
            else self.config.maximum_change_ratio
        )
        return reconstruct_morphology_variant(
            source_text=source.text,
            protected=protected,
            transformed_text=transformed,
            variant_doc_id=variant_doc_id,
            transformation=transformation,
            minimum_change_ratio=minimum_change_ratio,
            maximum_change_ratio=maximum_change_ratio,
            contract_path=contract_path,
            style=normalized_style,
            style_temperature=effective_style_temperature,
            reformat_with_style=effective_reformat,
            style_fallback_chunk_indices=self.style_fallback_chunk_indices,
        )

    def _create_transformed_text(
        self,
        *,
        variant_doc_id: str,
        protected_text: str,
        transformation: MorphologyTransformation,
        style: str | None,
        style_temperature: float | None,
        reformat_with_style: bool,
        transformation_instruction: str,
    ) -> str:
        if transformation is MorphologyTransformation.INTENTIONAL_TYPOS:
            return apply_intentional_typos(
                protected_text,
                seed_key=variant_doc_id,
                typo_rate=self.config.typo_rate,
                max_typos=self.config.max_typos,
            )
        if transformation is MorphologyTransformation.RANDOM_LAYOUT:
            return apply_random_layout(
                protected_text,
                seed_key=variant_doc_id,
                widths=self.config.layout_widths,
            )
        if self.client is None:
            raise MorphologyError("Morphology model client is required for this transformation")
        return self._transform_text(
            doc_id=variant_doc_id,
            protected_text=protected_text,
            transformation=transformation,
            style=style,
            style_temperature=style_temperature,
            reformat_with_style=reformat_with_style,
            transformation_instruction=transformation_instruction,
        )

    def _transform_text(
        self,
        *,
        doc_id: str,
        protected_text: str,
        transformation: MorphologyTransformation,
        style: str | None,
        style_temperature: float | None,
        reformat_with_style: bool,
        transformation_instruction: str,
    ) -> str:
        if transformation is MorphologyTransformation.CUSTOM_STYLE:
            return self._transform_style_text(
                doc_id=doc_id,
                protected_text=protected_text,
                transformation=transformation,
                style=style,
                style_temperature=style_temperature,
                reformat_with_style=reformat_with_style,
                transformation_instruction=transformation_instruction,
            )

        output: list[str] = []
        chunk_index = 0
        for block in re.split(r"(\n{2,})", protected_text):
            if not block or re.fullmatch(r"\n{2,}", block):
                output.append(block)
                continue
            if not _is_transformable(block):
                output.append(block)
                continue
            transformed_chunks = []
            for chunk in _split_chunk(block, self.config.max_chunk_chars):
                chunk_index += 1
                transformed_chunks.append(
                    self._transform_chunk(
                        doc_id=doc_id,
                        chunk=chunk,
                        chunk_index=chunk_index,
                        transformation=transformation,
                        style=style,
                        style_temperature=style_temperature,
                        reformat_with_style=False,
                        transformation_instruction=transformation_instruction,
                    )
                )
            output.append(" ".join(transformed_chunks))
        transformed_text = "".join(output)
        if protected_text.endswith("\n") and not transformed_text.endswith("\n"):
            transformed_text += "\n"
        return transformed_text

    def _transform_style_text(
        self,
        *,
        doc_id: str,
        protected_text: str,
        transformation: MorphologyTransformation,
        style: str | None,
        style_temperature: float | None,
        reformat_with_style: bool,
        transformation_instruction: str,
    ) -> str:
        output: list[str] = []
        chunk_index = 0
        for block in re.split(r"(\n{2,})", protected_text):
            if not block or re.fullmatch(r"\n{2,}", block):
                output.append(block)
                continue
            if not _is_style_prose(block):
                output.append(block)
                continue
            transformed_chunks = []
            for chunk in _split_style_block(
                block,
                maximum_chars=self.config.style_max_chunk_chars,
                maximum_tokens=self.config.style_max_protected_tokens,
                maximum_sentences=self.config.style_max_sentences_per_chunk,
            ):
                chunk_index += 1
                transformed_chunks.append(
                    self._transform_chunk(
                        doc_id=doc_id,
                        chunk=chunk,
                        chunk_index=chunk_index,
                        transformation=transformation,
                        style=style,
                        style_temperature=style_temperature,
                        reformat_with_style=reformat_with_style,
                        transformation_instruction=transformation_instruction,
                    )
                )
            output.append(("\n" if reformat_with_style else " ").join(transformed_chunks))
        transformed_text = "".join(output)
        if protected_text.endswith("\n") and not transformed_text.endswith("\n"):
            transformed_text += "\n"
        return transformed_text

    def _transform_chunk(
        self,
        *,
        doc_id: str,
        chunk: str,
        chunk_index: int,
        transformation: MorphologyTransformation,
        style: str | None,
        style_temperature: float | None,
        reformat_with_style: bool,
        transformation_instruction: str,
    ) -> str:
        if transformation is MorphologyTransformation.CUSTOM_STYLE:
            original_prompt = render_prompt_template(
                self.config.prompts.style_user,
                requested_style=style,
                reformatting_instruction=style_reformatting_instruction(reformat_with_style),
                protected_text=chunk,
            )
            system_prompt = self.config.prompts.style_system
            if style_temperature is None:
                raise MorphologyError("Style temperature is required for custom style")
            temperature = style_temperature
        else:
            original_prompt = render_prompt_template(
                self.config.prompts.user,
                transformation_instruction=transformation_instruction,
                protected_text=chunk,
            )
            system_prompt = self.config.prompts.system
            temperature = self.config.temperature
        user_prompt = original_prompt
        validation_error = ""
        for attempt in range(1, self.config.max_chunk_attempts + 1):
            task_id = f"morphology_{transformation.value}_{chunk_index:03d}"
            if attempt > 1:
                task_id += f"_retry_{attempt:02d}"
            result = self.client.invoke(
                doc_id=doc_id,
                task_id=task_id,
                stage=MORPHOLOGY_MODEL_STAGE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            transformed = _strip_fence(result.text)
            validation_error = _chunk_validation_error(
                chunk,
                transformed,
                preserve_token_order=(transformation is MorphologyTransformation.CUSTOM_STYLE),
                protected_entity_surfaces=self._protected_entity_surfaces,
            )
            if not validation_error:
                return transformed
            if transformation is MorphologyTransformation.CUSTOM_STYLE:
                user_prompt = render_prompt_template(
                    self.config.prompts.style_retry,
                    requested_style=style,
                    validation_error=validation_error,
                    expected_tokens=" -> ".join(re.findall(PROTECTED_TOKEN_PATTERN, chunk))
                    or "none",
                    protected_text=chunk,
                )
                temperature = normalize_style_temperature(self.config.style_retry_temperature)
            else:
                user_prompt = render_prompt_template(
                    self.config.prompts.retry,
                    original_prompt=original_prompt,
                    validation_error=validation_error,
                    previous_text=transformed,
                )
        self._style_fallback_chunk_indices.append(chunk_index)
        return chunk


def _transformation_instruction(
    transformation: MorphologyTransformation,
    style: str | None,
) -> str:
    if transformation is MorphologyTransformation.CUSTOM_STYLE:
        return custom_style_instruction(style)
    return TRANSFORMATION_INSTRUCTIONS[transformation]


def _is_transformable(text: str) -> bool:
    words = re.findall(r"\b[^\W\d_]+\b", text, flags=re.UNICODE)
    return len(words) >= 4 and not (len(text) < 200 and text.strip().isupper())


def _is_style_prose(text: str) -> bool:
    return _is_transformable(text) and bool(re.search(r"[.!?](?:\s|$)", text))


def _split_style_block(
    text: str,
    *,
    maximum_chars: int,
    maximum_tokens: int,
    maximum_sentences: int,
) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(re.findall(PROTECTED_TOKEN_PATTERN, sentence))
        candidate = " ".join((*current, sentence))
        exceeds_limit = current and (
            len(current) >= maximum_sentences
            or current_tokens + sentence_tokens > maximum_tokens
            or len(candidate) > maximum_chars
        )
        if exceeds_limit:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_chunk(text: str, maximum_chars: int) -> list[str]:
    if len(text) <= maximum_chars:
        return [text]
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum_chars:
        cut = remaining.rfind(" ", 0, maximum_chars + 1)
        if cut <= 0:
            cut = maximum_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _chunk_validation_error(
    source: str,
    transformed: str,
    *,
    preserve_token_order: bool = False,
    protected_entity_surfaces: tuple[str, ...] = (),
) -> str:
    if not transformed:
        return "model returned empty text"
    expected = Counter(re.findall(PROTECTED_TOKEN_PATTERN, source))
    actual = Counter(re.findall(PROTECTED_TOKEN_PATTERN, transformed))
    if actual == expected:
        if preserve_token_order and re.findall(PROTECTED_TOKEN_PATTERN, transformed) != re.findall(
            PROTECTED_TOKEN_PATTERN, source
        ):
            return "protected token order changed"
        leaked_occurrences = sum(
            _count_occurrences(transformed, surface) for surface in protected_entity_surfaces
        )
        if leaked_occurrences:
            return (
                f"model reproduced {leaked_occurrences} protected entity occurrence(s) "
                "outside protected tokens"
            )
        return ""
    missing = _format_token_difference(expected - actual)
    extra = _format_token_difference(actual - expected)
    return (
        f"protected token inventory changed (missing: {missing}; duplicated or unexpected: {extra})"
    )


def _format_token_difference(tokens: Counter[str]) -> str:
    if not tokens:
        return "none"
    return ", ".join(
        token if count == 1 else f"{token} x{count}" for token, count in sorted(tokens.items())
    )


def _count_occurrences(text: str, value: str) -> int:
    count = 0
    start = text.find(value)
    while start != -1:
        count += 1
        start = text.find(value, start + 1)
    return count
