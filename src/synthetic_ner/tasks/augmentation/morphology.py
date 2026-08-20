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

    def create_variant(
        self,
        *,
        source: MorphologySource,
        transformation: MorphologyTransformation,
        variant_doc_id: str,
        contract_path: Path | str,
    ) -> MorphologyVariant:
        protected = protect_document_text(source.text, source.annotations)
        transformed = self._create_transformed_text(
            variant_doc_id=variant_doc_id,
            protected_text=protected.text,
            transformation=transformation,
        )
        minimum_change_ratio = (
            self.config.deterministic_minimum_change_ratio
            if transformation in DETERMINISTIC_TRANSFORMATIONS
            else self.config.minimum_change_ratio
        )
        return reconstruct_morphology_variant(
            source_text=source.text,
            protected=protected,
            transformed_text=transformed,
            variant_doc_id=variant_doc_id,
            transformation=transformation,
            minimum_change_ratio=minimum_change_ratio,
            maximum_change_ratio=self.config.maximum_change_ratio,
            contract_path=contract_path,
        )

    def _create_transformed_text(
        self,
        *,
        variant_doc_id: str,
        protected_text: str,
        transformation: MorphologyTransformation,
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
        )

    def _transform_text(
        self,
        *,
        doc_id: str,
        protected_text: str,
        transformation: MorphologyTransformation,
    ) -> str:
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
                    )
                )
            output.append(" ".join(transformed_chunks))
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
    ) -> str:
        user_prompt = render_prompt_template(
            self.config.prompts.user,
            transformation_instruction=TRANSFORMATION_INSTRUCTIONS[transformation],
            protected_text=chunk,
        )
        result = self.client.invoke(
            doc_id=doc_id,
            task_id=f"morphology_{transformation.value}_{chunk_index:03d}",
            stage=MORPHOLOGY_MODEL_STAGE,
            system_prompt=self.config.prompts.system,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        transformed = _strip_fence(result.text)
        if not transformed:
            raise MorphologyError("Morphology model returned empty text")
        _validate_chunk_tokens(chunk, transformed)
        return transformed


def _is_transformable(text: str) -> bool:
    words = re.findall(r"\b[^\W\d_]+\b", text, flags=re.UNICODE)
    return len(words) >= 4 and not (len(text) < 200 and text.strip().isupper())


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


def _validate_chunk_tokens(source: str, transformed: str) -> None:
    expected = Counter(re.findall(PROTECTED_TOKEN_PATTERN, source))
    actual = Counter(re.findall(PROTECTED_TOKEN_PATTERN, transformed))
    if actual != expected:
        raise MorphologyError(
            "Morphology model changed the protected token inventory for a text chunk"
        )
