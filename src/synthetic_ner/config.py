"""Typed helpers for config.yaml sections."""

from __future__ import annotations

from src.synthetic_ner.types.generation_config import GenerationConfig


def build_generation_config(cfg: dict) -> GenerationConfig:
    generation_cfg = cfg.get("generation")
    if not isinstance(generation_cfg, dict):
        raise ValueError("Top-level generation section must be a mapping")

    words_per_page = generation_cfg.get("words_per_page")
    if not isinstance(words_per_page, int) or words_per_page <= 0:
        raise ValueError("generation.words_per_page must be a positive integer")

    raw_section_weights = generation_cfg.get("section_weights")
    if not isinstance(raw_section_weights, dict) or not raw_section_weights:
        raise ValueError("generation.section_weights must be a non-empty mapping")

    section_weights: dict[str, dict[str, float]] = {}
    for doc_type, section_map in raw_section_weights.items():
        if not isinstance(doc_type, str) or not doc_type.strip():
            raise ValueError("generation.section_weights keys must be non-empty strings")
        if not isinstance(section_map, dict) or not section_map:
            raise ValueError(
                f"generation.section_weights.{doc_type} must be a non-empty mapping"
            )

        normalized_sections: dict[str, float] = {}
        for section_name, weight in section_map.items():
            if not isinstance(section_name, str) or not section_name.strip():
                raise ValueError(
                    f"generation.section_weights.{doc_type} contains an empty section name"
                )
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                raise ValueError(
                    f"generation.section_weights.{doc_type}.{section_name} "
                    "must be a positive number"
                )
            normalized_sections[section_name] = float(weight)

        section_weights[doc_type] = normalized_sections

    return GenerationConfig(
        words_per_page=words_per_page,
        section_weights=section_weights,
    )


def resolve_doc_types(generation_cfg: GenerationConfig) -> tuple[str, ...]:
    return tuple(generation_cfg.section_weights.keys())


def resolve_section_order(generation_cfg: GenerationConfig, doc_type: str) -> list[str]:
    section_weights = generation_cfg.section_weights.get(doc_type)
    if section_weights is None:
        available = ", ".join(sorted(generation_cfg.section_weights))
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Available configured doc types: {available}"
        )
    return list(section_weights.keys())
