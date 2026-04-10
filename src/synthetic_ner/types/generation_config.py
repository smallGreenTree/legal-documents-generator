from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    words_per_page: int
    section_weights: dict[str, dict[str, float]]
