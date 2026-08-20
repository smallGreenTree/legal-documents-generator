"""Load the standalone morphological augmentation configuration."""

from pathlib import Path
from typing import Any

from src.synthetic_ner.configuration.files import load_config
from src.synthetic_ner.core.paths import resolve_project_path
from src.synthetic_ner.types.augmentation import (
    MorphologyPromptsConfig,
    MorphologyWorkflowConfig,
)


def load_morphology_workflow_config(config_path: Path | str) -> MorphologyWorkflowConfig:
    path = Path(config_path).resolve()
    root = _mapping(load_config(path), str(path))
    raw = _mapping(root.get("augmentation"), "augmentation")
    prompts_path = resolve_project_path(
        path.parent,
        _string(raw.get("prompts_config_path"), "augmentation.prompts_config_path"),
    )
    prompts_root = _mapping(load_config(prompts_path), str(prompts_path))
    prompts = _mapping(prompts_root.get("prompts"), f"{prompts_path}.prompts")
    minimum = _ratio(raw.get("minimum_change_ratio"), "augmentation.minimum_change_ratio")
    maximum = _ratio(raw.get("maximum_change_ratio"), "augmentation.maximum_change_ratio")
    if minimum >= maximum:
        raise ValueError("augmentation.minimum_change_ratio must be less than maximum_change_ratio")
    return MorphologyWorkflowConfig(
        temperature=_number(raw.get("temperature"), "augmentation.temperature"),
        max_output_tokens=_positive_int(
            raw.get("max_output_tokens"),
            "augmentation.max_output_tokens",
        ),
        max_chunk_chars=_positive_int(
            raw.get("max_chunk_chars"),
            "augmentation.max_chunk_chars",
        ),
        minimum_change_ratio=minimum,
        maximum_change_ratio=maximum,
        prompts=MorphologyPromptsConfig(
            system=_string(prompts.get("system"), "augmentation.prompts.system"),
            user=_string(prompts.get("user"), "augmentation.prompts.user"),
        ),
        deterministic_minimum_change_ratio=_ratio(
            raw.get("deterministic_minimum_change_ratio"),
            "augmentation.deterministic_minimum_change_ratio",
        ),
        typo_rate=_ratio(raw.get("typo_rate"), "augmentation.typo_rate"),
        max_typos=_positive_int(raw.get("max_typos"), "augmentation.max_typos"),
        layout_widths=_positive_int_tuple(
            raw.get("layout_widths"),
            "augmentation.layout_widths",
        ),
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _ratio(value: Any, path: str) -> float:
    number = _number(value, path)
    if not 0 <= number <= 1:
        raise ValueError(f"{path} must be between 0 and 1")
    return number


def _positive_int_tuple(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return tuple(_positive_int(item, f"{path}[{index}]") for index, item in enumerate(value))
