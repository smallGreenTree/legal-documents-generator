"""Read raw YAML configuration files."""

from pathlib import Path

import yaml


def load_config(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
