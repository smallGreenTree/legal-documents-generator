"""Shared utility helpers."""

import csv
from pathlib import Path

import yaml
from src.synthetic_ner.constants import (
    GROUNDTRUTH_HEADER,
    INLINE_TEMPLATE_ENV,
    TITLE_PREFIXES,
)


def load_config(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_groundtruth(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(GROUNDTRUTH_HEADER)
        for row in rows:
            writer.writerow(row)


def is_auto(value) -> bool:
    return value is None or value == "auto"


def resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def strip_titles(name: str) -> str:
    for title in TITLE_PREFIXES:
        name = name.replace(title, "").strip()
    return name


def split_address(address: str) -> tuple[str, str]:
    if not isinstance(address, str):
        return "", ""
    if ", " in address:
        return tuple(address.rsplit(", ", 1))
    if "," in address:
        return tuple(part.strip() for part in address.rsplit(",", 1))
    return address, ""


def make_initials(name: str) -> str:
    parts = [part for part in strip_titles(name).split() if part]
    if not parts:
        return ""
    return ".".join(part[0].upper() for part in parts) + "."


def render_inline_template(template: str, **context) -> str:
    return INLINE_TEMPLATE_ENV.from_string(template).render(**context)
