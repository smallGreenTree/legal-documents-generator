"""Resolve paths relative to the project root."""

from pathlib import Path


def resolve_project_path(project_root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate
