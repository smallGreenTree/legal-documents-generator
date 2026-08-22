"""Read semantic generator version and source provenance."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import tomllib

DEFAULT_GENERATOR_VERSION = "0.1.0"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def get_generator_version(project_root: Path | str | None = None) -> str:
    """Return the semantic generator version from pyproject.toml."""
    root = _project_root(project_root)
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return DEFAULT_GENERATOR_VERSION

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = str(data.get("tool", {}).get("poetry", {}).get("version", "")).strip()
    if not SEMVER_RE.fullmatch(version):
        raise ValueError("Generator version must use semantic X.X.X format in pyproject.toml")
    return version


def get_version_provenance(project_root: Path | str | None = None) -> dict[str, str]:
    """Return semantic version and git provenance for report stamping."""
    root = _project_root(project_root)
    git = _git_provenance(root)
    return {
        "version": get_generator_version(root),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
    }


def _project_root(project_root: Path | str | None = None) -> Path:
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]


def _git_provenance(project_root: Path) -> dict[str, str]:
    return {
        "commit": _git_value(project_root, "rev-parse", "HEAD"),
        "branch": _git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "true" if _git_value(project_root, "status", "--short") else "false",
    }


def _git_value(project_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()
