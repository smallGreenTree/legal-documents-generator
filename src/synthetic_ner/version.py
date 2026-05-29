"""Semantic version helpers for generator provenance."""

from __future__ import annotations

import re
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

import tomllib
import yaml

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
        raise ValueError(
            "Generator version must use semantic X.X.X format in pyproject.toml"
        )
    return version


def get_version_provenance(project_root: Path | str | None = None) -> dict[str, Any]:
    """Return version manifest and git provenance for report stamping."""
    root = _project_root(project_root)
    version = get_generator_version(root)
    manifest_path = root / "generator_versions.yaml"
    manifest = _load_version_manifest(manifest_path)
    version_record = manifest.get("versions", {}).get(version, {})
    if not version_record:
        version_record = {
            "git_tag": f"generator-v{version}",
            "summary": "Version is not listed in generator_versions.yaml.",
            "features": [],
            "report_schema_version": "unknown",
        }
    git = _git_provenance(root)
    return {
        "version": version,
        "git_tag": version_record.get("git_tag") or f"generator-v{version}",
        "summary": version_record.get("summary") or "n/a",
        "features": _string_list(version_record.get("features")),
        "report_schema_version": version_record.get("report_schema_version") or "unknown",
        "manifest_hash": _manifest_hash(manifest_path),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
    }


def _project_root(project_root: Path | str | None = None) -> Path:
    return Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]


def _load_version_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _manifest_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"sha256:{sha256(path.read_bytes()).hexdigest()}"


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
