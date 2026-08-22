import pytest
from src.synthetic_ner.metadata.version import get_generator_version, get_version_provenance


def test_generator_version_comes_from_pyproject_semver(tmp_path):
    tmp_path.joinpath("pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.poetry]",
                'name = "synthetic-ner"',
                'version = "1.2.3"',
            ]
        ),
        encoding="utf-8",
    )

    assert get_generator_version(tmp_path) == "1.2.3"


def test_version_provenance_reads_version_without_manifest(tmp_path):
    tmp_path.joinpath("pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.poetry]",
                'name = "synthetic-ner"',
                'version = "1.2.3"',
            ]
        ),
        encoding="utf-8",
    )
    provenance = get_version_provenance(tmp_path)

    assert provenance["version"] == "1.2.3"
    assert set(provenance) == {"version", "git_commit", "git_branch", "git_dirty"}


def test_generator_version_rejects_non_semver(tmp_path):
    tmp_path.joinpath("pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.poetry]",
                'name = "synthetic-ner"',
                'version = "1.2"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic X.X.X"):
        get_generator_version(tmp_path)
