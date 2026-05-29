import pytest
from src.synthetic_ner.version import get_generator_version, get_version_provenance


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


def test_version_provenance_reads_manifest_and_hash(tmp_path):
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
    tmp_path.joinpath("generator_versions.yaml").write_text(
        "\n".join(
            [
                "versions:",
                '  "1.2.3":',
                '    git_tag: "generator-v1.2.3"',
                '    report_schema_version: "2.0.0"',
                '    summary: "Test release"',
                "    features:",
                '      - "Feature A"',
            ]
        ),
        encoding="utf-8",
    )

    provenance = get_version_provenance(tmp_path)

    assert provenance["version"] == "1.2.3"
    assert provenance["git_tag"] == "generator-v1.2.3"
    assert provenance["summary"] == "Test release"
    assert provenance["features"] == ["Feature A"]
    assert provenance["report_schema_version"] == "2.0.0"
    assert provenance["manifest_hash"].startswith("sha256:")


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
