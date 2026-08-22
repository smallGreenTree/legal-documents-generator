from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "synthetic_ner"


def test_package_root_contains_only_stable_entry_points():
    root_modules = {path.name for path in PACKAGE_ROOT.glob("*.py")}

    assert root_modules == {"__init__.py", "cli.py"}


def test_root_responsibilities_have_named_packages():
    package_names = {
        "case_generation",
        "configuration",
        "core",
        "document",
        "integrations",
        "metadata",
        "text",
    }

    assert all(
        (PACKAGE_ROOT / package_name / "__init__.py").is_file() for package_name in package_names
    )
