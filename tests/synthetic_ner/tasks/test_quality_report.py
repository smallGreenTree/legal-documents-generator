from types import SimpleNamespace

from src.synthetic_ner.tasks.quality_report import (
    build_quality_report,
    format_markdown_report,
    load_quality_scoring_config,
)


def test_quality_report_scores_existing_document_artifacts(tmp_path):
    context = SimpleNamespace(
        output_dir=tmp_path / "output",
        memory_dir=tmp_path / "memory",
        section_word_targets={"persons": 120, "history": 120},
    )
    doc_id = "en_indictment_financial_fraud_001"
    memory_dir = context.memory_dir / f"case_{doc_id}"
    memory_dir.mkdir(parents=True)
    memory_dir.joinpath("CASE_MEMORY.md").write_text(
        "\n".join(
            [
                "# CASE_MEMORY",
                "## Defendants",
                "- Philomena Hoffmann | role: chief executive officer | nationality: German",
                "## Organisations",
                "- SIMON INTERNATIONAL LTD | VAT: FRK7506155054 | address: 1 Test Street",
                "## Allowed References",
                "### Case References and Dates",
                "- Filing date: 27 September 2025",
                "### Allowed Person Surface Forms",
                "- Philomena Hoffmann | allowed forms: Philomena Hoffmann",
                "### Allowed Organisations",
                "- SIMON INTERNATIONAL LTD | VAT: FRK7506155054 | address: 1 Test Street",
            ]
        ),
        encoding="utf-8",
    )
    _write_section(
        context.output_dir,
        doc_id,
        "persons",
        "r0",
        "Philomena Hoffmann is identified in relation to SIMON INTERNATIONAL LTD. "
        * 12,
    )
    _write_section(
        context.output_dir,
        doc_id,
        "history",
        "r2",
        "On 1 January 2026, Philomena Hoffmann was linked to SIMON INTERNATIONAL LTD. "
        * 12,
    )

    report = build_quality_report(context, doc_id)
    markdown = format_markdown_report(report)

    assert report["doc_id"] == doc_id
    assert report["overall_score"] < 100
    assert report["sections"][0]["section"] == "persons"
    assert report["sections"][1]["section"] == "history"
    assert report["sections"][0]["score"] > report["sections"][1]["score"]
    assert report["sections"][1]["revision"] == 2
    assert report["sections"][1]["score_breakdown"]["issue_penalty"] > 0
    assert report["sections"][1]["score_breakdown"]["revision_penalty"] == 12
    assert "Section mentions unknown date '1 January 2026'." in report["sections"][1]["issues"]
    assert "Overall score" in markdown
    assert "## Quality Score Explanation" in markdown
    assert "Validator issues" in markdown


def test_quality_report_uses_configured_scoring_weights(tmp_path):
    config_path = tmp_path / "config_quality.yaml"
    config_path.write_text(
        "\n".join(
            [
                "quality_scoring:",
                "  validator_issue_penalty: 20",
                "  validator_issue_penalty_cap: 80",
                "  revision_penalty: 1",
                "  revision_penalty_cap: 3",
                "  short_section_penalty: 5",
                "  short_section_word_threshold: 20",
            ]
        ),
        encoding="utf-8",
    )

    scoring = load_quality_scoring_config(config_path)

    assert scoring["validator_issue_penalty"] == 20
    assert scoring["revision_penalty"] == 1


def _write_section(output_dir, doc_id, section, revision, text):
    section_dir = output_dir / "_partial" / doc_id / "sections" / section / revision
    section_dir.mkdir(parents=True)
    section_dir.joinpath("combined.txt").write_text(text, encoding="utf-8")
