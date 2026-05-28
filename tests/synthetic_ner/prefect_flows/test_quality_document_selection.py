from types import SimpleNamespace

import pytest
from src.synthetic_ner.prefect_flows.utils import (
    _ensure_quality_document_exists,
    _quality_candidate_markdown_table,
    _quality_document_candidates,
)


def test_quality_document_candidates_list_generated_artifacts(tmp_path):
    context = SimpleNamespace(
        doc_type="indictment",
        fraud_type="financial_fraud",
        output_dir=tmp_path / "output",
        memory_dir=tmp_path / "memory",
        schema_dir=tmp_path / "schemas",
    )
    doc_003 = "en_indictment_financial_fraud_003"
    doc_004 = "en_indictment_financial_fraud_004"
    other_doc = "en_witness_statement_financial_fraud_001"

    _write_document_artifacts(context, doc_003, sections=("persons",))
    _write_document_artifacts(context, doc_004, sections=("persons", "history"))
    _write_document_artifacts(context, other_doc, sections=("persons",))

    candidates = _quality_document_candidates(context)

    assert [candidate["doc_id"] for candidate in candidates] == [doc_004, doc_003]
    assert candidates[0]["final_document"] is True
    assert candidates[0]["generation_report"] is True
    assert candidates[0]["case_memory"] is True
    assert candidates[0]["schema"] is True
    assert candidates[0]["section_artifacts"] == 2


def test_quality_candidate_markdown_table_shows_doc_ids(tmp_path):
    context = SimpleNamespace(
        doc_type="indictment",
        fraud_type="financial_fraud",
        output_dir=tmp_path / "output",
        memory_dir=tmp_path / "memory",
        schema_dir=tmp_path / "schemas",
    )
    doc_id = "en_indictment_financial_fraud_003"
    _write_document_artifacts(context, doc_id, sections=("persons",))

    markdown = _quality_candidate_markdown_table(_quality_document_candidates(context))

    assert "| Document ID | Final | Report | Memory | Schema | Sections | Modified |" in markdown
    assert f"`{doc_id}`" in markdown


def test_ensure_quality_document_exists_rejects_unknown_doc_id(tmp_path):
    context = SimpleNamespace(
        doc_type="indictment",
        fraud_type="financial_fraud",
        output_dir=tmp_path / "output",
        memory_dir=tmp_path / "memory",
        schema_dir=tmp_path / "schemas",
    )
    _write_document_artifacts(
        context,
        "en_indictment_financial_fraud_003",
        sections=("persons",),
    )

    with pytest.raises(SystemExit, match="Unknown document id"):
        _ensure_quality_document_exists(context, "en_indictment_financial_fraud_004")


def _write_document_artifacts(context, doc_id, sections):
    doc_dir = context.output_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.joinpath(f"{doc_id}.txt").write_text("document", encoding="utf-8")
    doc_dir.joinpath("generation_report.md").write_text("report", encoding="utf-8")

    memory_dir = context.memory_dir / f"case_{doc_id}"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.joinpath("CASE_MEMORY.md").write_text("memory", encoding="utf-8")

    context.schema_dir.mkdir(parents=True, exist_ok=True)
    context.schema_dir.joinpath(f"{doc_id}.json").write_text("{}", encoding="utf-8")

    for section in sections:
        section_dir = context.output_dir / "_partial" / doc_id / "sections" / section
        section_dir.mkdir(parents=True, exist_ok=True)
