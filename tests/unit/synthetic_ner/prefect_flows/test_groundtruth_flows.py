import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.prefect_flows import generation, groundtruth


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def test_generation_waits_for_groundtruth_before_audit(tmp_path, monkeypatch):
    calls = []
    context = SimpleNamespace(documents=1, output_dir=tmp_path / "output")
    doc_id = "doc1"
    document = object()
    schema = {"edges": []}

    monkeypatch.setattr(generation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(generation, "select_scenario", lambda **_kwargs: {})
    monkeypatch.setattr(generation, "ingest_configs", lambda **_kwargs: context)
    monkeypatch.setattr(generation, "_current_flow_run_id", lambda: "flow-run")
    monkeypatch.setattr(generation, "select_doc_id", lambda _context: doc_id)
    monkeypatch.setattr(generation, "resolve_entities", lambda _context: document)
    monkeypatch.setattr(
        generation,
        "build_case_schema",
        lambda *_args: (doc_id, schema),
    )
    monkeypatch.setattr(
        generation,
        "run_langgraph_mlflow",
        lambda *_args: calls.append("document"),
    )

    def complete_groundtruth(**_kwargs):
        calls.append("groundtruth")
        return {"status": "completed", "doc_id": doc_id}

    monkeypatch.setattr(generation, "generate_document_groundtruth", complete_groundtruth)
    monkeypatch.setattr(
        generation,
        "audit_created_files",
        lambda *_args: calls.append("audit"),
    )
    monkeypatch.setattr(generation, "get_run_logger", lambda: _Logger())

    result = generation.generate_dataset.fn(project_root=str(tmp_path))

    assert result == [doc_id]
    assert calls == ["document", "groundtruth", "audit"]


def test_generation_does_not_audit_incomplete_groundtruth(tmp_path, monkeypatch):
    context = SimpleNamespace(documents=1, output_dir=tmp_path / "output")
    monkeypatch.setattr(generation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(generation, "select_scenario", lambda **_kwargs: {})
    monkeypatch.setattr(generation, "ingest_configs", lambda **_kwargs: context)
    monkeypatch.setattr(generation, "_current_flow_run_id", lambda: None)
    monkeypatch.setattr(generation, "select_doc_id", lambda _context: "doc1")
    monkeypatch.setattr(generation, "resolve_entities", lambda _context: object())
    monkeypatch.setattr(
        generation,
        "build_case_schema",
        lambda *_args: ("doc1", {"edges": []}),
    )
    monkeypatch.setattr(generation, "run_langgraph_mlflow", lambda *_args: None)
    monkeypatch.setattr(
        generation,
        "generate_document_groundtruth",
        lambda **_kwargs: {"status": "failed", "doc_id": "doc1"},
    )
    monkeypatch.setattr(
        generation,
        "audit_created_files",
        lambda *_args: pytest.fail("audit must not run"),
    )

    with pytest.raises(RuntimeError, match="did not complete"):
        generation.generate_dataset.fn(project_root=str(tmp_path))


def test_manual_directory_flow_processes_every_package_before_failing(tmp_path, monkeypatch):
    for doc_id in ("doc1", "doc2", "doc3"):
        doc_dir = tmp_path / doc_id
        doc_dir.mkdir()
        (doc_dir / "document_manifest.json").write_text("{}\n", encoding="utf-8")

    processed = []

    def process_document(*, document_dir, **_kwargs):
        doc_id = Path(document_dir).name
        processed.append(doc_id)
        if doc_id == "doc2":
            raise ValueError("invalid annotations")
        return {"status": "completed", "doc_id": doc_id}

    monkeypatch.setattr(groundtruth, "generate_document_groundtruth", process_document)
    monkeypatch.setattr(groundtruth, "get_run_logger", lambda: _Logger())

    with pytest.raises(RuntimeError, match="1 of 3 failed"):
        groundtruth.generate_groundtruth_directory.fn(
            input_directory=str(tmp_path),
            project_root=str(tmp_path),
        )

    assert processed == ["doc1", "doc2", "doc3"]
    report = json.loads((tmp_path / "groundtruth_batch_report.json").read_text())
    assert report["documents_completed"] == 2
    assert report["documents_failed"] == 1
    assert [result["status"] for result in report["results"]] == [
        "completed",
        "failed",
        "completed",
    ]
