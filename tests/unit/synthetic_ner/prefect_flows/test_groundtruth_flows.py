import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.prefect_flows import generation, groundtruth

PROJECT_ROOT = Path(__file__).resolve().parents[4]


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

    monkeypatch.setattr(generation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(generation, "select_scenario", lambda **_kwargs: {})
    monkeypatch.setattr(generation, "ingest_configs", lambda **_kwargs: context)
    monkeypatch.setattr(generation, "_current_flow_run_id", lambda: "flow-run")
    monkeypatch.setattr(generation, "select_doc_id", lambda _context: doc_id)
    monkeypatch.setattr(generation, "resolve_entities", lambda _context: document)
    monkeypatch.setattr(
        generation,
        "save_resolved_entities",
        lambda *_args: calls.append("entities-saved"),
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
    assert calls == ["entities-saved", "document", "groundtruth", "audit"]


def test_generation_does_not_audit_incomplete_groundtruth(tmp_path, monkeypatch):
    context = SimpleNamespace(documents=1, output_dir=tmp_path / "output")
    monkeypatch.setattr(generation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(generation, "select_scenario", lambda **_kwargs: {})
    monkeypatch.setattr(generation, "ingest_configs", lambda **_kwargs: context)
    monkeypatch.setattr(generation, "_current_flow_run_id", lambda: None)
    monkeypatch.setattr(generation, "select_doc_id", lambda _context: "doc1")
    monkeypatch.setattr(generation, "resolve_entities", lambda _context: object())
    monkeypatch.setattr(generation, "save_resolved_entities", lambda *_args: None)
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


def test_document_groundtruth_flow_runs_modular_tasks_in_order(tmp_path, monkeypatch):
    calls = []
    source = {
        "document_dir": str(tmp_path / "doc1"),
        "doc_id": "doc1",
        "document_text": "Alice",
    }
    references = [{"entity_text": "Alice", "label": "PERSON"}]
    annotations = [
        {
            "annotation_id": "doc1-001",
            "doc_id": "doc1",
            "entity_text": "Alice",
            "label": "PERSON",
            "start_char": 0,
            "end_char": 5,
        }
    ]
    monkeypatch.setattr(groundtruth, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(
        groundtruth,
        "load_frozen_groundtruth_inputs",
        lambda _path: calls.append("load") or source,
    )
    monkeypatch.setattr(
        groundtruth,
        "select_used_groundtruth_entities",
        lambda _source: calls.append("select") or references,
    )
    monkeypatch.setattr(
        groundtruth,
        "calculate_groundtruth_annotations",
        lambda **_kwargs: calls.append("offsets") or annotations,
    )
    monkeypatch.setattr(
        groundtruth,
        "publish_validated_groundtruth",
        lambda **_kwargs: calls.append("publish") or {"status": "completed", "doc_id": "doc1"},
    )

    result = groundtruth.generate_document_groundtruth.fn(
        document_dir=str(tmp_path / "doc1"),
        project_root=str(tmp_path),
    )

    assert result == {"status": "completed", "doc_id": "doc1"}
    assert calls == ["load", "select", "offsets", "publish"]


def test_modular_groundtruth_tasks_create_validated_tsv(tmp_path, monkeypatch):
    doc_dir = tmp_path / "doc1"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Alice appeared. Alice left.\n", encoding="utf-8")
    (doc_dir / "document_inputs.json").write_text(
        json.dumps(
            {
                "defendants": [{"name": "Alice"}],
                "collateral": [],
                "charged_orgs": [],
                "associated_orgs": [],
                "metadata": {},
                "amounts": {},
                "counts_list": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(groundtruth, "get_run_logger", lambda: _Logger())

    source = groundtruth.load_frozen_groundtruth_inputs.fn(str(doc_dir))
    references = groundtruth.select_used_groundtruth_entities.fn(source)
    annotations = groundtruth.calculate_groundtruth_annotations.fn(
        source=source,
        references=references,
        contract_path=str(PROJECT_ROOT / "groundtruth_contract.yaml"),
    )
    result = groundtruth.publish_validated_groundtruth.fn(
        source=source,
        references=references,
        annotation_rows=annotations,
        contract_path=str(PROJECT_ROOT / "groundtruth_contract.yaml"),
    )

    assert result["status"] == "completed"
    assert [row["start_char"] for row in annotations] == [0, 16]
    assert (doc_dir / "groundtruth.tsv").is_file()
    assert (doc_dir / "groundtruth_manifest.json").is_file()


def test_manual_directory_flow_processes_every_package_before_failing(tmp_path, monkeypatch):
    for doc_id in ("doc1", "doc2", "doc3"):
        doc_dir = tmp_path / doc_id
        doc_dir.mkdir()
        (doc_dir / f"{doc_id}.txt").write_text("document\n", encoding="utf-8")
        (doc_dir / "document_inputs.json").write_text(
            json.dumps(
                {
                    "defendants": [],
                    "collateral": [],
                    "charged_orgs": [],
                    "associated_orgs": [],
                    "metadata": {},
                    "amounts": {},
                    "counts_list": [],
                }
            ),
            encoding="utf-8",
        )

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
