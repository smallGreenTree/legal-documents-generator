from types import SimpleNamespace

import pytest
from src.synthetic_ner.prefect_flows import augmentation
from src.synthetic_ner.types.augmentation import MorphologyTransformation


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def test_flow_creates_one_variant_for_each_selected_checkbox(tmp_path, monkeypatch):
    source = SimpleNamespace(doc_id="doc1")
    calls = []
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(augmentation, "discover_morphology_documents", lambda _path: [source])
    monkeypatch.setattr(
        augmentation,
        "create_morphology_variant",
        lambda **kwargs: (
            calls.append((kwargs["source"].doc_id, kwargs["transformation"]))
            or {"status": "completed", "variant_doc_id": str(kwargs["transformation"])}
        ),
    )
    monkeypatch.setattr(augmentation, "publish_morphology_batch_report", lambda **kwargs: kwargs)
    monkeypatch.setattr(augmentation, "get_run_logger", lambda: _Logger())

    result = augmentation.generate_morphological_variations.fn(
        input_path="output/doc1",
        project_root=str(tmp_path),
        review=False,
        active_to_passive=True,
        verbal_to_nominal=False,
        possessive_reframe=True,
        intentional_typos=True,
        random_layout=False,
    )

    assert calls == [
        ("doc1", MorphologyTransformation.ACTIVE_TO_PASSIVE),
        ("doc1", MorphologyTransformation.POSSESSIVE_REFRAME),
        ("doc1", MorphologyTransformation.INTENTIONAL_TYPOS),
    ]
    assert len(result["results"]) == 3


def test_flow_rejects_empty_transformation_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        augmentation.generate_morphological_variations.fn(
            input_path="output/doc1",
            project_root=str(tmp_path),
            review=False,
            active_to_passive=False,
            verbal_to_nominal=False,
            possessive_reframe=False,
            intentional_typos=False,
            random_layout=False,
        )


def test_flow_processes_remaining_documents_and_reports_failures(tmp_path, monkeypatch):
    sources = [SimpleNamespace(doc_id="doc1"), SimpleNamespace(doc_id="doc2")]
    processed = []
    captured_report = {}
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(augmentation, "discover_morphology_documents", lambda _path: sources)

    def create(**kwargs):
        doc_id = kwargs["source"].doc_id
        processed.append(doc_id)
        if doc_id == "doc1":
            raise ValueError("unsafe variation")
        return {"status": "completed", "variant_doc_id": f"{doc_id}-variant"}

    def report(**kwargs):
        captured_report.update(kwargs)
        return {**kwargs, "report_path": str(tmp_path / "report.json")}

    monkeypatch.setattr(augmentation, "create_morphology_variant", create)
    monkeypatch.setattr(augmentation, "publish_morphology_batch_report", report)
    monkeypatch.setattr(augmentation, "get_run_logger", lambda: _Logger())

    with pytest.raises(RuntimeError, match="1 of 2"):
        augmentation.generate_morphological_variations.fn(
            input_path="output",
            project_root=str(tmp_path),
            review=False,
            active_to_passive=True,
            verbal_to_nominal=False,
            possessive_reframe=False,
        )

    assert processed == ["doc1", "doc2"]
    assert [result["status"] for result in captured_report["results"]] == [
        "failed",
        "completed",
    ]
