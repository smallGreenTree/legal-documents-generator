from types import SimpleNamespace

import pytest
from src.synthetic_ner.prefect_flows import augmentation
from src.synthetic_ner.types.augmentation import (
    MorphologyReviewInput,
    MorphologyTransformation,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def test_flow_creates_one_variant_for_each_selected_checkbox(tmp_path, monkeypatch):
    source = SimpleNamespace(doc_id="doc1")
    calls = []
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(
        augmentation,
        "discover_morphology_documents",
        lambda _path, _contract: [source],
    )
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
            style="",
        )


def test_flow_creates_custom_style_variant_without_checkbox_selection(tmp_path, monkeypatch):
    source = SimpleNamespace(doc_id="doc1")
    calls = []
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(
        augmentation,
        "discover_morphology_documents",
        lambda _path, _contract: [source],
    )
    monkeypatch.setattr(
        augmentation,
        "create_morphology_variant",
        lambda **kwargs: (
            calls.append(
                (
                    kwargs["transformation"],
                    kwargs["style"],
                    kwargs["style_temperature"],
                    kwargs["reformat_with_style"],
                )
            )
            or {"status": "completed", "variant_doc_id": "styled-doc1"}
        ),
    )
    monkeypatch.setattr(augmentation, "publish_morphology_batch_report", lambda **kwargs: kwargs)
    monkeypatch.setattr(augmentation, "get_run_logger", lambda: _Logger())

    result = augmentation.generate_morphological_variations.fn(
        input_path="output/doc1",
        project_root=str(tmp_path),
        review=False,
        active_to_passive=False,
        verbal_to_nominal=False,
        possessive_reframe=False,
        intentional_typos=False,
        random_layout=False,
        style="poetic legal prose",
        style_temperature=1.1,
        reformat_with_style=True,
    )

    assert calls == [
        (
            MorphologyTransformation.CUSTOM_STYLE,
            "poetic legal prose",
            1.1,
            True,
        )
    ]
    assert len(result["results"]) == 1


def test_review_form_makes_style_controls_prominent_and_bounded():
    input_model = augmentation._required_prefilled_input_model(
        MorphologyReviewInput,
        description="test",
        input_path="output/doc1",
        style="gritty rap verse with end-rhyming couplets",
        style_temperature=0.8,
        reformat_with_style=True,
        active_to_passive=False,
        verbal_to_nominal=False,
        possessive_reframe=False,
        intentional_typos=False,
        random_layout=False,
    )

    properties = input_model.model_json_schema()["properties"]

    assert properties["style"]["title"] == "CUSTOM STYLE REQUEST"
    assert "primary creative instruction" in properties["style"]["description"].lower()
    assert properties["style_temperature"] == {
        "default": 0.8,
        "description": (
            "Creativity for the custom-style rewrite only: lower is steadier; "
            "higher is more adventurous."
        ),
        "maximum": 1.5,
        "minimum": 0.0,
        "multipleOf": 0.1,
        "position": 2,
        "title": "Style temperature (0.0-1.5)",
        "type": "number",
    }
    assert properties["reformat_with_style"]["default"] is True
    assert properties["reformat_with_style"]["position"] == 3


def test_flow_processes_remaining_documents_and_reports_failures(tmp_path, monkeypatch):
    sources = [SimpleNamespace(doc_id="doc1"), SimpleNamespace(doc_id="doc2")]
    processed = []
    captured_report = {}
    monkeypatch.setattr(augmentation, "resolve_flow_project_root", lambda _value: tmp_path)
    monkeypatch.setattr(
        augmentation,
        "discover_morphology_documents",
        lambda _path, _contract: sources,
    )

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
