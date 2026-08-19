import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.engine import save_document_artifacts
from src.synthetic_ner.tasks.groundtruth import (
    GROUNDTRUTH_HEADER,
    GroundTruthError,
    MentionAnnotation,
    build_entity_references,
    build_mention_annotations,
    discover_document_packages,
    generate_groundtruth_for_document,
    load_groundtruth_contract,
    read_groundtruth_tsv,
    validate_mention_annotations,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = PROJECT_ROOT / "groundtruth_contract.yaml"


def test_repeated_entity_occurrences_have_unique_ids_and_offsets():
    document_text = "Alice signed the contract. Alice then left."
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [{"entity_text": "Alice", "label": "PERSON"}]

    annotations = build_mention_annotations(
        doc_id="doc1",
        document_text=document_text,
        references=references,
        contract=contract,
    )

    assert annotations == [
        MentionAnnotation("doc1-001", "doc1", "Alice", "PERSON", 0, 5),
        MentionAnnotation("doc1-002", "doc1", "Alice", "PERSON", 27, 32),
    ]


def test_unicode_and_multiline_offsets_use_python_string_indexes():
    document_text = "Élodie signed.\nOn line two, Élodie paid €10."
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [
        {"entity_text": "Élodie", "label": "PERSON"},
        {"entity_text": "€10", "label": "AMOUNT"},
    ]

    annotations = build_mention_annotations(
        doc_id="unicode-doc",
        document_text=document_text,
        references=references,
        contract=contract,
    )

    assert [(row.entity_text, row.start_char, row.end_char) for row in annotations] == [
        ("Élodie", 0, 6),
        ("Élodie", 28, 34),
        ("€10", 40, 43),
    ]


def test_conflicting_labels_on_the_same_span_fail_validation():
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [
        {"entity_text": "Alice", "label": "PERSON"},
        {"entity_text": "Alice", "label": "ORG"},
    ]
    annotations = build_mention_annotations(
        doc_id="doc1",
        document_text="Alice",
        references=references,
        contract=contract,
    )

    with pytest.raises(GroundTruthError, match="span conflicts"):
        validate_mention_annotations(
            doc_id="doc1",
            document_text="Alice",
            annotations=annotations,
            references=references,
            contract=contract,
        )


def test_invalid_offsets_and_duplicate_matching_keys_fail_validation():
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [{"entity_text": "Alice", "label": "PERSON"}]
    annotations = [
        MentionAnnotation("doc1-001", "doc1", "Alice", "PERSON", 0, 5),
        MentionAnnotation("doc1-002", "doc1", "Alice", "PERSON", 0, 5),
        MentionAnnotation("doc1-003", "doc1", "Alice", "PERSON", 0, 99),
    ]

    with pytest.raises(GroundTruthError) as exc_info:
        validate_mention_annotations(
            doc_id="doc1",
            document_text="Alice",
            annotations=annotations,
            references=references,
            contract=contract,
        )

    assert any("matching key is duplicated" in issue for issue in exc_info.value.issues)
    assert any("invalid offsets" in issue for issue in exc_info.value.issues)


def test_surname_only_mentions_are_annotated_when_full_name_is_known():
    document_text = "Hynek Němec appointed the committee. Němec then signed the contract."
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [
        {"entity_text": "Hynek Němec", "label": "PERSON"},
        {"entity_text": "Němec", "label": "PERSON"},
        {"entity_text": "Hynek", "label": "PERSON"},
    ]

    annotations = build_mention_annotations(
        doc_id="doc1",
        document_text=document_text,
        references=references,
        contract=contract,
    )

    assert [(row.entity_text, row.start_char, row.end_char) for row in annotations] == [
        ("Hynek Němec", 0, 11),
        ("Němec", 37, 42),
    ]


def test_title_span_keeps_standalone_surname_mentions():
    document_text = "Dr Němec filed the case. Němec later appeared."
    contract = load_groundtruth_contract(CONTRACT_PATH)
    references = [
        {"entity_text": "Dr Němec", "label": "TITLE"},
        {"entity_text": "Němec", "label": "PERSON"},
    ]

    annotations = build_mention_annotations(
        doc_id="doc1",
        document_text=document_text,
        references=references,
        contract=contract,
    )

    assert [(row.entity_text, row.label, row.start_char, row.end_char) for row in annotations] == [
        ("Dr Němec", "TITLE", 0, 8),
        ("Němec", "PERSON", 25, 30),
    ]


def test_only_initial_entities_present_in_final_text_are_annotated(tmp_path):
    doc_dir = _write_package(
        tmp_path,
        document_text=(
            "Alice Example appeared twice. Alice Example mentioned Unknown Ltd and £99,999."
        ),
        defendants=[
            {
                "name": "Alice Example",
                "short_name": "Alice",
                "surface_forms_list": ["Alice Example", "Alice"],
            },
            {"name": "Missing Person"},
        ],
        charged_orgs=[{"name": "Known Ltd"}],
        amounts={"total_loss": "£10"},
    )

    result = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert result["status"] == "completed"
    annotations = read_groundtruth_tsv(doc_dir / "groundtruth.tsv")
    assert [(row.entity_text, row.label) for row in annotations] == [
        ("Alice Example", "PERSON"),
        ("Alice Example", "PERSON"),
    ]
    excluded = {"Missing Person", "Unknown Ltd", "£99,999"}
    assert not any(row.entity_text in excluded for row in annotations)


def test_groundtruth_is_reproduced_idempotently_from_saved_inputs(tmp_path):
    doc_dir = _write_package(
        tmp_path,
        document_text=(
            "Alice Example appeared before Test Synthetic Court. Alice Example paid £10."
        ),
        defendants=[{"name": "Alice Example"}],
        metadata={"court": "Test Synthetic Court"},
        amounts={"total_loss": "£10"},
    )

    first = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )
    second = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert first["groundtruth_sha256"] == second["groundtruth_sha256"]
    assert b"\r\n" not in (doc_dir / "groundtruth.tsv").read_bytes()
    annotations = read_groundtruth_tsv(doc_dir / "groundtruth.tsv")
    manifest = json.loads((doc_dir / "groundtruth_manifest.json").read_text())
    assert "document_inputs_sha256" in manifest
    assert "generation_report_sha256" not in manifest
    assert [(row.entity_text, row.label) for row in annotations] == [
        ("Alice Example", "PERSON"),
        ("Test Synthetic Court", "NEGATIVE_CONTROL"),
        ("Alice Example", "PERSON"),
        ("£10", "AMOUNT"),
    ]
    assert GROUNDTRUTH_HEADER == (
        "annotation_id",
        "doc_id",
        "entity_text",
        "label",
        "start_char",
        "end_char",
    )


def test_partial_existing_groundtruth_is_regenerated(tmp_path):
    doc_dir = _write_package(
        tmp_path,
        document_text="Alice Example appeared.",
        defendants=[{"name": "Alice Example"}],
    )
    invalid_tsv = "invalid\theader\n"
    (doc_dir / "groundtruth.tsv").write_text(invalid_tsv, encoding="utf-8")

    result = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert result["status"] == "completed"
    assert result["reused"] is False
    assert (doc_dir / "groundtruth.tsv").read_text(encoding="utf-8") != invalid_tsv
    assert (doc_dir / "groundtruth_manifest.json").is_file()


def test_stale_manifest_without_groundtruth_is_regenerated(tmp_path):
    doc_dir = _write_package(
        tmp_path,
        document_text="Alice Example appeared.",
        defendants=[{"name": "Alice Example"}],
    )
    (doc_dir / "groundtruth_manifest.json").write_text(
        json.dumps({"status": "completed", "doc_id": "doc1"}),
        encoding="utf-8",
    )

    result = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert result["status"] == "completed"
    assert result["reused"] is False
    assert (doc_dir / "groundtruth.tsv").is_file()
    manifest = json.loads((doc_dir / "groundtruth_manifest.json").read_text())
    assert manifest["annotation_count"] == 1
    assert "document_inputs_sha256" in manifest


def test_saved_document_package_is_enough_for_surname_groundtruth(tmp_path):
    document = DocumentInputs(
        defendants=[
            {
                "name": "Hynek Němec",
                "short_name": "Hynek",
                "initials": "H.N.",
                "title_surname": "Dr Němec",
                "surface_forms_list": ["Hynek Němec"],
            }
        ],
        collateral=[],
        charged_orgs=[],
        associated_orgs=[],
        metadata={},
        amounts={},
        counts_list=[],
    )
    context = SimpleNamespace(
        output_dir=tmp_path / "output",
        generation_cfg=SimpleNamespace(words_per_page=300),
    )
    document_text = "Hynek Němec appointed the committee. Němec signed. H.N attended."
    save_document_artifacts(context, document, "doc1", document_text)
    doc_dir = context.output_dir / "doc1"

    assert discover_document_packages(context.output_dir) == [doc_dir]
    result = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert result["status"] == "completed"
    annotations = read_groundtruth_tsv(doc_dir / "groundtruth.tsv")
    assert [(row.entity_text, row.label) for row in annotations] == [
        ("Hynek Němec", "PERSON"),
        ("Němec", "PERSON"),
        ("H.N", "INITIAL"),
    ]


def test_person_catalogue_includes_surname_and_stripped_initials():
    document = DocumentInputs(
        defendants=[
            {
                "name": "Hynek Němec",
                "short_name": "Hynek",
                "initials": "H.N.",
                "title_surname": "Dr Němec",
                "surface_forms_list": ["Hynek Němec", "H.N."],
            }
        ],
        collateral=[],
        charged_orgs=[],
        associated_orgs=[],
        metadata={},
        amounts={},
        counts_list=[],
    )

    references = build_entity_references(document)
    keys = {(row["entity_text"], row["label"]) for row in references}

    assert ("Hynek Němec", "PERSON") in keys
    assert ("Němec", "PERSON") in keys
    assert ("Hynek", "PERSON") in keys
    assert ("H.N.", "INITIAL") in keys
    assert ("H.N", "INITIAL") in keys
    assert ("Dr Němec", "TITLE") in keys


def test_missing_saved_document_inputs_are_rejected(tmp_path):
    doc_dir = tmp_path / "doc1"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Alice appeared.\n", encoding="utf-8")

    with pytest.raises(GroundTruthError, match="document_inputs.json"):
        generate_groundtruth_for_document(document_dir=doc_dir, contract_path=CONTRACT_PATH)


def test_invalid_saved_document_inputs_are_rejected(tmp_path):
    doc_dir = tmp_path / "doc1"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Alice appeared.\n", encoding="utf-8")
    (doc_dir / "document_inputs.json").write_text(
        json.dumps({"defendants": "not a list"}),
        encoding="utf-8",
    )

    with pytest.raises(GroundTruthError, match="defendants must be a list"):
        generate_groundtruth_for_document(document_dir=doc_dir, contract_path=CONTRACT_PATH)


def _write_package(
    root: Path,
    *,
    document_text: str,
    defendants: list[dict] | None = None,
    collateral: list[dict] | None = None,
    charged_orgs: list[dict] | None = None,
    associated_orgs: list[dict] | None = None,
    metadata: dict | None = None,
    amounts: dict | None = None,
    counts_list: list[dict] | None = None,
) -> Path:
    doc_dir = root / "doc1"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text(document_text + "\n", encoding="utf-8")
    payload = {
        "defendants": defendants or [],
        "collateral": collateral or [],
        "charged_orgs": charged_orgs or [],
        "associated_orgs": associated_orgs or [],
        "metadata": metadata or {},
        "amounts": amounts or {},
        "counts_list": counts_list or [],
        "evidence_categories": [],
        "scenario_brief": {},
    }
    (doc_dir / "document_inputs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return doc_dir
