import json
from pathlib import Path

import pytest
from src.synthetic_ner.tasks.groundtruth import (
    GROUNDTRUTH_HEADER,
    GroundTruthError,
    MentionAnnotation,
    build_entity_references,
    build_mention_annotations,
    generate_groundtruth_for_document,
    load_groundtruth_contract,
    read_groundtruth_tsv,
    validate_mention_annotations,
    write_document_reference_artifacts,
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
    for annotation in annotations:
        assert document_text[annotation.start_char : annotation.end_char] == annotation.entity_text


def test_unicode_punctuation_and_multiline_offsets_use_python_string_indexes():
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
    document_text = "Alice"
    references = [
        {"entity_text": "Alice", "label": "PERSON"},
        {"entity_text": "Alice", "label": "ORG"},
    ]
    annotations = build_mention_annotations(
        doc_id="doc1",
        document_text=document_text,
        references=references,
        contract=contract,
    )

    with pytest.raises(GroundTruthError, match="span conflicts"):
        validate_mention_annotations(
            doc_id="doc1",
            document_text=document_text,
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


def test_reference_catalogue_includes_variants_and_negative_controls():
    references = build_entity_references(_document_inputs(), address_surface_forms=3)
    by_key = {(row["entity_text"], row["label"]): row for row in references}

    assert ("Alice Example", "PERSON") in by_key
    assert ("Ali Example", "PERSON") in by_key
    assert ("A.E.", "INITIAL") in by_key
    assert ("Dr Example", "TITLE") in by_key
    assert ("Serious Fraud Office", "NEGATIVE_CONTROL") in by_key
    assert ("Test Synthetic Court", "NEGATIVE_CONTROL") in by_key
    assert ("10 Legal Street, London EC1A 1AA", "ADDRESS") in by_key
    assert ("10 Legal Street", "ADDRESS") in by_key
    assert ("London EC1A 1AA", "ADDRESS") in by_key


def test_groundtruth_is_generated_after_document_and_reproduced_idempotently(tmp_path):
    doc_id = "doc1"
    doc_dir = tmp_path / doc_id
    doc_dir.mkdir()
    document_path = doc_dir / f"{doc_id}.txt"
    document_path.write_text(
        "Alice Example appeared before Test Synthetic Court. Alice Example paid £10.\n",
        encoding="utf-8",
    )
    write_document_reference_artifacts(
        doc_dir=doc_dir,
        doc_id=doc_id,
        document=_document_inputs(),
        document_path=document_path,
        address_surface_forms=3,
    )

    first = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )
    second = generate_groundtruth_for_document(
        document_dir=doc_dir,
        contract_path=CONTRACT_PATH,
    )

    assert first["status"] == "completed"
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["groundtruth_sha256"] == second["groundtruth_sha256"]
    assert b"\r\n" not in (doc_dir / "groundtruth.tsv").read_bytes()
    annotations = read_groundtruth_tsv(doc_dir / "groundtruth.tsv")
    manifest = json.loads((doc_dir / "groundtruth_manifest.json").read_text())
    assert manifest["overlap_policy"] == {
        "allow_nested_same_label": ["ADDRESS"],
        "prefer_longest_same_label": True,
    }
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


def test_invalid_existing_groundtruth_is_not_repaired_and_writes_error_report(tmp_path):
    doc_id = "doc1"
    doc_dir = tmp_path / doc_id
    doc_dir.mkdir()
    document_path = doc_dir / f"{doc_id}.txt"
    document_path.write_text("Alice Example appeared.\n", encoding="utf-8")
    write_document_reference_artifacts(
        doc_dir=doc_dir,
        doc_id=doc_id,
        document=_document_inputs(),
        document_path=document_path,
        address_surface_forms=3,
    )
    invalid_tsv = "invalid\theader\n"
    (doc_dir / "groundtruth.tsv").write_text(invalid_tsv, encoding="utf-8")

    with pytest.raises(GroundTruthError, match="will not be overwritten"):
        generate_groundtruth_for_document(
            document_dir=doc_dir,
            contract_path=CONTRACT_PATH,
        )

    assert (doc_dir / "groundtruth.tsv").read_text(encoding="utf-8") == invalid_tsv
    error_report = json.loads(
        (doc_dir / "groundtruth_validation_errors.json").read_text(encoding="utf-8")
    )
    assert error_report["doc_id"] == doc_id
    assert "will not be overwritten" in error_report["issues"][0]


def _document_inputs() -> DocumentInputs:
    person = {
        "name": "Alice Example",
        "initials": "A.E.",
        "title_surname": "Dr Example",
        "short_name": "Alice",
        "surface_forms_list": [
            "Alice Example",
            "A.E.",
            "Dr Example",
            "Alice",
            "Ali Example",
        ],
        "dob": "1 January 1980",
        "address": "10 Legal Street, London EC1A 1AA",
        "street": "10 Legal Street",
        "city_postcode": "London EC1A 1AA",
        "is_defendant": True,
    }
    organisation = {
        "name": "EXAMPLE LTD",
        "address": "1 Company House, London W1A 1AA",
        "street": "1 Company House",
        "city_postcode": "London W1A 1AA",
        "vat": "GB123456789",
    }
    return DocumentInputs(
        defendants=[person],
        collateral=[],
        charged_orgs=[organisation],
        associated_orgs=[],
        metadata={
            "court": "Test Synthetic Court",
            "case_number": "CPS/2026/1234",
            "legal_reference": "1234567/890",
            "cross_ref": "C/2026/5678",
            "filing_date": "3 March 2026",
            "offence_period": None,
        },
        counts_list=[{"particulars": "Alice Example paid £10."}],
        amounts={"total_loss": "£10"},
    )
