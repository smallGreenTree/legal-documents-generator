import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.tasks.augmentation import (
    build_variant_id,
    discover_morphology_sources,
    protect_document_text,
    reconstruct_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.morphology import MorphologyAugmenter
from src.synthetic_ner.tasks.augmentation.publication import (
    existing_variant_result,
    publish_morphology_variant,
)
from src.synthetic_ner.tasks.groundtruth import generate_groundtruth_for_document
from src.synthetic_ner.tasks.groundtruth.models import MentionAnnotation
from src.synthetic_ner.types.augmentation import (
    MorphologyError,
    MorphologyPromptsConfig,
    MorphologyTransformation,
    MorphologyWorkflowConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONTRACT_PATH = PROJECT_ROOT / "groundtruth_contract.yaml"


def test_discovers_one_complete_package_from_txt_or_package_path(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")

    from_file = discover_morphology_sources(document_dir / "doc1.txt")
    from_package = discover_morphology_sources(document_dir)

    assert [source.doc_id for source in from_file] == ["doc1"]
    assert [source.doc_id for source in from_package] == ["doc1"]
    assert from_file[0].annotations == from_package[0].annotations


def test_discovers_every_complete_package_in_parent_folder(tmp_path):
    _complete_document_package(tmp_path, "doc1")
    _complete_document_package(tmp_path, "doc2")

    sources = discover_morphology_sources(tmp_path)

    assert [source.doc_id for source in sources] == ["doc1", "doc2"]


def test_rejects_source_without_completed_groundtruth(tmp_path):
    document_dir = tmp_path / "doc1"
    document_dir.mkdir()
    (document_dir / "doc1.txt").write_text("Alice appeared.\n", encoding="utf-8")

    with pytest.raises(MorphologyError, match="complete ground truth"):
        discover_morphology_sources(document_dir / "doc1.txt")


def test_protects_mentions_and_reconstructs_offsets_after_reordering():
    source_text = "Alice awarded the contract for £10."
    annotations = (
        MentionAnnotation("doc1-001", "doc1", "Alice", "PERSON", 0, 5),
        MentionAnnotation("doc1-002", "doc1", "£10", "AMOUNT", 31, 34),
    )
    protected = protect_document_text(source_text, annotations)
    person_token = protected.mentions[0].token
    amount_token = protected.mentions[1].token

    transformed = f"The contract for {amount_token} was awarded by {person_token}."
    variant = reconstruct_morphology_variant(
        source_text=source_text,
        protected=protected,
        transformed_text=transformed,
        variant_doc_id="doc1__morph-active-to-passive__v01",
        transformation=MorphologyTransformation.ACTIVE_TO_PASSIVE,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        contract_path=CONTRACT_PATH,
    )

    assert variant.text == "The contract for £10 was awarded by Alice."
    assert [(row.entity_text, row.start_char, row.end_char) for row in variant.annotations] == [
        ("£10", 17, 20),
        ("Alice", 36, 41),
    ]


def test_rejects_missing_protected_mention():
    source_text = "Alice appeared."
    annotations = (MentionAnnotation("doc1-001", "doc1", "Alice", "PERSON", 0, 5),)
    protected = protect_document_text(source_text, annotations)

    with pytest.raises(MorphologyError, match="exactly once"):
        reconstruct_morphology_variant(
            source_text=source_text,
            protected=protected,
            transformed_text="A person appeared.",
            variant_doc_id="doc1__morph-verbal-to-nominal__v01",
            transformation=MorphologyTransformation.VERBAL_TO_NOMINAL,
            minimum_change_ratio=0.01,
            maximum_change_ratio=0.95,
            contract_path=CONTRACT_PATH,
        )


def test_preserves_nested_address_annotations_as_one_protected_span():
    address = "Matenská 38, Zákupy 107 62"
    source_text = f"At {address}."
    full_start = source_text.index(address)
    street = "Matenská 38"
    locality = "Zákupy 107 62"
    annotations = (
        MentionAnnotation(
            "doc1-001",
            "doc1",
            street,
            "ADDRESS",
            full_start,
            full_start + len(street),
        ),
        MentionAnnotation(
            "doc1-002",
            "doc1",
            address,
            "ADDRESS",
            full_start,
            full_start + len(address),
        ),
        MentionAnnotation(
            "doc1-003",
            "doc1",
            locality,
            "ADDRESS",
            source_text.index(locality),
            source_text.index(locality) + len(locality),
        ),
    )

    protected = protect_document_text(source_text, annotations)
    variant = reconstruct_morphology_variant(
        source_text=source_text,
        protected=protected,
        transformed_text=f"The recorded residence was {protected.mentions[0].token}.",
        variant_doc_id="doc1__morph-verbal-to-nominal__v01",
        transformation=MorphologyTransformation.VERBAL_TO_NOMINAL,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        contract_path=CONTRACT_PATH,
    )

    assert variant.text == f"The recorded residence was {address}."
    assert {(row.entity_text, row.label) for row in variant.annotations} == {
        (street, "ADDRESS"),
        (address, "ADDRESS"),
        (locality, "ADDRESS"),
    }


def test_rejects_unchanged_text():
    source_text = "Alice appeared."
    annotations = (MentionAnnotation("doc1-001", "doc1", "Alice", "PERSON", 0, 5),)
    protected = protect_document_text(source_text, annotations)

    with pytest.raises(MorphologyError, match="minimum change ratio"):
        reconstruct_morphology_variant(
            source_text=source_text,
            protected=protected,
            transformed_text=protected.text,
            variant_doc_id="doc1__morph-verbal-to-nominal__v01",
            transformation=MorphologyTransformation.VERBAL_TO_NOMINAL,
            minimum_change_ratio=0.01,
            maximum_change_ratio=0.95,
            contract_path=CONTRACT_PATH,
        )


@pytest.mark.parametrize(
    ("transformation", "expected"),
    [
        (
            MorphologyTransformation.ACTIVE_TO_PASSIVE,
            "doc1__morph-active-to-passive__v01",
        ),
        (
            MorphologyTransformation.VERBAL_TO_NOMINAL,
            "doc1__morph-verbal-to-nominal__v01",
        ),
        (
            MorphologyTransformation.POSSESSIVE_REFRAME,
            "doc1__morph-possessive-reframe__v01",
        ),
        (
            MorphologyTransformation.INTENTIONAL_TYPOS,
            "doc1__morph-intentional-typos__v01",
        ),
        (
            MorphologyTransformation.RANDOM_LAYOUT,
            "doc1__morph-random-layout__v01",
        ),
    ],
)
def test_variant_id_names_the_morphological_change(transformation, expected):
    assert build_variant_id("doc1", transformation) == expected


def test_augmenter_calls_morphology_stage_and_restores_protected_values(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    client = _PassiveVoiceClient()
    config = MorphologyWorkflowConfig(
        temperature=0.2,
        max_output_tokens=500,
        max_chunk_chars=6000,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        prompts=MorphologyPromptsConfig(
            system="Protect every token.",
            user="RULE: {{ transformation_instruction }}\nTEXT:\n{{ protected_text }}",
        ),
    )

    variant = MorphologyAugmenter(client=client, config=config).create_variant(
        source=source,
        transformation=MorphologyTransformation.ACTIVE_TO_PASSIVE,
        variant_doc_id="doc1__morph-active-to-passive__v01",
        contract_path=CONTRACT_PATH,
    )

    assert variant.text == "The contract for £10 was awarded by Alice.\n"
    assert client.calls == [("morphology", 0.2, 500)]


def test_publishes_and_reuses_named_complete_variant_package(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    protected = protect_document_text(source.text, source.annotations)
    person_token = protected.mentions[0].token
    amount_token = protected.literals[0].token
    transformed = f"The contract for {amount_token} was awarded by {person_token}.\n"
    variant_id = build_variant_id(source.doc_id, MorphologyTransformation.ACTIVE_TO_PASSIVE)
    variant = reconstruct_morphology_variant(
        source_text=source.text,
        protected=protected,
        transformed_text=transformed,
        variant_doc_id=variant_id,
        transformation=MorphologyTransformation.ACTIVE_TO_PASSIVE,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        contract_path=CONTRACT_PATH,
    )

    result = publish_morphology_variant(
        source=source,
        variant=variant,
        contract_path=CONTRACT_PATH,
    )
    reused = existing_variant_result(
        source,
        variant_id,
        MorphologyTransformation.ACTIVE_TO_PASSIVE,
    )

    output_dir = document_dir / "augmentations" / variant_id
    assert result["variant_directory"] == str(output_dir)
    assert (output_dir / f"{variant_id}.txt").is_file()
    assert (output_dir / "groundtruth.tsv").is_file()
    manifest = json.loads((output_dir / "augmentation_manifest.json").read_text())
    assert manifest["transformation"] == "active-to-passive"
    assert "passive-voice" in manifest["transformation_explanation"]
    assert reused is not None and reused["reused"] is True


@pytest.mark.parametrize(
    "transformation",
    [
        MorphologyTransformation.INTENTIONAL_TYPOS,
        MorphologyTransformation.RANDOM_LAYOUT,
    ],
)
def test_deterministic_variations_preserve_entities_without_calling_model(
    tmp_path,
    transformation,
):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    config = MorphologyWorkflowConfig(
        temperature=0.2,
        max_output_tokens=500,
        max_chunk_chars=6000,
        minimum_change_ratio=0.01,
        deterministic_minimum_change_ratio=0.0001,
        maximum_change_ratio=0.95,
        typo_rate=0.5,
        max_typos=3,
        layout_widths=(32, 40),
        prompts=MorphologyPromptsConfig(system="unused", user="unused"),
    )
    augmenter = MorphologyAugmenter(client=_ForbiddenClient(), config=config)
    variant_id = build_variant_id(source.doc_id, transformation)

    first = augmenter.create_variant(
        source=source,
        transformation=transformation,
        variant_doc_id=variant_id,
        contract_path=CONTRACT_PATH,
    )
    second = augmenter.create_variant(
        source=source,
        transformation=transformation,
        variant_doc_id=variant_id,
        contract_path=CONTRACT_PATH,
    )

    assert first.text == second.text
    assert first.text != source.text
    assert "Alice" in first.text
    assert "£10" in first.text
    assert [(row.entity_text, row.label) for row in first.annotations] == [("Alice", "PERSON")]


def _complete_document_package(root: Path, doc_id: str) -> Path:
    document_dir = root / doc_id
    document_dir.mkdir()
    (document_dir / f"{doc_id}.txt").write_text(
        "Alice awarded the contract for £10.\n",
        encoding="utf-8",
    )
    (document_dir / "document_inputs.json").write_text(
        json.dumps(
            {
                "defendants": [{"name": "Alice"}],
                "collateral": [],
                "charged_orgs": [],
                "associated_orgs": [],
                "metadata": {},
                "amounts": {"total": "£10"},
                "counts_list": [],
            }
        ),
        encoding="utf-8",
    )
    generate_groundtruth_for_document(
        document_dir=document_dir,
        contract_path=CONTRACT_PATH,
    )
    return document_dir


class _PassiveVoiceClient:
    def __init__(self):
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append((kwargs["stage"], kwargs["temperature"], kwargs["max_output_tokens"]))
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        person_token = re.findall(r"⟦NER_\d{4}⟧", protected_text)[0]
        amount_token = re.findall(r"⟦LITERAL_\d{4}⟧", protected_text)[0]
        return SimpleNamespace(
            text=f"The contract for {amount_token} was awarded by {person_token}.",
            metadata={},
        )


class _ForbiddenClient:
    def invoke(self, **_kwargs):
        raise AssertionError("deterministic variation must not call the model")
