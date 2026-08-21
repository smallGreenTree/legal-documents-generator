import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.tasks.augmentation import (
    build_variant_id,
    discover_morphology_sources,
    protect_document_text,
    reconstruct_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.morphology import (
    MorphologyAugmenter,
    _chunk_validation_error,
)
from src.synthetic_ner.tasks.augmentation.publication import (
    existing_variant_result,
    publish_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.style import normalize_style_temperature
from src.synthetic_ner.tasks.groundtruth import generate_groundtruth_for_document
from src.synthetic_ner.tasks.groundtruth.models import MentionAnnotation
from src.synthetic_ner.types.augmentation import (
    MorphologyError,
    MorphologyPromptsConfig,
    MorphologySource,
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


def test_discovers_flat_golden_txt_with_prefixed_groundtruth_catalog(tmp_path):
    document_path = _flat_golden_pair(tmp_path, "golden1")

    sources = discover_morphology_sources(document_path, contract_path=CONTRACT_PATH)

    assert [source.doc_id for source in sources] == ["golden1"]
    assert [(row.entity_text, row.label) for row in sources[0].annotations] == [
        ("Alice", "PERSON"),
        ("£10", "AMOUNT"),
    ]


def test_discovers_every_flat_golden_pair_in_folder(tmp_path):
    _flat_golden_pair(tmp_path, "golden1")
    _flat_golden_pair(tmp_path, "golden2")

    sources = discover_morphology_sources(tmp_path, contract_path=CONTRACT_PATH)

    assert [source.doc_id for source in sources] == ["golden1", "golden2"]


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


def test_custom_style_variant_id_contains_safe_style_slug():
    assert (
        build_variant_id(
            "doc1",
            MorphologyTransformation.CUSTOM_STYLE,
            style="Poetic legal prose!",
            style_temperature=0.8,
            reformat_with_style=True,
        )
        == "doc1__style-poetic-legal-prose__t0p8__reformatted__v01"
    )


def test_style_temperature_requires_slider_increment():
    assert normalize_style_temperature(1.1) == 1.1
    with pytest.raises(MorphologyError, match="increments of 0.1"):
        normalize_style_temperature(0.85)


def test_augmenter_calls_morphology_stage_and_restores_protected_values(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    client = _PassiveVoiceClient()
    config = MorphologyWorkflowConfig(
        temperature=0.2,
        style_temperature=0.8,
        style_retry_temperature=0.2,
        style_maximum_change_ratio=0.95,
        style_max_chunk_chars=1200,
        style_max_protected_tokens=8,
        style_max_sentences_per_chunk=2,
        max_output_tokens=500,
        max_chunk_chars=6000,
        max_chunk_attempts=2,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        prompts=MorphologyPromptsConfig(
            system="Protect every token.",
            user="RULE: {{ transformation_instruction }}\nTEXT:\n{{ protected_text }}",
            retry=(
                "{{ original_prompt }}\nERROR: {{ validation_error }}\n"
                "PREVIOUS:\n{{ previous_text }}"
            ),
            style_system="Make the requested style unmistakable.",
            style_user=(
                "STYLE: {{ requested_style }}\nUse internal or end rhyme for rap.\n"
                "{{ reformatting_instruction }}\nFINAL STYLE DIRECTIVE\n"
                "TEXT:\n{{ protected_text }}"
            ),
            style_retry="unused",
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


def test_custom_style_is_included_in_prompt_and_variant(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    client = _PassiveVoiceClient()
    style = "wry nineteenth-century American satire"

    variant = MorphologyAugmenter(
        client=client,
        config=_morphology_config(max_chunk_attempts=2),
    ).create_variant(
        source=source,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        variant_doc_id=(
            "doc1__style-wry-nineteenth-century-american-satire__t1p1__reformatted__v01"
        ),
        contract_path=CONTRACT_PATH,
        style=style,
        style_temperature=1.1,
        reformat_with_style=True,
    )

    assert variant.style == style
    assert variant.style_temperature == 1.1
    assert variant.reformat_with_style is True
    assert style in client.user_prompts[0]
    assert "internal or end rhyme" in client.user_prompts[0]
    assert "change line breaks, sentence boundaries and indentation" in client.user_prompts[0]
    assert "FINAL STYLE DIRECTIVE" in client.user_prompts[0]
    assert client.system_prompts == ["Make the requested style unmistakable."]
    assert client.calls == [("morphology", 1.1, 500)]

    result = publish_morphology_variant(
        source=source,
        variant=variant,
        contract_path=CONTRACT_PATH,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    reused = existing_variant_result(
        source,
        variant.doc_id,
        MorphologyTransformation.CUSTOM_STYLE,
        style=style,
        style_temperature=1.1,
        reformat_with_style=True,
    )

    assert manifest["style"] == style
    assert manifest["style_temperature"] == 1.1
    assert manifest["reformat_with_style"] is True
    assert manifest["style_fallback_chunk_count"] == 0
    assert manifest["transformation"] == "custom-style"
    assert reused is not None and reused["reused"] is True


def test_custom_style_uses_its_own_change_ratio_limit(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    config = replace(
        _morphology_config(max_chunk_attempts=2),
        maximum_change_ratio=0.05,
        style_maximum_change_ratio=0.95,
    )

    variant = MorphologyAugmenter(client=_PassiveVoiceClient(), config=config).create_variant(
        source=source,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        variant_doc_id="doc1__style-rap-couplets__t0p8__reformatted__v01",
        contract_path=CONTRACT_PATH,
        style="rap couplets",
        style_temperature=0.8,
        reformat_with_style=True,
    )

    assert variant.change_ratio > config.maximum_change_ratio
    assert variant.change_ratio <= config.style_maximum_change_ratio


def test_custom_style_retries_original_source_at_safe_temperature(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    client = _StyleMissingThenValidClient()

    variant = MorphologyAugmenter(
        client=client,
        config=_morphology_config(max_chunk_attempts=2),
    ).create_variant(
        source=source,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        variant_doc_id="doc1__style-rap-couplets__t0p9__reformatted__v01",
        contract_path=CONTRACT_PATH,
        style="rap couplets",
        style_temperature=0.9,
        reformat_with_style=True,
    )

    assert variant.text == "On beat: Alice awarded the contract for £10.\n"
    assert client.temperatures == [0.9, 0.2]
    assert "EXPECTED PROTECTED TOKEN SEQUENCE" in client.retry_prompt
    assert "PREVIOUS REJECTED OUTPUT" not in client.retry_prompt
    assert client.rejected_output not in client.retry_prompt


def test_custom_style_chunks_are_sentence_and_token_bounded(tmp_path):
    source = _style_source(tmp_path)
    client = _RecordingStyleClient()
    config = _morphology_config(max_chunk_attempts=2)
    config = replace(
        config,
        style_max_protected_tokens=3,
        style_max_sentences_per_chunk=2,
    )

    variant = MorphologyAugmenter(client=client, config=config).create_variant(
        source=source,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        variant_doc_id="style-doc__style-rap-couplets__t0p8__reformatted__v01",
        contract_path=CONTRACT_PATH,
        style="rap couplets",
        style_temperature=0.8,
        reformat_with_style=True,
    )

    assert variant.text.startswith("Serious Fraud Office\nTo the Crown Court\n\nINDICTMENT\n\n")
    assert all("Serious Fraud Office" not in chunk for chunk in client.source_chunks)
    assert all(len(re.findall(r"⟦NER_\d{4}⟧", chunk)) <= 3 for chunk in client.source_chunks)
    assert len(client.source_chunks) == 4


def test_custom_style_validation_rejects_reordered_tokens():
    source = "⟦NER_0001⟧ instructed ⟦NER_0002⟧."
    transformed = "⟦NER_0002⟧ was instructed by ⟦NER_0001⟧."

    assert _chunk_validation_error(source, transformed) == ""
    assert (
        _chunk_validation_error(source, transformed, preserve_token_order=True)
        == "protected token order changed"
    )


def test_custom_style_validation_rejects_raw_protected_entity_copy():
    source = "⟦NER_0001⟧ instructed ⟦NER_0002⟧."
    transformed = "⟦NER_0001⟧ instructed ⟦NER_0002⟧. Alice appeared."

    assert (
        _chunk_validation_error(
            source,
            transformed,
            preserve_token_order=True,
            protected_entity_surfaces=("Alice",),
        )
        == "model reproduced 1 protected entity occurrence(s) outside protected tokens"
    )


def test_custom_style_falls_back_to_original_chunk_after_retries():
    config = _morphology_config(max_chunk_attempts=2)
    augmenter = MorphologyAugmenter(client=_AlwaysMissingTokenClient(), config=config)
    chunk = "⟦NER_0001⟧ awarded the contract for ⟦LITERAL_0001⟧."

    transformed = augmenter._transform_chunk(
        doc_id="doc1",
        chunk=chunk,
        chunk_index=7,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        style="rap couplets",
        style_temperature=0.9,
        reformat_with_style=True,
        transformation_instruction="unused",
    )

    assert transformed == chunk
    assert augmenter.style_fallback_chunk_indices == (7,)


def test_custom_style_falls_back_when_model_copies_raw_entity():
    config = _morphology_config(max_chunk_attempts=2)
    augmenter = MorphologyAugmenter(client=_RawEntityCopyClient(), config=config)
    augmenter._protected_entity_surfaces = ("Alice",)
    chunk = "⟦NER_0001⟧ awarded the contract for ⟦LITERAL_0001⟧."

    transformed = augmenter._transform_chunk(
        doc_id="doc1",
        chunk=chunk,
        chunk_index=3,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        style="rap couplets",
        style_temperature=0.9,
        reformat_with_style=True,
        transformation_instruction="unused",
    )

    assert transformed == chunk
    assert augmenter.style_fallback_chunk_indices == (3,)


def test_custom_style_publishes_when_one_chunk_falls_back_for_raw_entity(tmp_path):
    source = _style_source(tmp_path)
    variant = MorphologyAugmenter(
        client=_RawEntityCopyThenValidStyleClient(),
        config=_morphology_config(max_chunk_attempts=2),
    ).create_variant(
        source=source,
        transformation=MorphologyTransformation.CUSTOM_STYLE,
        variant_doc_id="style-doc__style-rap-couplets__t0p8__reformatted__v01",
        contract_path=CONTRACT_PATH,
        style="rap couplets",
        style_temperature=0.8,
        reformat_with_style=True,
    )

    result = publish_morphology_variant(
        source=source,
        variant=variant,
        contract_path=CONTRACT_PATH,
    )

    assert variant.style_fallback_chunk_indices == (1,)
    assert variant.text != source.text
    assert result["status"] == "completed"
    assert Path(result["document_path"]).is_file()
    assert Path(result["groundtruth_path"]).is_file()


def test_retries_only_failed_chunk_after_protected_token_duplication(tmp_path):
    document_dir = _complete_document_package(tmp_path, "doc1")
    source = discover_morphology_sources(document_dir)[0]
    client = _DuplicateThenValidClient()
    config = _morphology_config(max_chunk_attempts=2)

    variant = MorphologyAugmenter(client=client, config=config).create_variant(
        source=source,
        transformation=MorphologyTransformation.ACTIVE_TO_PASSIVE,
        variant_doc_id="doc1__morph-active-to-passive__v01",
        contract_path=CONTRACT_PATH,
    )

    assert variant.text == "The contract for £10 was awarded by Alice.\n"
    assert client.task_ids == [
        "morphology_active-to-passive_001",
        "morphology_active-to-passive_001_retry_02",
    ]
    assert "duplicated or unexpected: ⟦NER_0001⟧" in client.retry_prompt


def test_standard_transformation_publishes_when_one_chunk_falls_back(tmp_path):
    source = _two_paragraph_source(tmp_path)
    variant = MorphologyAugmenter(
        client=_FirstChunkMismatchThenValidClient(),
        config=_morphology_config(max_chunk_attempts=2),
    ).create_variant(
        source=source,
        transformation=MorphologyTransformation.ACTIVE_TO_PASSIVE,
        variant_doc_id="two-paragraphs__morph-active-to-passive__v01",
        contract_path=CONTRACT_PATH,
    )

    result = publish_morphology_variant(
        source=source,
        variant=variant,
        contract_path=CONTRACT_PATH,
    )

    assert variant.style_fallback_chunk_indices == (1,)
    assert variant.text.startswith("Alice awarded the contract for £10 after review by Bob.")
    assert "Reframed:" in variant.text
    assert result["status"] == "completed"
    assert Path(result["document_path"]).is_file()
    assert Path(result["groundtruth_path"]).is_file()


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


def test_publishes_flat_golden_variant_as_complete_package(tmp_path):
    source = discover_morphology_sources(
        _flat_golden_pair(tmp_path, "golden1"),
        contract_path=CONTRACT_PATH,
    )[0]
    protected = protect_document_text(source.text, source.annotations)
    person_token = protected.mentions[0].token
    amount_token = protected.mentions[1].token
    variant_id = build_variant_id(source.doc_id, MorphologyTransformation.ACTIVE_TO_PASSIVE)
    variant = reconstruct_morphology_variant(
        source_text=source.text,
        protected=protected,
        transformed_text=f"The contract for {amount_token} was awarded by {person_token}.\n",
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

    output_dir = tmp_path / "augmentations" / variant_id
    inputs = json.loads((output_dir / "document_inputs.json").read_text())
    assert result["variant_directory"] == str(output_dir)
    assert [(row["entity_text"], row["label"]) for row in inputs["entity_references"]] == [
        ("Alice", "PERSON"),
        ("£10", "AMOUNT"),
    ]
    assert (output_dir / "groundtruth_manifest.json").is_file()
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
        style_temperature=0.8,
        style_retry_temperature=0.2,
        style_maximum_change_ratio=0.95,
        style_max_chunk_chars=1200,
        style_max_protected_tokens=8,
        style_max_sentences_per_chunk=2,
        max_output_tokens=500,
        max_chunk_chars=6000,
        max_chunk_attempts=2,
        minimum_change_ratio=0.01,
        deterministic_minimum_change_ratio=0.0001,
        maximum_change_ratio=0.95,
        typo_rate=0.5,
        max_typos=3,
        layout_widths=(32, 40),
        prompts=MorphologyPromptsConfig(
            system="unused",
            user="unused",
            retry="unused",
            style_system="unused",
            style_user="unused",
            style_retry="unused",
        ),
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


def _flat_golden_pair(root: Path, doc_id: str) -> Path:
    document_path = root / f"{doc_id}.txt"
    document_path.write_text("Alice awarded the contract for £10.\n", encoding="utf-8")
    (root / f"groundtruth_{doc_id}.tsv").write_text(
        "doc_id\tentity_text\tlabel\tshould_propose\tnotes\n"
        f"{doc_id}\tAlice\tPERSON\tyes\tperson\n"
        f"{doc_id}\t£10\tAMOUNT\tyes\tamount\n",
        encoding="utf-8",
    )
    return document_path


def _style_source(root: Path) -> MorphologySource:
    text = (
        "Serious Fraud Office\nTo the Crown Court\n\nINDICTMENT\n\n"
        "Alice met Bob. Alice instructed Carol. Bob contacted Carol. "
        "Carol replied to Alice.\n"
    )
    document_path = root / "style-doc.txt"
    groundtruth_path = root / "groundtruth.tsv"
    document_path.write_text(text, encoding="utf-8")
    groundtruth_path.write_text("test", encoding="utf-8")
    spans = []
    for name in ("Alice", "Bob", "Alice", "Carol", "Bob", "Carol", "Carol", "Alice"):
        start = text.index(name, spans[-1][1] if spans else 0)
        spans.append((start, start + len(name), name))
    annotations = tuple(
        MentionAnnotation(
            annotation_id=f"style-doc-{index:03d}",
            doc_id="style-doc",
            entity_text=name,
            label="PERSON",
            start_char=start,
            end_char=end,
        )
        for index, (start, end, name) in enumerate(spans, start=1)
    )
    return MorphologySource(
        doc_id="style-doc",
        package_dir=root,
        document_path=document_path,
        document_inputs_path=None,
        groundtruth_path=groundtruth_path,
        text=text,
        annotations=annotations,
        entity_references=tuple(
            {"entity_text": name, "label": "PERSON"} for name in ("Alice", "Bob", "Carol")
        ),
    )


def _two_paragraph_source(root: Path) -> MorphologySource:
    doc_id = "two-paragraphs"
    text = (
        "Alice awarded the contract for £10 after review by Bob.\n\n"
        "Carol reviewed the records for £20 before reporting to Dave.\n"
    )
    document_path = root / f"{doc_id}.txt"
    groundtruth_path = root / "groundtruth.tsv"
    document_path.write_text(text, encoding="utf-8")
    groundtruth_path.write_text("test", encoding="utf-8")
    references = (
        ("Alice", "PERSON"),
        ("£10", "AMOUNT"),
        ("Bob", "PERSON"),
        ("Carol", "PERSON"),
        ("£20", "AMOUNT"),
        ("Dave", "PERSON"),
    )
    annotations = tuple(
        MentionAnnotation(
            annotation_id=f"{doc_id}-{index:03d}",
            doc_id=doc_id,
            entity_text=entity_text,
            label=label,
            start_char=text.index(entity_text),
            end_char=text.index(entity_text) + len(entity_text),
        )
        for index, (entity_text, label) in enumerate(references, start=1)
    )
    return MorphologySource(
        doc_id=doc_id,
        package_dir=root,
        document_path=document_path,
        document_inputs_path=None,
        groundtruth_path=groundtruth_path,
        text=text,
        annotations=annotations,
        entity_references=tuple(
            {"entity_text": entity_text, "label": label} for entity_text, label in references
        ),
    )


class _PassiveVoiceClient:
    def __init__(self):
        self.calls = []
        self.user_prompts = []
        self.system_prompts = []

    def invoke(self, **kwargs):
        self.calls.append((kwargs["stage"], kwargs["temperature"], kwargs["max_output_tokens"]))
        self.user_prompts.append(kwargs["user_prompt"])
        self.system_prompts.append(kwargs["system_prompt"])
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        person_token = re.findall(r"⟦NER_\d{4}⟧", protected_text)[0]
        amount_token = re.findall(r"⟦LITERAL_\d{4}⟧", protected_text)[0]
        if "custom-style" in kwargs["task_id"]:
            return SimpleNamespace(
                text=f"{person_token} handled the contract for {amount_token}, with bite.",
                metadata={},
            )
        return SimpleNamespace(
            text=f"The contract for {amount_token} was awarded by {person_token}.",
            metadata={},
        )


class _ForbiddenClient:
    def invoke(self, **_kwargs):
        raise AssertionError("deterministic variation must not call the model")


class _DuplicateThenValidClient:
    def __init__(self):
        self.task_ids = []
        self.retry_prompt = ""

    def invoke(self, **kwargs):
        self.task_ids.append(kwargs["task_id"])
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        person_token = re.findall(r"⟦NER_\d{4}⟧", protected_text)[0]
        amount_token = re.findall(r"⟦LITERAL_\d{4}⟧", protected_text)[0]
        if len(self.task_ids) == 1:
            text = f"{person_token} awarded it and {person_token} appeared with {amount_token}."
        else:
            self.retry_prompt = kwargs["user_prompt"]
            text = f"The contract for {amount_token} was awarded by {person_token}."
        return SimpleNamespace(text=text, metadata={})


class _AlwaysMissingTokenClient:
    def invoke(self, **kwargs):
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        amount_token = re.findall(r"⟦LITERAL_\d{4}⟧", protected_text)[0]
        return SimpleNamespace(text=f"The contract for {amount_token} was awarded.", metadata={})


class _FirstChunkMismatchThenValidClient:
    def __init__(self):
        self.calls = 0
        self.first_chunk = ""

    def invoke(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.first_chunk = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        if self.calls <= 2:
            first_token = re.findall(r"⟦(?:NER|LITERAL)_\d{4}⟧", self.first_chunk)[0]
            return SimpleNamespace(
                text=self.first_chunk.replace(first_token, "", 1),
                metadata={},
            )
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        return SimpleNamespace(text=f"Reframed: {protected_text}", metadata={})


class _RawEntityCopyClient:
    def invoke(self, **kwargs):
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        return SimpleNamespace(text=f"{protected_text} Alice appeared.", metadata={})


class _RawEntityCopyThenValidStyleClient:
    def __init__(self):
        self.calls = 0

    def invoke(self, **kwargs):
        self.calls += 1
        protected_text = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        if self.calls <= 2:
            return SimpleNamespace(text=f"{protected_text} Alice appeared.", metadata={})
        return SimpleNamespace(text=f"On beat: {protected_text}", metadata={})


class _StyleMissingThenValidClient:
    def __init__(self):
        self.temperatures = []
        self.retry_prompt = ""
        self.rejected_output = ""
        self.source = ""

    def invoke(self, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        if len(self.temperatures) == 1:
            self.source = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
            person_token = re.findall(r"⟦NER_\d{4}⟧", self.source)[0]
            self.rejected_output = f"On beat: {self.source.replace(person_token, '')}".strip()
            return SimpleNamespace(text=self.rejected_output, metadata={})
        self.retry_prompt = kwargs["user_prompt"]
        return SimpleNamespace(text=f"On beat: {self.source}", metadata={})


class _RecordingStyleClient:
    def __init__(self):
        self.source_chunks = []

    def invoke(self, **kwargs):
        source = kwargs["user_prompt"].split("TEXT:\n", 1)[1]
        self.source_chunks.append(source)
        return SimpleNamespace(text=f"On beat: {source}", metadata={})


def _morphology_config(*, max_chunk_attempts: int) -> MorphologyWorkflowConfig:
    return MorphologyWorkflowConfig(
        temperature=0.2,
        style_temperature=0.8,
        style_retry_temperature=0.2,
        style_maximum_change_ratio=0.95,
        style_max_chunk_chars=1200,
        style_max_protected_tokens=8,
        style_max_sentences_per_chunk=2,
        max_output_tokens=500,
        max_chunk_chars=6000,
        max_chunk_attempts=max_chunk_attempts,
        minimum_change_ratio=0.01,
        maximum_change_ratio=0.95,
        prompts=MorphologyPromptsConfig(
            system="Protect every token.",
            user="RULE: {{ transformation_instruction }}\nTEXT:\n{{ protected_text }}",
            retry=(
                "{{ original_prompt }}\nERROR: {{ validation_error }}\n"
                "PREVIOUS:\n{{ previous_text }}"
            ),
            style_system="Make the requested style unmistakable.",
            style_user=(
                "STYLE: {{ requested_style }}\nUse internal or end rhyme for rap.\n"
                "{{ reformatting_instruction }}\nFINAL STYLE DIRECTIVE\n"
                "TEXT:\n{{ protected_text }}"
            ),
            style_retry=(
                "STYLE: {{ requested_style }}\nERROR: {{ validation_error }}\n"
                "EXPECTED PROTECTED TOKEN SEQUENCE: {{ expected_tokens }}\n"
                "TEXT:\n{{ protected_text }}"
            ),
        ),
    )
