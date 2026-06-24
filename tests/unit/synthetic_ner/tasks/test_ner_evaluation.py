import json
from pathlib import Path

from src.synthetic_ner.tasks.ner_evaluation.evaluator import (
    EntityRow,
    Prediction,
    evaluate_document_ner,
    score_soft,
)


def test_soft_matching_is_one_to_one_for_address_components():
    expected = [
        EntityRow("doc-1", "1 Legal Street, London EC1A 1AA", "ADDRESS"),
        EntityRow("doc-1", "1 Legal Street", "ADDRESS"),
        EntityRow("doc-1", "London EC1A 1AA", "ADDRESS"),
    ]
    predictions = [
        Prediction("1 Legal Street, London EC1A 1AA", "ADDRESS"),
    ]

    result = score_soft(expected, predictions)

    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["fp"] == 0
    assert result["metrics"]["fn"] == 2


def test_evaluate_document_ner_calibrates_groundtruth_with_rendered_memory(tmp_path: Path):
    doc_id = "doc-1"
    output_dir = tmp_path / "output" / doc_id
    memory_dir = tmp_path / "memory" / f"case_{doc_id}"
    output_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    (output_dir / f"{doc_id}.txt").write_text(
        "ALICE SMITH transferred £10 to PRICE LTD on 1 March 2025. "
        "PRICE LTD operated from 1 Legal Street, London EC1A 1AA.",
        encoding="utf-8",
    )
    (output_dir / "groundtruth.tsv").write_text(
        "\n".join(
            [
                "doc_id\tentity_text\tlabel\tshould_propose\tnotes",
                "doc-1\tAlice Smith\tPERSON\tyes\tperson",
                "doc-1\t2 March 2025\tDATE\tyes\tabsent date",
                "doc-1\tSerious Fraud Office\tNEGATIVE_CONTROL\tno\tprosecution",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (memory_dir / "CASE_MEMORY.md").write_text(
        "\n".join(
            [
                "# CASE_MEMORY",
                "## Document",
                "- Filing date: 1 March 2025",
                "## Organisations",
                "- PRICE LTD | VAT: GB123456789 | address: 1 Legal Street, London EC1A 1AA",
                "## Amounts",
                "- Total alleged loss: £10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        output_dir / "repo_ner_predictions.jsonl",
        [
            {
                "entities": [
                    {"entity_name": "ALICE SMITH", "label": "Person"},
                    {"entity_name": "PRICE LTD", "label": "Organization"},
                    {"entity_name": "£10", "label": "Amount"},
                    {
                        "entity_name": "1 Legal Street, London EC1A 1AA",
                        "label": "Address",
                    },
                ]
            }
        ],
    )

    result = evaluate_document_ner(
        project_root=tmp_path,
        doc_id=doc_id,
        calibration_mode="apply_with_memory",
    )

    assert result["calibration"]["removed_absent_rows"] == 1
    assert result["calibration"]["surface_normalizations"] == 1
    assert result["calibration"]["added_from_memory"] == 6
    assert result["strict"]["metrics"]["tp"] == 4
    assert result["strict"]["metrics"]["fn"] == 3
    assert Path(result["paths"]["report"]).exists()
    calibrated = (output_dir / "evaluation" / "calibrated_groundtruth.tsv").read_text(
        encoding="utf-8"
    )
    assert "ALICE SMITH\tPERSON" in calibrated
    assert "2 March 2025" not in calibrated


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
