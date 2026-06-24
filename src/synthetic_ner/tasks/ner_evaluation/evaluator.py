"""Deterministic NER evaluation for generated document artifacts."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.synthetic_ner.constants import GROUNDTRUTH_HEADER

LABEL_ALIASES = {
    "Address": "ADDRESS",
    "Amount": "AMOUNT",
    "CaseLegalReference": "CASE_REFERENCE",
    "Date": "DATE",
    "Organization": "ORG",
    "Organisation": "ORG",
    "Person": "PERSON",
    "VAT": "VAT",
}
CALIBRATION_MODES = {"off", "diagnose", "apply_safe", "apply_with_memory"}
SOFT_THRESHOLDS = {
    "ADDRESS": 0.70,
    "AMOUNT": 1.00,
    "CASE_REFERENCE": 0.90,
    "DATE": 0.90,
    "ORG": 0.80,
    "PERSON": 0.80,
    "VAT": 1.00,
}
DATE_PATTERN = re.compile(
    r"\b\d{1,2} "
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December) "
    r"\d{4}\b"
)
AMOUNT_PATTERN = re.compile(r"[£€$]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
VAT_PATTERN = re.compile(r"\b[A-Z]{2}\d{9,12}\b")
CASE_REF_PATTERN = re.compile(r"\b(?:CPS/\d{4}/\d+|C/\d{4}/\d+)\b")
SECTION_REF_PATTERN = re.compile(
    r"\bsections? \d+(?: and \d+)? of "
    r"(?:the )?(?:same Act|[A-Z][A-Za-z ]+ Act \d{4})\b",
    re.IGNORECASE,
)
ORG_HINT_PATTERN = re.compile(r"\b(?:LTD|LIMITED|HOLDINGS|GROUP|SERVICES)\b")


@dataclass(frozen=True)
class EntityRow:
    doc_id: str
    text: str
    label: str
    should_propose: str = "yes"
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return normalize_surface(self.text), canonical_label(self.label)


@dataclass(frozen=True)
class Prediction:
    text: str
    label: str
    reason: str = ""
    abs_start_pos: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return normalize_surface(self.text), canonical_label(self.label)


def evaluate_document_ner(
    *,
    project_root: Path,
    doc_id: str,
    predictions_path: Path | str | None = None,
    memory_path: Path | str | None = None,
    calibration_mode: str = "apply_safe",
) -> dict[str, Any]:
    """Evaluate one document's NER predictions against generated groundtruth."""
    if calibration_mode not in CALIBRATION_MODES:
        modes = ", ".join(sorted(CALIBRATION_MODES))
        raise ValueError(f"calibration_mode must be one of: {modes}")

    paths = resolve_evaluation_paths(
        project_root=project_root,
        doc_id=doc_id,
        predictions_path=predictions_path,
        memory_path=memory_path,
    )
    document_text = paths["document"].read_text(encoding="utf-8")
    groundtruth_rows = read_groundtruth(paths["groundtruth"])
    predictions = read_predictions(paths["predictions"])
    memory_text = (
        paths["memory"].read_text(encoding="utf-8")
        if paths["memory"] is not None and paths["memory"].exists()
        else ""
    )

    calibration = calibrate_groundtruth(
        doc_id=doc_id,
        rows=groundtruth_rows,
        document_text=document_text,
        memory_text=memory_text,
        mode=calibration_mode,
    )
    expected = calibration["rows"]
    unique_predictions = dedupe_predictions(predictions)
    strict = score_strict(expected, unique_predictions)
    soft = score_soft(expected, unique_predictions)
    by_label = score_by_label(expected, unique_predictions)
    negative_controls = find_negative_control_predictions(
        calibration["negative_controls"],
        unique_predictions,
    )

    result = {
        "doc_id": doc_id,
        "paths": {key: str(value) if value is not None else None for key, value in paths.items()},
        "calibration_mode": calibration_mode,
        "calibration": calibration["summary"],
        "strict": strict,
        "soft": soft,
        "per_label": by_label,
        "negative_control_predictions": negative_controls,
        "prediction_count": len(predictions),
        "unique_prediction_count": len(unique_predictions),
    }

    output_dir = paths["output_dir"] / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_groundtruth(output_dir / "calibrated_groundtruth.tsv", expected)
    write_json(output_dir / "metrics.json", result)
    (output_dir / "calibration_report.md").write_text(
        render_calibration_report(doc_id, calibration), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    result["paths"]["evaluation_dir"] = str(output_dir)
    result["paths"]["report"] = str(output_dir / "report.md")
    result["paths"]["metrics"] = str(output_dir / "metrics.json")
    return result


def resolve_evaluation_paths(
    *,
    project_root: Path,
    doc_id: str,
    predictions_path: Path | str | None,
    memory_path: Path | str | None,
) -> dict[str, Path | None]:
    output_dir = project_root / "output" / doc_id
    resolved_predictions = _resolve_existing_path(
        explicit=predictions_path,
        candidates=[
            output_dir / "repo_ner_predictions.jsonl",
            output_dir / "ner_predictions.jsonl",
            output_dir / "predictions.jsonl",
        ],
        description="NER predictions JSONL",
    )
    resolved_memory = _resolve_optional_path(
        explicit=memory_path,
        candidates=[
            project_root / "memory" / f"case_{doc_id}" / "CASE_MEMORY.md",
            output_dir / "memory.md",
            output_dir / "CASE_MEMORY.md",
        ],
    )
    return {
        "output_dir": output_dir,
        "document": _require_file(output_dir / f"{doc_id}.txt", "document text"),
        "groundtruth": _require_file(output_dir / "groundtruth.tsv", "groundtruth TSV"),
        "predictions": resolved_predictions,
        "memory": resolved_memory,
    }


def read_groundtruth(path: Path) -> list[EntityRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                EntityRow(
                    doc_id=row.get("doc_id", ""),
                    text=row.get("entity_text", ""),
                    label=canonical_label(row.get("label", "")),
                    should_propose=row.get("should_propose", "yes"),
                    notes=row.get("notes", ""),
                )
            )
    return rows


def read_predictions(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        for entity in payload.get("entities", []):
            text = str(entity.get("entity_name", "")).strip()
            if not text:
                continue
            label = canonical_label(str(entity.get("label") or entity.get("entity_label") or ""))
            predictions.append(
                Prediction(
                    text=text,
                    label=label,
                    reason=str(entity.get("reason", "")),
                    abs_start_pos=entity.get("abs_start_pos"),
                )
            )
    return predictions


def calibrate_groundtruth(
    *,
    doc_id: str,
    rows: list[EntityRow],
    document_text: str,
    memory_text: str,
    mode: str,
) -> dict[str, Any]:
    positive_rows = [
        row
        for row in rows
        if row.should_propose.casefold() == "yes" and row.label != "NEGATIVE_CONTROL"
    ]
    negative_controls = [
        row
        for row in rows
        if row.should_propose.casefold() != "yes" or row.label == "NEGATIVE_CONTROL"
    ]
    absent_rows = [row for row in positive_rows if not surface_in_text(row.text, document_text)]
    normalized_rows = [
        (row, exact_surface_from_text(row.text, document_text))
        for row in positive_rows
        if surface_in_text(row.text, document_text)
        and exact_surface_from_text(row.text, document_text) != row.text
    ]
    memory_candidates = extract_memory_candidates(doc_id, memory_text)
    missing_memory_rows = [
        row
        for row in memory_candidates
        if surface_in_text(row.text, document_text)
        and row.key not in {candidate.key for candidate in positive_rows}
    ]

    if mode in {"off", "diagnose"}:
        calibrated_rows = positive_rows
        added_from_memory: list[EntityRow] = []
        removed_absent: list[EntityRow] = []
        normalized: list[tuple[EntityRow, str]] = []
    else:
        calibrated_rows = [
            _with_text(row, exact_surface_from_text(row.text, document_text))
            for row in positive_rows
            if surface_in_text(row.text, document_text)
        ]
        removed_absent = absent_rows
        normalized = normalized_rows
        added_from_memory = []
        if mode == "apply_with_memory":
            existing_keys = {row.key for row in calibrated_rows}
            for row in missing_memory_rows:
                adjusted = _with_text(row, exact_surface_from_text(row.text, document_text))
                if adjusted.key not in existing_keys:
                    calibrated_rows.append(adjusted)
                    added_from_memory.append(adjusted)
                    existing_keys.add(adjusted.key)

    return {
        "rows": calibrated_rows,
        "negative_controls": negative_controls,
        "summary": {
            "raw_rows": len(rows),
            "raw_positive_rows": len(positive_rows),
            "negative_control_rows": len(negative_controls),
            "absent_positive_rows": len(absent_rows),
            "removed_absent_rows": len(removed_absent),
            "surface_normalizations": len(normalized),
            "memory_candidates": len(memory_candidates),
            "rendered_memory_candidates_missing_from_groundtruth": len(missing_memory_rows),
            "added_from_memory": len(added_from_memory),
            "calibrated_rows": len(calibrated_rows),
        },
        "diagnostics": {
            "absent_rows": absent_rows,
            "normalized_rows": normalized_rows,
            "memory_candidates": memory_candidates,
            "missing_memory_rows": missing_memory_rows,
            "added_from_memory": added_from_memory,
        },
    }


def score_strict(expected: list[EntityRow], predictions: list[Prediction]) -> dict[str, Any]:
    expected_keys = {row.key for row in expected}
    prediction_keys = {prediction.key for prediction in predictions}
    true_positive_keys = expected_keys & prediction_keys
    false_negative_keys = expected_keys - prediction_keys
    false_positive_keys = prediction_keys - expected_keys
    return metric_payload(
        tp=len(true_positive_keys),
        fp=len(false_positive_keys),
        fn=len(false_negative_keys),
        matches=[
            {"expected": row.text, "predicted": row.text, "label": row.label}
            for row in expected
            if row.key in true_positive_keys
        ],
        false_positives=[
            prediction_to_dict(prediction)
            for prediction in predictions
            if prediction.key in false_positive_keys
        ],
        false_negatives=[
            row_to_dict(row) for row in expected if row.key in false_negative_keys
        ],
    )


def score_soft(expected: list[EntityRow], predictions: list[Prediction]) -> dict[str, Any]:
    matched_prediction_indexes: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_negatives: list[EntityRow] = []
    for row in expected:
        candidate = best_soft_match(row, predictions, matched_prediction_indexes)
        if candidate is None:
            false_negatives.append(row)
            continue
        index, prediction, coverage = candidate
        matched_prediction_indexes.add(index)
        matches.append(
            {
                "expected": row.text,
                "predicted": prediction.text,
                "label": row.label,
                "coverage": coverage,
                "threshold": SOFT_THRESHOLDS.get(row.label, 0.80),
            }
        )

    false_positives = [
        prediction_to_dict(prediction)
        for index, prediction in enumerate(predictions)
        if index not in matched_prediction_indexes
    ]
    return metric_payload(
        tp=len(matches),
        fp=len(false_positives),
        fn=len(false_negatives),
        matches=matches,
        false_positives=false_positives,
        false_negatives=[row_to_dict(row) for row in false_negatives],
    )


def score_by_label(expected: list[EntityRow], predictions: list[Prediction]) -> dict[str, Any]:
    labels = sorted(
        {row.label for row in expected} | {prediction.label for prediction in predictions}
    )
    return {
        label: score_soft(
            [row for row in expected if row.label == label],
            [prediction for prediction in predictions if prediction.label == label],
        )["metrics"]
        for label in labels
    }


def best_soft_match(
    expected: EntityRow,
    predictions: list[Prediction],
    used_indexes: set[int],
) -> tuple[int, Prediction, float] | None:
    threshold = SOFT_THRESHOLDS.get(expected.label, 0.80)
    candidates: list[tuple[float, int, Prediction]] = []
    for index, prediction in enumerate(predictions):
        if index in used_indexes or prediction.label != expected.label:
            continue
        coverage = expected_token_coverage(expected.text, prediction.text)
        if coverage >= threshold:
            candidates.append((coverage, index, prediction))
    if not candidates:
        return None
    coverage, index, prediction = max(candidates, key=lambda item: (item[0], -item[1]))
    return index, prediction, coverage


def metric_payload(
    *,
    tp: int,
    fp: int,
    fn: int,
    matches: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> dict[str, Any]:
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": 0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "bayesian": bayesian_metric_summary(tp=tp, fp=fp, fn=fn),
        },
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def bayesian_metric_summary(*, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = beta_posterior_summary(tp, tp + fp)
    recall = beta_posterior_summary(tp, tp + fn)
    samples = 20_000
    rng = random.Random(20260619 + tp * 101 + fp * 17 + fn)
    f1_samples: list[float] = []
    for _ in range(samples):
        precision_draw = rng.betavariate(1 + tp, 1 + fp)
        recall_draw = rng.betavariate(1 + tp, 1 + fn)
        f1_samples.append(divide(2 * precision_draw * recall_draw, precision_draw + recall_draw))
    return {
        "prior": "Beta(1, 1)",
        "precision": precision,
        "recall": recall,
        "f1": posterior_sample_summary(f1_samples),
    }


def beta_posterior_summary(successes: int, trials: int) -> dict[str, float]:
    failures = max(trials - successes, 0)
    alpha = 1 + successes
    beta = 1 + failures
    rng = random.Random(20260619 + successes * 131 + trials * 29)
    samples = sorted(rng.betavariate(alpha, beta) for _ in range(20_000))
    mle = divide(successes, trials)
    map_value = beta_map(alpha, beta)
    return {
        "mle": mle,
        "map": map_value,
        "mean": alpha / (alpha + beta),
        "ci_95_low": quantile(samples, 0.025),
        "ci_95_high": quantile(samples, 0.975),
    }


def posterior_sample_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": quantile(ordered, 0.50),
        "mean": sum(ordered) / len(ordered),
        "ci_95_low": quantile(ordered, 0.025),
        "ci_95_high": quantile(ordered, 0.975),
    }


def extract_memory_candidates(doc_id: str, memory_text: str) -> list[EntityRow]:
    if not memory_text:
        return []
    candidates: list[EntityRow] = []
    for text in CASE_REF_PATTERN.findall(memory_text):
        candidates.append(EntityRow(doc_id, text, "CASE_REFERENCE", notes="memory case reference"))
    for text in SECTION_REF_PATTERN.findall(memory_text):
        candidates.append(
            EntityRow(doc_id, _clean_text(text), "CASE_REFERENCE", notes="memory statute")
        )
    for text in DATE_PATTERN.findall(memory_text):
        candidates.append(EntityRow(doc_id, text, "DATE", notes="memory date"))
    for text in AMOUNT_PATTERN.findall(memory_text):
        candidates.append(EntityRow(doc_id, text, "AMOUNT", notes="memory amount"))
    for text in VAT_PATTERN.findall(memory_text):
        candidates.append(EntityRow(doc_id, text, "VAT", notes="memory VAT"))
    candidates.extend(_memory_person_candidates(doc_id, memory_text))
    candidates.extend(_memory_organisation_candidates(doc_id, memory_text))
    return dedupe_rows(candidates)


def _memory_person_candidates(doc_id: str, memory_text: str) -> list[EntityRow]:
    people: list[EntityRow] = []
    active = False
    for line in memory_text.splitlines():
        if line.startswith("## "):
            active = line.strip() in {"## Defendants", "## Collateral"}
            continue
        if not active or not line.startswith("- "):
            continue
        name = line[2:].split("|", 1)[0].strip()
        if name and name.casefold() != "none":
            people.append(EntityRow(doc_id, name, "PERSON", notes="memory person"))
    return people


def _memory_organisation_candidates(doc_id: str, memory_text: str) -> list[EntityRow]:
    organisations: list[EntityRow] = []
    active = False
    for line in memory_text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            active = line.strip() in {"## Organisations", "### Allowed Organisations"}
            continue
        if not active or not line.startswith("- "):
            continue
        body = line[2:].strip()
        name = body.split("|", 1)[0].strip()
        if name and ORG_HINT_PATTERN.search(name):
            organisations.append(EntityRow(doc_id, name, "ORG", notes="memory organisation"))
        address = _field_after(body, "address:")
        if address:
            organisations.append(EntityRow(doc_id, address, "ADDRESS", notes="memory address"))
            street, city_postcode = split_address(address)
            if street:
                organisations.append(
                    EntityRow(doc_id, street, "ADDRESS", notes="memory address street")
                )
            if city_postcode:
                organisations.append(
                    EntityRow(doc_id, city_postcode, "ADDRESS", notes="memory address city")
                )
    return organisations


def find_negative_control_predictions(
    negative_controls: list[EntityRow],
    predictions: list[Prediction],
) -> list[dict[str, Any]]:
    negative_texts = {normalize_surface(row.text) for row in negative_controls}
    return [
        prediction_to_dict(prediction)
        for prediction in predictions
        if normalize_surface(prediction.text) in negative_texts
    ]


def dedupe_predictions(predictions: Iterable[Prediction]) -> list[Prediction]:
    seen: set[tuple[str, str]] = set()
    unique: list[Prediction] = []
    for prediction in predictions:
        if prediction.key in seen:
            continue
        seen.add(prediction.key)
        unique.append(prediction)
    return unique


def dedupe_rows(rows: Iterable[EntityRow]) -> list[EntityRow]:
    seen: set[tuple[str, str]] = set()
    unique: list[EntityRow] = []
    for row in rows:
        if row.key in seen:
            continue
        seen.add(row.key)
        unique.append(row)
    return unique


def write_groundtruth(path: Path, rows: list[EntityRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(GROUNDTRUTH_HEADER)
        for row in rows:
            writer.writerow([row.doc_id, row.text, row.label, row.should_propose, row.notes])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def render_report(result: dict[str, Any]) -> str:
    strict = result["strict"]["metrics"]
    soft = result["soft"]["metrics"]
    lines = [
        "# NER Evaluation",
        "",
        "## Run",
        "",
        f"- Document: `{report_display_path(result['paths']['document'])}`",
        f"- Groundtruth: `{report_display_path(result['paths']['groundtruth'])}`",
        f"- Predictions: `{report_display_path(result['paths']['predictions'])}`",
        f"- Calibration mode: `{result['calibration_mode']}`",
        "",
        "## Summary",
        "",
        "| Mode | TP | FP | FN | Precision | Recall | F1 | F1 95% CrI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("Strict", strict),
        metric_row("Soft", soft),
        "",
        "## Calibration",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in result["calibration"].items())
    lines.extend(["", "## Per Label", ""])
    lines.extend(
        [
            "| Label | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, metrics in result["per_label"].items():
        lines.append(
            f"| {label} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{pct(metrics['precision'])} | {pct(metrics['recall'])} | {pct(metrics['f1'])} |"
        )
    lines.extend(render_error_section("Soft False Positives", result["soft"]["false_positives"]))
    lines.extend(render_error_section("Soft False Negatives", result["soft"]["false_negatives"]))
    lines.extend(render_match_section(result["soft"]["matches"]))
    if result["negative_control_predictions"]:
        lines.extend(["", "## Negative Control Predictions", ""])
        lines.extend(["| Entity | Label |", "|---|---|"])
        for prediction in result["negative_control_predictions"]:
            lines.append(f"| {prediction['text']} | {prediction['label']} |")
    return "\n".join(lines).rstrip() + "\n"


def render_calibration_report(doc_id: str, calibration: dict[str, Any]) -> str:
    lines = [
        "# Groundtruth Calibration",
        "",
        f"- Document: `{doc_id}`",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in calibration["summary"].items())
    lines.extend(
        render_row_section("Absent Groundtruth Rows", calibration["diagnostics"]["absent_rows"])
    )
    lines.extend(
        render_normalization_section(
            "Surface Normalizations",
            calibration["diagnostics"]["normalized_rows"],
        )
    )
    lines.extend(
        render_row_section(
            "Memory Candidates Added",
            calibration["diagnostics"]["added_from_memory"],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_error_section(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", f"## {title}", "", "| Entity | Label | Notes |", "|---|---|---|"]
    if not rows:
        lines.append("| none |  |  |")
        return lines
    for row in rows:
        lines.append(f"| {row['text']} | {row['label']} | {row.get('notes', '')} |")
    return lines


def render_match_section(matches: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Soft Matches",
        "",
        "| Expected | Predicted | Label | Coverage | Threshold |",
        "|---|---|---|---:|---:|",
    ]
    for match in matches:
        lines.append(
            f"| {match['expected']} | {match['predicted']} | {match['label']} | "
            f"{pct(match.get('coverage', 1.0))} | {pct(match.get('threshold', 1.0))} |"
        )
    return lines


def render_row_section(title: str, rows: list[EntityRow]) -> list[str]:
    lines = ["", f"## {title}", "", "| Entity | Label | Notes |", "|---|---|---|"]
    if not rows:
        lines.append("| none |  |  |")
        return lines
    for row in rows:
        lines.append(f"| {row.text} | {row.label} | {row.notes} |")
    return lines


def render_normalization_section(title: str, rows: list[tuple[EntityRow, str]]) -> list[str]:
    lines = ["", f"## {title}", "", "| Groundtruth | Rendered Surface | Label |", "|---|---|---|"]
    if not rows:
        lines.append("| none |  |  |")
        return lines
    for row, rendered_surface in rows:
        lines.append(f"| {row.text} | {rendered_surface} | {row.label} |")
    return lines


def metric_row(name: str, metrics: dict[str, Any]) -> str:
    f1 = metrics["bayesian"]["f1"]
    return (
        f"| {name} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
        f"{pct(metrics['precision'])} | {pct(metrics['recall'])} | {pct(metrics['f1'])} | "
        f"{pct(f1['ci_95_low'])} - {pct(f1['ci_95_high'])} |"
    )


def report_display_path(value: str | None) -> str:
    if value is None:
        return ""
    path = Path(value)
    if path.is_absolute() and "synthetic_dataset_NER" not in path.parts:
        return f".../{path.name}"
    return value


def row_to_dict(row: EntityRow) -> dict[str, Any]:
    return {"text": row.text, "label": row.label, "notes": row.notes}


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    return {"text": prediction.text, "label": prediction.label, "notes": prediction.reason}


def exact_surface_from_text(surface: str, document_text: str) -> str:
    pattern = flexible_surface_pattern(surface)
    match = pattern.search(document_text)
    return match.group(0) if match else surface


def surface_in_text(surface: str, document_text: str) -> bool:
    return flexible_surface_pattern(surface).search(document_text) is not None


def flexible_surface_pattern(surface: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in surface.split()]
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


def expected_token_coverage(expected: str, predicted: str) -> float:
    expected_tokens = tokenize(expected)
    predicted_tokens = set(tokenize(predicted))
    if not expected_tokens:
        return 0.0
    return sum(1 for token in expected_tokens if token in predicted_tokens) / len(expected_tokens)


def tokenize(value: str) -> list[str]:
    return re.findall(r"[\w£€$./-]+", normalize_surface(value))


def normalize_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label).upper()


def beta_map(alpha: int, beta: int) -> float:
    if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
    if alpha <= 1 < beta:
        return 0.0
    if beta <= 1 < alpha:
        return 1.0
    return 0.5


def quantile(samples: list[float], probability: float) -> float:
    if not samples:
        return 0.0
    index = min(round(probability * (len(samples) - 1)), len(samples) - 1)
    return samples[index]


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def split_address(address: str) -> tuple[str, str]:
    if ", " in address:
        street, city_postcode = address.rsplit(", ", 1)
        return street.strip(), city_postcode.strip()
    if "," in address:
        street, city_postcode = address.rsplit(",", 1)
        return street.strip(), city_postcode.strip()
    return address.strip(), ""


def _field_after(value: str, marker: str) -> str:
    index = value.casefold().find(marker)
    if index < 0:
        return ""
    return value[index + len(marker) :].split("|", 1)[0].strip()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .;")


def _with_text(row: EntityRow, text: str) -> EntityRow:
    return EntityRow(row.doc_id, text, row.label, row.should_propose, row.notes)


def _resolve_existing_path(
    *,
    explicit: Path | str | None,
    candidates: list[Path],
    description: str,
) -> Path:
    if explicit is not None:
        return _require_file(Path(explicit), description)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}; checked: {candidate_text}")


def _resolve_optional_path(
    *,
    explicit: Path | str | None,
    candidates: list[Path],
) -> Path | None:
    if explicit is not None:
        return _require_file(Path(explicit), "memory markdown")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path
