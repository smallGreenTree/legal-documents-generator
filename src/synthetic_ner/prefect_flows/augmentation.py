"""Prefect flow for controlled morphological NER augmentation."""

from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger
from prefect.flow_runs import pause_flow_run

from src.synthetic_ner.core.paths import resolve_project_path
from src.synthetic_ner.prefect_flows.utils import (
    _required_prefilled_input_model,
    resolve_flow_project_root,
)
from src.synthetic_ner.tasks.augmentation.prefect_tasks import (
    create_morphology_variant,
    discover_morphology_documents,
    publish_morphology_batch_report,
)
from src.synthetic_ner.types.augmentation import (
    MorphologyReviewInput,
    MorphologySource,
    MorphologyTransformation,
)


@flow(name="synthetic-ner-morphological-document")
def generate_document_morphological_variations(
    *,
    source: MorphologySource,
    transformations: tuple[MorphologyTransformation, ...],
    project_root: str,
    config_path: str,
    case_config: str,
    contract_path: str,
    style: str,
    style_temperature: float,
    reformat_with_style: bool,
) -> list[dict[str, Any]]:
    """Create every requested morphology variant for one source document."""
    results: list[dict[str, Any]] = []
    for transformation in transformations:
        requested_style, requested_temperature, requested_reformat = _variant_options(
            transformation=transformation,
            style=style,
            style_temperature=style_temperature,
            reformat_with_style=reformat_with_style,
        )
        try:
            result = create_morphology_variant(
                source=source,
                transformation=transformation,
                project_root=project_root,
                config_path=config_path,
                case_config=case_config,
                contract_path=contract_path,
                style=requested_style,
                style_temperature=requested_temperature,
                reformat_with_style=requested_reformat,
            )
        except Exception as exc:
            result = _failed_variant_result(
                source=source,
                transformation=transformation,
                style=requested_style,
                style_temperature=requested_temperature,
                reformat_with_style=requested_reformat,
                error=exc,
            )
            get_run_logger().error(
                "Morphology augmentation failed for %s using %s: %s",
                source.doc_id,
                transformation.value,
                exc,
            )
        results.append(result)
    return results


@flow(name="synthetic-ner-morphological-augmentation")
def generate_morphological_variations(
    input_path: str = "",
    project_root: str | None = None,
    config_path: str = "config.yaml",
    case_config: str = "config_case/case_1.yaml",
    contract_path: str = "groundtruth_contract.yaml",
    review: bool = True,
    review_timeout_seconds: int = 3600,
    active_to_passive: bool = True,
    verbal_to_nominal: bool = True,
    possessive_reframe: bool = True,
    intentional_typos: bool = False,
    random_layout: bool = False,
    style: str = "",
    style_temperature: float = 0.8,
    reformat_with_style: bool = True,
) -> dict:
    """Create one validated package per selected transformation and source document."""
    root = resolve_flow_project_root(project_root)
    if review:
        selection = _review_selection(
            input_path=input_path,
            timeout_seconds=review_timeout_seconds,
            active_to_passive=active_to_passive,
            verbal_to_nominal=verbal_to_nominal,
            possessive_reframe=possessive_reframe,
            intentional_typos=intentional_typos,
            random_layout=random_layout,
            style=style,
            style_temperature=style_temperature,
            reformat_with_style=reformat_with_style,
        )
        input_path = selection.input_path
        active_to_passive = selection.active_to_passive
        verbal_to_nominal = selection.verbal_to_nominal
        possessive_reframe = selection.possessive_reframe
        intentional_typos = selection.intentional_typos
        random_layout = selection.random_layout
        style = selection.style
        style_temperature = selection.style_temperature
        reformat_with_style = selection.reformat_with_style
    transformations = _selected_transformations(
        active_to_passive=active_to_passive,
        verbal_to_nominal=verbal_to_nominal,
        possessive_reframe=possessive_reframe,
        intentional_typos=intentional_typos,
        random_layout=random_layout,
        style=style,
    )
    if not input_path.strip():
        raise ValueError("A document .txt path or package folder path is required")
    resolved_input = resolve_project_path(root, input_path).expanduser().resolve()
    resolved_contract = resolve_project_path(root, contract_path)
    sources = discover_morphology_documents(
        str(resolved_input),
        str(resolved_contract),
    )
    results: list[dict[str, Any]] = []
    for source in sources:
        try:
            document_results = generate_document_morphological_variations(
                source=source,
                transformations=transformations,
                project_root=str(root),
                config_path=config_path,
                case_config=case_config,
                contract_path=contract_path,
                style=style,
                style_temperature=style_temperature,
                reformat_with_style=reformat_with_style,
            )
        except Exception as exc:
            document_results = _failed_document_results(
                source=source,
                transformations=transformations,
                style=style,
                style_temperature=style_temperature,
                reformat_with_style=reformat_with_style,
                error=exc,
            )
            get_run_logger().error(
                "Morphology document flow failed for %s: %s",
                source.doc_id,
                exc,
            )
        results.extend(document_results)
    report = publish_morphology_batch_report(
        input_path=str(resolved_input),
        results=results,
    )
    get_run_logger().info(
        "Morphology augmentation completed: %s document(s), %s variant(s)",
        len(sources),
        len(results),
    )
    failed = sum(result.get("status") != "completed" for result in results)
    if failed:
        get_run_logger().warning(
            "Morphology augmentation completed all documents with %s of %s variants failed; see %s",
            failed,
            len(results),
            report["report_path"],
        )
    return report


def _variant_options(
    *,
    transformation: MorphologyTransformation,
    style: str,
    style_temperature: float,
    reformat_with_style: bool,
) -> tuple[str | None, float | None, bool]:
    if transformation is MorphologyTransformation.CUSTOM_STYLE:
        return style, style_temperature, reformat_with_style
    return None, None, False


def _failed_variant_result(
    *,
    source: MorphologySource,
    transformation: MorphologyTransformation,
    style: str | None,
    style_temperature: float | None,
    reformat_with_style: bool,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "source_doc_id": source.doc_id,
        "transformation": transformation.value,
        "style": style,
        "style_temperature": style_temperature,
        "reformat_with_style": reformat_with_style,
        "error": str(error),
    }


def _failed_document_results(
    *,
    source: MorphologySource,
    transformations: tuple[MorphologyTransformation, ...],
    style: str,
    style_temperature: float,
    reformat_with_style: bool,
    error: Exception,
) -> list[dict[str, Any]]:
    return [
        _failed_variant_result(
            source=source,
            transformation=transformation,
            style=requested_style,
            style_temperature=requested_temperature,
            reformat_with_style=requested_reformat,
            error=error,
        )
        for transformation in transformations
        for requested_style, requested_temperature, requested_reformat in (
            _variant_options(
                transformation=transformation,
                style=style,
                style_temperature=style_temperature,
                reformat_with_style=reformat_with_style,
            ),
        )
    ]


def _review_selection(
    *,
    input_path: str,
    timeout_seconds: int,
    active_to_passive: bool,
    verbal_to_nominal: bool,
    possessive_reframe: bool,
    intentional_typos: bool,
    random_layout: bool,
    style: str,
    style_temperature: float,
    reformat_with_style: bool,
) -> MorphologyReviewInput:
    review_input = _required_prefilled_input_model(
        MorphologyReviewInput,
        description=(
            "Paste a local .txt, document-package folder, or parent folder path. "
            "Each selected checkbox creates one variant per discovered document. "
            "CUSTOM STYLE REQUEST creates one additional variant; its temperature "
            "and reformat switch do not affect the checkbox transformations."
        ),
        input_path=input_path,
        style=style,
        style_temperature=style_temperature,
        reformat_with_style=reformat_with_style,
        active_to_passive=active_to_passive,
        verbal_to_nominal=verbal_to_nominal,
        possessive_reframe=possessive_reframe,
        intentional_typos=intentional_typos,
        random_layout=random_layout,
    )
    response = pause_flow_run(
        wait_for_input=review_input,
        timeout=timeout_seconds,
        key="morphology-augmentation-selection",
    )
    if response is None:
        return MorphologyReviewInput(
            input_path=input_path,
            style=style,
            style_temperature=style_temperature,
            reformat_with_style=reformat_with_style,
            active_to_passive=active_to_passive,
            verbal_to_nominal=verbal_to_nominal,
            possessive_reframe=possessive_reframe,
            intentional_typos=intentional_typos,
            random_layout=random_layout,
        )
    return MorphologyReviewInput(
        input_path=response.input_path,
        style=response.style,
        style_temperature=response.style_temperature,
        reformat_with_style=response.reformat_with_style,
        active_to_passive=response.active_to_passive,
        verbal_to_nominal=response.verbal_to_nominal,
        possessive_reframe=response.possessive_reframe,
        intentional_typos=response.intentional_typos,
        random_layout=response.random_layout,
    )


def _selected_transformations(
    *,
    active_to_passive: bool,
    verbal_to_nominal: bool,
    possessive_reframe: bool,
    intentional_typos: bool,
    random_layout: bool,
    style: str,
) -> tuple[MorphologyTransformation, ...]:
    selected = tuple(
        transformation
        for enabled, transformation in (
            (active_to_passive, MorphologyTransformation.ACTIVE_TO_PASSIVE),
            (verbal_to_nominal, MorphologyTransformation.VERBAL_TO_NOMINAL),
            (possessive_reframe, MorphologyTransformation.POSSESSIVE_REFRAME),
            (intentional_typos, MorphologyTransformation.INTENTIONAL_TYPOS),
            (random_layout, MorphologyTransformation.RANDOM_LAYOUT),
            (bool(style.strip()), MorphologyTransformation.CUSTOM_STYLE),
        )
        if enabled
    )
    if not selected:
        raise ValueError("Select at least one transformation or provide a custom style")
    return selected
