"""Prefect flow for controlled morphological NER augmentation."""

from __future__ import annotations

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
    MorphologyTransformation,
)


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
        )
        input_path = selection.input_path
        active_to_passive = selection.active_to_passive
        verbal_to_nominal = selection.verbal_to_nominal
        possessive_reframe = selection.possessive_reframe
        intentional_typos = selection.intentional_typos
        random_layout = selection.random_layout
    transformations = _selected_transformations(
        active_to_passive=active_to_passive,
        verbal_to_nominal=verbal_to_nominal,
        possessive_reframe=possessive_reframe,
        intentional_typos=intentional_typos,
        random_layout=random_layout,
    )
    if not input_path.strip():
        raise ValueError("A document .txt path or package folder path is required")
    resolved_input = resolve_project_path(root, input_path).expanduser().resolve()
    sources = discover_morphology_documents(str(resolved_input))
    results = []
    for source in sources:
        for transformation in transformations:
            try:
                result = create_morphology_variant(
                    source=source,
                    transformation=transformation,
                    project_root=str(root),
                    config_path=config_path,
                    case_config=case_config,
                    contract_path=contract_path,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "source_doc_id": source.doc_id,
                    "transformation": transformation.value,
                    "error": str(exc),
                }
                get_run_logger().error(
                    "Morphology augmentation failed for %s using %s: %s",
                    source.doc_id,
                    transformation.value,
                    exc,
                )
            results.append(result)
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
        raise RuntimeError(
            f"Morphology augmentation completed all requests but {failed} of "
            f"{len(results)} failed; see {report['report_path']}"
        )
    return report


def _review_selection(
    *,
    input_path: str,
    timeout_seconds: int,
    active_to_passive: bool,
    verbal_to_nominal: bool,
    possessive_reframe: bool,
    intentional_typos: bool,
    random_layout: bool,
) -> MorphologyReviewInput:
    review_input = _required_prefilled_input_model(
        MorphologyReviewInput,
        description=(
            "Paste a local .txt, document-package folder, or parent folder path. "
            "Each selected checkbox creates one variant per discovered document."
        ),
        input_path=input_path,
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
            active_to_passive=active_to_passive,
            verbal_to_nominal=verbal_to_nominal,
            possessive_reframe=possessive_reframe,
            intentional_typos=intentional_typos,
            random_layout=random_layout,
        )
    return MorphologyReviewInput(
        input_path=response.input_path,
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
) -> tuple[MorphologyTransformation, ...]:
    selected = tuple(
        transformation
        for enabled, transformation in (
            (active_to_passive, MorphologyTransformation.ACTIVE_TO_PASSIVE),
            (verbal_to_nominal, MorphologyTransformation.VERBAL_TO_NOMINAL),
            (possessive_reframe, MorphologyTransformation.POSSESSIVE_REFRAME),
            (intentional_typos, MorphologyTransformation.INTENTIONAL_TYPOS),
            (random_layout, MorphologyTransformation.RANDOM_LAYOUT),
        )
        if enabled
    )
    if not selected:
        raise ValueError("Select at least one morphological transformation")
    return selected
