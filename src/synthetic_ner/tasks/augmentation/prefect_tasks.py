"""Prefect-visible tasks for morphological augmentation."""

from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.configuration.augmentation import load_morphology_workflow_config
from src.synthetic_ner.configuration.loader import load_app_config
from src.synthetic_ner.core.paths import resolve_project_path
from src.synthetic_ner.model_providers.factory import build_model_client
from src.synthetic_ner.tasks.augmentation import (
    build_variant_id,
    discover_morphology_sources,
)
from src.synthetic_ner.tasks.augmentation.constants import (
    DETERMINISTIC_TRANSFORMATIONS,
    MORPHOLOGY_MODEL_STAGE,
    MORPHOLOGY_PIPELINE_STAGE,
)
from src.synthetic_ner.tasks.augmentation.morphology import MorphologyAugmenter
from src.synthetic_ner.tasks.augmentation.publication import (
    existing_variant_result,
    publish_batch_report,
    publish_morphology_variant,
)
from src.synthetic_ner.tasks.augmentation.style import (
    normalize_style,
    normalize_style_temperature,
)
from src.synthetic_ner.tasks.document_generation.observability.tracer import TraceStore
from src.synthetic_ner.types.augmentation import MorphologySource, MorphologyTransformation


@task(name="discover-morphology-documents")
def discover_morphology_documents(
    input_path: str,
    contract_path: str,
) -> list[MorphologySource]:
    sources = discover_morphology_sources(input_path, contract_path=contract_path)
    get_run_logger().info(
        "Discovered %s validated document package(s) for morphology augmentation",
        len(sources),
    )
    return sources


@task(name="create-morphology-variant")
def create_morphology_variant(
    *,
    source: MorphologySource,
    transformation: MorphologyTransformation,
    project_root: str,
    config_path: str,
    case_config: str,
    contract_path: str,
    style: str | None = None,
    style_temperature: float | None = None,
    reformat_with_style: bool = False,
) -> dict[str, Any]:
    root = Path(project_root)
    resolved_config = resolve_project_path(root, config_path)
    morphology_config = load_morphology_workflow_config(resolved_config)
    normalized_style = (
        normalize_style(style) if transformation is MorphologyTransformation.CUSTOM_STYLE else None
    )
    effective_style_temperature = (
        normalize_style_temperature(
            morphology_config.style_temperature if style_temperature is None else style_temperature
        )
        if transformation is MorphologyTransformation.CUSTOM_STYLE
        else None
    )
    effective_reformat = bool(reformat_with_style) and (
        transformation is MorphologyTransformation.CUSTOM_STYLE
    )
    variant_doc_id = build_variant_id(
        source.doc_id,
        transformation,
        style=normalized_style,
        style_temperature=effective_style_temperature,
        reformat_with_style=effective_reformat,
    )
    existing = existing_variant_result(
        source,
        variant_doc_id,
        transformation,
        style=normalized_style,
        style_temperature=effective_style_temperature,
        reformat_with_style=effective_reformat,
    )
    if existing is not None:
        get_run_logger().info("Reusing completed morphology variant %s", variant_doc_id)
        return existing

    load_env_files(root)
    resolved_contract = resolve_project_path(root, contract_path)
    if transformation in DETERMINISTIC_TRANSFORMATIONS:
        variant = MorphologyAugmenter(client=None, config=morphology_config).create_variant(
            source=source,
            transformation=transformation,
            variant_doc_id=variant_doc_id,
            contract_path=resolved_contract,
            style=normalized_style,
            style_temperature=effective_style_temperature,
            reformat_with_style=effective_reformat,
        )
        result = publish_morphology_variant(
            source=source,
            variant=variant,
            contract_path=resolved_contract,
        )
        get_run_logger().info(
            "Created %s from %s using %s",
            variant_doc_id,
            source.doc_id,
            transformation.value,
        )
        return result

    app_config = load_app_config(
        resolved_config,
        resolve_project_path(root, case_config),
    )
    tracer = TraceStore(
        app_config.mlflow,
        run_metadata={
            "doc_id": variant_doc_id,
            "workflow_run_id": variant_doc_id,
            "source_doc_id": source.doc_id,
            "transformation": transformation.value,
            "style": normalized_style,
            "style_temperature": effective_style_temperature,
            "reformat_with_style": effective_reformat,
        },
    )
    tracer.start_document_run(
        doc_id=variant_doc_id,
        input_payload={
            "source_doc_id": source.doc_id,
            "transformation": transformation.value,
            "style": normalized_style,
            "style_temperature": effective_style_temperature,
            "reformat_with_style": effective_reformat,
        },
        metadata={
            "source_doc_id": source.doc_id,
            "transformation": transformation.value,
            "pipeline_stage": MORPHOLOGY_PIPELINE_STAGE,
            "style_temperature": effective_style_temperature,
            "reformat_with_style": effective_reformat,
        },
    )
    result: dict[str, Any] | None = None
    try:
        augmenter = MorphologyAugmenter(
            client=build_model_client(
                stage=MORPHOLOGY_MODEL_STAGE,
                routing=app_config.model_routing,
                tracer=tracer,
            ),
            config=morphology_config,
        )
        variant = augmenter.create_variant(
            source=source,
            transformation=transformation,
            variant_doc_id=variant_doc_id,
            contract_path=resolved_contract,
            style=normalized_style,
            style_temperature=effective_style_temperature,
            reformat_with_style=effective_reformat,
        )
        if variant.style_fallback_chunk_indices:
            get_run_logger().warning(
                "Preserved %s unsafe morphology chunk(s) unchanged: %s",
                len(variant.style_fallback_chunk_indices),
                variant.style_fallback_chunk_indices,
            )
        result = publish_morphology_variant(
            source=source,
            variant=variant,
            contract_path=resolved_contract,
        )
        get_run_logger().info(
            "Created %s from %s using %s",
            variant_doc_id,
            source.doc_id,
            transformation.value,
        )
        return result
    finally:
        tracer.end_document_run(output_payload=result)


@task(name="publish-morphology-batch-report")
def publish_morphology_batch_report(
    *,
    input_path: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    report = publish_batch_report(input_path=input_path, results=results)
    get_run_logger().info("Morphology batch report: %s", report["report_path"])
    return report
