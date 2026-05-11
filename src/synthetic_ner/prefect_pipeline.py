"""Prefect orchestration for audited synthetic NER generation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.context import get_run_context
from src.synthetic_ner.cli import load_env_files
from src.synthetic_ner.engine import (
    build_runtime_context,
    resolve_document_inputs,
    resolve_schema_for_document,
)
from src.synthetic_ner.tasks.orchestrator import run_document_graph

ARTIFACT_TEXT_LIMIT = 24_000


@task(name="configs-ingestion")
def ingest_configs(
    *,
    project_root: Path,
    case_config: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
) -> Any:
    """Load config files and build the runtime context."""
    load_env_files(project_root)
    args = _build_args(
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
    )
    context = build_runtime_context(args, project_root)
    logger = get_run_logger()
    logger.info(
        "Loaded configs: doc_type=%s fraud_type=%s documents=%s output_dir=%s",
        context.doc_type,
        context.fraud_type,
        context.documents,
        context.output_dir,
    )
    _publish_config_artifacts(
        project_root=project_root,
        case_config=case_config,
        context=context,
    )
    return context


@task(name="faker-entities")
def resolve_entities(context: Any) -> Any:
    """Generate or normalize people, organisations, addresses and counts."""
    document = resolve_document_inputs(context)
    logger = get_run_logger()
    logger.info(
        "Resolved entities: defendants=%s collateral=%s charged_orgs=%s associated_orgs=%s",
        len(document.defendants),
        len(document.collateral),
        len(document.charged_orgs),
        len(document.associated_orgs),
    )
    _publish_entity_artifacts(document)
    return document


@task(name="case-schema")
def build_case_schema(context: Any, document: Any, document_index: int) -> tuple[str, dict]:
    """Build or load the relationship schema for one document."""
    doc_id, schema = resolve_schema_for_document(context, document, document_index)
    get_run_logger().info("Resolved schema for %s with %s edges", doc_id, len(schema["edges"]))
    _publish_schema_artifacts(doc_id, schema)
    return doc_id, schema


@task(name="langgraph-langfuse-generation")
def run_langgraph_langfuse(context: Any, document: Any, schema: dict, doc_id: str) -> str:
    """Run the LangGraph generation workflow with Langfuse tracing enabled by config."""
    try:
        run_document_graph(
            context=context,
            document=document,
            schema=schema,
            doc_id=doc_id,
        )
    finally:
        _publish_memory_artifacts(context, doc_id)
    get_run_logger().info("Completed LangGraph/Langfuse generation for %s", doc_id)
    return doc_id


@task(name="end-of-pipeline-file-audit")
def audit_created_files(context: Any, doc_id: str) -> Path:
    """Write a document-level manifest of generated files and checksums."""
    audit_path = context.output_dir / doc_id / "file_audit.json"
    files = _collect_document_files(context, doc_id, exclude={audit_path})
    payload = {
        "doc_id": doc_id,
        "created_at": _utc_now(),
        "roots": {
            "output": str(context.output_dir / doc_id),
            "partial_output": str(context.output_dir / "_partial" / doc_id),
            "schema": str(context.schema_dir / f"{doc_id}.json"),
            "memory": str(context.memory_dir / f"case_{doc_id}"),
        },
        "files": files,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["files"].append(_file_record(audit_path))
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_prefect_artifacts(context, doc_id, payload)
    get_run_logger().info("Wrote file audit for %s to %s", doc_id, audit_path)
    return audit_path


@flow(name="synthetic-ner-generation")
def generate_dataset(
    case_config: str = "config_case/case_1.yaml",
    documents: int | None = None,
    doc_type: str | None = None,
    fraud_type: str | None = None,
    from_schema: str | None = None,
    project_root: str | None = None,
) -> list[str]:
    """Orchestrate a complete generation run with Prefect-visible pipeline stages."""
    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else Path(__file__).resolve().parents[2]
    )
    context = ingest_configs(
        project_root=resolved_project_root,
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
    )

    doc_ids: list[str] = []
    for document_index in range(context.documents):
        document = resolve_entities(context)
        doc_id, schema = build_case_schema(context, document, document_index)
        run_langgraph_langfuse(context, document, schema, doc_id)
        audit_created_files(context, doc_id)
        doc_ids.append(doc_id)

    get_run_logger().info("Generation flow completed for documents: %s", ", ".join(doc_ids))
    return doc_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic NER generation through Prefect.")
    parser.add_argument("--case-config", default="config_case/case_1.yaml")
    parser.add_argument("--documents", "--count", dest="documents", type=int, default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--fraud-type", default=None)
    parser.add_argument("--from-schema", default=None)
    parser.add_argument("--project-root", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(
        case_config=args.case_config,
        documents=args.documents,
        doc_type=args.doc_type,
        fraud_type=args.fraud_type,
        from_schema=args.from_schema,
        project_root=args.project_root,
    )


def _build_args(
    *,
    case_config: str,
    documents: int | None,
    doc_type: str | None,
    fraud_type: str | None,
    from_schema: str | None,
) -> Namespace:
    return Namespace(
        case_config=case_config,
        documents=documents,
        doc_type=doc_type,
        fraud_type=fraud_type,
        from_schema=from_schema,
        workflow_mode="langgraph",
    )


def _collect_document_files(
    context: Any,
    doc_id: str,
    *,
    exclude: set[Path],
) -> list[dict[str, Any]]:
    roots = [
        context.output_dir / doc_id,
        context.output_dir / "_partial" / doc_id,
        context.schema_dir / f"{doc_id}.json",
        context.memory_dir / f"case_{doc_id}",
    ]
    excluded = {path.resolve() for path in exclude}
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        paths = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in paths:
            resolved = path.resolve()
            if resolved in excluded or resolved in seen:
                continue
            records.append(_file_record(path))
            seen.add(resolved)
    return sorted(records, key=lambda item: item["path"])


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
    }


def _publish_config_artifacts(*, project_root: Path, case_config: str, context: Any) -> None:
    case_config_path = project_root / case_config
    template_path = project_root / "templates" / f"en_{context.doc_type}.j2"
    create_markdown_artifact(
        key=_artifact_key("run-config-summary"),
        description="Run configuration summary",
        markdown="\n".join(
            [
                "# Run Configuration",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Document type | `{context.doc_type}` |",
                f"| Fraud type | `{context.fraud_type}` |",
                f"| Documents | `{context.documents}` |",
                f"| Output directory | `{context.output_dir}` |",
                f"| Schema directory | `{context.schema_dir}` |",
                f"| Memory directory | `{context.memory_dir}` |",
                f"| Case config | `{case_config_path}` |",
                f"| Template | `{template_path}` |",
                f"| Workflow mode | `{context.workflow_cfg.mode}` |",
            ]
        ),
    )
    _publish_file_markdown(
        key=_artifact_key("config-yaml"),
        description="Root config.yaml used for the run",
        path=project_root / "config.yaml",
        language="yaml",
    )
    _publish_file_markdown(
        key=_artifact_key("case-config"),
        description="Case/scenario config used for the run",
        path=case_config_path,
        language="yaml",
    )
    _publish_file_markdown(
        key=_artifact_key("document-template"),
        description="Jinja document template used for the run",
        path=template_path,
        language="jinja",
    )


def _publish_entity_artifacts(document: Any) -> None:
    create_table_artifact(
        key=_artifact_key("entity-summary"),
        description="Resolved entity counts from Faker or explicit config",
        table=[
            {"entity_group": "defendants", "count": len(document.defendants)},
            {"entity_group": "collateral", "count": len(document.collateral)},
            {"entity_group": "charged_orgs", "count": len(document.charged_orgs)},
            {"entity_group": "associated_orgs", "count": len(document.associated_orgs)},
        ],
    )
    create_table_artifact(
        key=_artifact_key("resolved-people"),
        description="Resolved people for this run",
        table=[
            {
                "name": person["name"],
                "role": person["role"],
                "nationality": person["nationality"],
                "is_defendant": person["is_defendant"],
                "surface_forms": ", ".join(person["surface_forms_list"]),
            }
            for person in [*document.defendants, *document.collateral]
        ],
    )
    create_table_artifact(
        key=_artifact_key("resolved-organisations"),
        description="Resolved organisations for this run",
        table=[
            {
                "name": org["name"],
                "vat": org["vat"],
                "address": org["address"],
                "group": "charged"
                if org in document.charged_orgs
                else "associated",
            }
            for org in [*document.charged_orgs, *document.associated_orgs]
        ],
    )


def _publish_schema_artifacts(doc_id: str, schema: dict) -> None:
    create_markdown_artifact(
        key=_artifact_key(doc_id, "schema-json"),
        description=f"Case schema JSON for {doc_id}",
        markdown=(
            f"# Case Schema: `{doc_id}`\n\n"
            "```json\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n"
            "```"
        ),
    )
    create_table_artifact(
        key=_artifact_key(doc_id, "schema-edges"),
        description=f"Relationship graph edges for {doc_id}",
        table=[
            {
                "from": edge.get("from"),
                "to": edge.get("to"),
                "type": edge.get("type"),
                "label": edge.get("label"),
            }
            for edge in schema.get("edges", [])
        ],
    )


def _publish_memory_artifacts(context: Any, doc_id: str) -> None:
    memory_dir = context.memory_dir / f"case_{doc_id}"
    _publish_file_markdown(
        key=_artifact_key(doc_id, "case-memory"),
        description=f"CASE_MEMORY.md for {doc_id}",
        path=memory_dir / "CASE_MEMORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "run-history"),
        description=f"RUN_HISTORY.md for {doc_id}",
        path=memory_dir / "RUN_HISTORY.md",
    )


def _publish_prefect_artifacts(context: Any, doc_id: str, audit_payload: dict[str, Any]) -> None:
    doc_dir = context.output_dir / doc_id
    memory_dir = context.memory_dir / f"case_{doc_id}"
    schema_path = context.schema_dir / f"{doc_id}.json"
    document_path = doc_dir / f"{doc_id}.txt"
    report_path = doc_dir / "generation_report.md"
    audit_path = doc_dir / "file_audit.json"

    create_markdown_artifact(
        key=_artifact_key(doc_id, "run-summary"),
        description=f"Generated document run summary for {doc_id}",
        markdown=_run_summary_markdown(
            doc_id=doc_id,
            document_path=document_path,
            schema_path=schema_path,
            memory_dir=memory_dir,
            report_path=report_path,
            audit_path=audit_path,
        ),
    )
    create_table_artifact(
        key=_artifact_key(doc_id, "created-files"),
        description=f"Created files for {doc_id}",
        table=[
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in audit_payload["files"]
        ],
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "case-memory-final"),
        description=f"CASE_MEMORY.md for {doc_id}",
        path=memory_dir / "CASE_MEMORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "run-history-final"),
        description=f"RUN_HISTORY.md for {doc_id}",
        path=memory_dir / "RUN_HISTORY.md",
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "document-text"),
        description=f"Generated document text for {doc_id}",
        path=document_path,
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "generation-report"),
        description=f"Generation report for {doc_id}",
        path=report_path,
    )
    _publish_file_markdown(
        key=_artifact_key(doc_id, "schema-json-final"),
        description=f"Case schema JSON for {doc_id}",
        path=schema_path,
        language="json",
    )


def _run_summary_markdown(
    *,
    doc_id: str,
    document_path: Path,
    schema_path: Path,
    memory_dir: Path,
    report_path: Path,
    audit_path: Path,
) -> str:
    return "\n".join(
        [
            f"# Generated Document: `{doc_id}`",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Document text | `{document_path}` |",
            f"| Case schema | `{schema_path}` |",
            f"| Case memory | `{memory_dir / 'CASE_MEMORY.md'}` |",
            f"| Run history | `{memory_dir / 'RUN_HISTORY.md'}` |",
            f"| Generation report | `{report_path}` |",
            f"| File audit | `{audit_path}` |",
        ]
    )


def _publish_file_markdown(
    *,
    key: str,
    description: str,
    path: Path,
    language: str = "markdown",
) -> None:
    if not path.exists():
        create_markdown_artifact(
            key=key,
            description=description,
            markdown=f"# Missing Artifact\n\nFile was not found:\n\n`{path}`",
        )
        return

    text = path.read_text(encoding="utf-8")
    suffix = ""
    if len(text) > ARTIFACT_TEXT_LIMIT:
        text = text[:ARTIFACT_TEXT_LIMIT].rstrip()
        suffix = (
            "\n\n_Artifact truncated for Prefect UI display. "
            f"Open the local file for full content: `{path}`_"
        )
    create_markdown_artifact(
        key=key,
        description=description,
        markdown=f"# `{path.name}`\n\n```{language}\n{text}\n```{suffix}",
    )


def _artifact_key(*parts: object) -> str:
    try:
        run_id = str(get_run_context().flow_run.id)[:8]
    except Exception:
        run_id = "local"
    raw = "-".join([run_id, *(str(part) for part in parts if part is not None)])
    key = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    return re.sub(r"-{2,}", "-", key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
