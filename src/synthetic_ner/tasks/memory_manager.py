"""Persistent case memory handling."""

from __future__ import annotations

from pathlib import Path

from src.synthetic_ner.tasks.facts import build_allowed_facts_section


class CaseMemoryManager:
    def __init__(self, base_dir: Path, summary_chars: int = 500) -> None:
        self.base_dir = base_dir
        self.summary_chars = summary_chars

    def create_initial_memory(
        self,
        *,
        doc_id: str,
        doc_type: str,
        fraud_type: str,
        document,
        schema: dict,
        section_order: list[str],
    ) -> Path:
        case_dir = self.base_dir / f"case_{doc_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        memory_path = case_dir / "CASE_MEMORY.md"
        memory_path.write_text(
            self._build_initial_memory(
                doc_id=doc_id,
                doc_type=doc_type,
                fraud_type=fraud_type,
                document=document,
                schema=schema,
                section_order=section_order,
            ),
            encoding="utf-8",
        )
        return memory_path

    def read_memory(self, memory_path: Path) -> str:
        return memory_path.read_text(encoding="utf-8")

    def append_document_plan(self, memory_path: Path, document_plan: str) -> None:
        self._append_block(memory_path, "## Document Plan", document_plan.strip())

    def append_section_result(
        self,
        memory_path: Path,
        *,
        section_name: str,
        section_plan: str,
        section_text: str,
        issues: list[str],
    ) -> None:
        summary = section_text.strip().replace("\n", " ")
        if len(summary) > self.summary_chars:
            summary = summary[: self.summary_chars].rstrip() + "..."

        issue_lines = "\n".join(f"- {issue}" for issue in issues) if issues else "- none"
        content = (
            f"Plan:\n{section_plan.strip()}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Issues:\n{issue_lines}"
        )
        self._append_block(memory_path, f"## Section Memory: {section_name}", content)

    def _append_block(self, memory_path: Path, heading: str, content: str) -> None:
        current = memory_path.read_text(encoding="utf-8").rstrip()
        updated = f"{current}\n\n{heading}\n{content.strip()}\n"
        memory_path.write_text(updated, encoding="utf-8")

    def _build_initial_memory(
        self,
        *,
        doc_id: str,
        doc_type: str,
        fraud_type: str,
        document,
        schema: dict,
        section_order: list[str],
    ) -> str:
        metadata = document.metadata
        defendants = "\n".join(
            (
                f"- {person['name_plain']} | role: {person['role']} | "
                f"nationality: {person['nationality']} | address: {person['address']}"
            )
            for person in document.defendants
        ) or "- none"
        collateral = "\n".join(
            (
                f"- {person['name_plain']} | role: {person['role']} | "
                f"nationality: {person['nationality']}"
            )
            for person in document.collateral
        ) or "- none"
        orgs = "\n".join(
            f"- {org['name']} | VAT: {org['vat']} | address: {org['address']}"
            for org in (document.charged_orgs + document.associated_orgs)
        ) or "- none"
        counts = "\n".join(
            (
                f"- {count['offence']} | {count['statute']} | "
                f"{count['particulars']}"
            )
            for count in document.counts_list
        ) or "- none"
        edges = "\n".join(
            f"- {edge['label']}"
            for edge in schema.get("edges", [])
        ) or "- none"
        sections = "\n".join(f"- {section_name}" for section_name in section_order)

        return f"""# CASE_MEMORY

## Document
- Doc ID: {doc_id}
- Document type: {doc_type}
- Fraud type: {fraud_type}
- Court: {metadata['court']}
- Case number: {metadata['case_number']}
- Cross reference: {metadata['cross_ref']}
- Filing date: {metadata['filing_date']}

## Defendants
{defendants}

## Collateral
{collateral}

## Organisations
{orgs}

## Counts
{counts}

## Relationship Graph
{edges}

## Required Sections
{sections}

{build_allowed_facts_section(document, schema)}

## Strict Rules
- Use only the entities, dates, companies, addresses, counts, and relationships listed here.
- Use only the exact person surface forms, VAT numbers, case references, and dates listed above.
- Do not invent new people, organisations, invoice codes, or procedural steps.
- Keep chronology internally consistent across all sections.
- If a fact is missing, omit it rather than guessing it.
"""
