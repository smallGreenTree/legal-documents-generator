"""Prompt context helpers for section-level generation."""

from __future__ import annotations

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS


def build_section_context(memory_text: str, section_name: str) -> str:
    """Build a compact evidence packet for one section prompt."""

    common_parts = [
        "# SECTION_CONTEXT",
        "",
        "## Section",
        f"- Name: {section_name}",
        f"- Purpose: {SECTION_DESCRIPTIONS.get(section_name, section_name)}",
        "",
        "## Required Treatment",
        *_section_guidance(section_name),
        "",
        "## Case Metadata",
        _extract_markdown_block(memory_text, "Document"),
        "",
        "## Allowed References",
        _extract_markdown_block(memory_text, "Allowed References"),
        "",
        "## Strict Rules",
        _extract_markdown_block(memory_text, "Strict Rules"),
    ]
    parts = [*common_parts, *_section_memory_parts(memory_text, section_name)]
    return "\n".join(part for part in parts if part.strip()).strip()


def _section_guidance(section_name: str) -> list[str]:
    guidance = {
        "persons": [
            "- Focus on defendant identity only.",
            "- Include names, dates of birth, nationality, roles and addresses when available.",
            "- Do not introduce allegations beyond identity and procedural role.",
        ],
        "companies": [
            "- Focus on charged company identity only.",
            "- Include registered names, VAT numbers and addresses when available.",
            "- Do not infer directors, registration dates or business activities.",
        ],
        "history": [
            "- Build a procedural and chronological bridge into the charges.",
            "- Use filing date, case references and offence period as anchors.",
            "- Avoid relisting every person and organisation.",
        ],
        "charges": [
            "- Use compact charge language tied to counts and charged period.",
            "- Connect only defendants, organisations, statutes and particulars present in context.",
            "- Do not invent count numbers, exhibits or statutory wording.",
        ],
        "facts": [
            "- Build the main chronological narrative from relationship facts and count particulars.",
            "- Connect control, instruction, conspiracy and fund-flow facts only when recorded.",
            "- Prefer concrete allowed references over generic accusations.",
        ],
        "evidence": [
            "- Describe evidence categories only from recorded relationships and references.",
            "- Explain relevance without inventing searches, emails, invoices or exhibit labels.",
            "- Keep the tone evidential, not argumentative.",
        ],
        "assessment": [
            "- Draw legal characterisation from already recorded facts.",
            "- Explain why the facts support the charges without adding new facts.",
            "- Avoid repeating the full factual chronology.",
        ],
    }
    return guidance.get(
        section_name,
        [
            "- Use only facts in this context.",
            "- Write section-specific prose rather than a general case summary.",
        ],
    )


def _section_memory_parts(memory_text: str, section_name: str) -> list[str]:
    section = section_name.lower()
    if section == "persons":
        return [
            "",
            "## Defendants",
            _extract_markdown_block(memory_text, "Defendants"),
            "",
            "## Collateral",
            _extract_markdown_block(memory_text, "Collateral"),
        ]
    if section == "companies":
        return [
            "",
            "## Organisations",
            _extract_markdown_block(memory_text, "Organisations"),
        ]
    if section == "history":
        return [
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
        ]
    if section == "charges":
        return [
            "",
            "## Defendants",
            _extract_markdown_block(memory_text, "Defendants"),
            "",
            "## Organisations",
            _extract_markdown_block(memory_text, "Organisations"),
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
        ]
    if section == "facts":
        return [
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
            "",
            "## Relationship Facts",
            _extract_markdown_block(memory_text, "Relationship Graph"),
        ]
    if section == "evidence":
        return [
            "",
            "## Organisations",
            _extract_markdown_block(memory_text, "Organisations"),
            "",
            "## Relationship Facts",
            _extract_markdown_block(memory_text, "Relationship Graph"),
        ]
    if section == "assessment":
        return [
            "",
            "## Counts",
            _extract_markdown_block(memory_text, "Counts"),
            "",
            "## Relationship Facts",
            _extract_markdown_block(memory_text, "Relationship Graph"),
        ]
    return [
        "",
        "## Counts",
        _extract_markdown_block(memory_text, "Counts"),
        "",
        "## Relationship Facts",
        _extract_markdown_block(memory_text, "Relationship Graph"),
    ]


def _extract_markdown_block(memory_text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = memory_text.find(marker)
    if start == -1:
        return "- none"

    start_index = start + len(marker)
    tail = memory_text[start_index:]
    next_h2 = tail.find("\n## ")
    block = tail[:next_h2] if next_h2 != -1 else tail
    return block.strip() or "- none"
