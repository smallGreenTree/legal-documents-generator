"""Constants shared across the document-generation package."""

from __future__ import annotations

import re
from pathlib import Path

from src.synthetic_ner.core.constants import COMPANY_SUFFIXES

SECTION_CONTRACTS_PATH = Path(__file__).resolve().parents[4] / "prompts" / "section_contracts.yaml"

PROSE_SECTION_ORDER = {
    "indictment": [
        "persons",
        "companies",
        "history",
        "charges",
        "facts",
        "evidence",
        "assessment",
    ],
    "court_decision": [
        "background",
        "findings",
        "conclusions",
        "sentence",
    ],
}

SECTION_DEPENDENCIES = {
    "facts": frozenset({"history", "charges"}),
    "evidence": frozenset({"facts"}),
    "assessment": frozenset({"facts"}),
}

SECTION_DESCRIPTIONS = {
    "persons": (
        "Persons section: identify the defendants in natural legal prose, "
        "preserving exact names, dates of birth, birthplaces, nationalities, "
        "roles and addresses."
    ),
    "companies": (
        "Companies section: identify the charged organisations in natural legal "
        "prose, preserving exact company names, addresses and VAT numbers."
    ),
    "history": (
        "Procedural history: how the investigation started, search warrants, "
        "key dates, documents seized."
    ),
    "charges": "Charges: precise allegations against each defendant and each company.",
    "facts": (
        "Statement of facts: detailed narrative with specific dates, GBP/EUR "
        "amounts, addresses, invoice references, document codes."
    ),
    "evidence": (
        "Evidence: a numbered list of exhibits (search records, bank "
        "statements, emails, invoices, witness statements)."
    ),
    "assessment": (
        "Legal assessment and motion: provisional legal characterisation of "
        "the conduct and a closing paragraph requesting the court to open "
        "proceedings."
    ),
    "background": (
        "Background: how the matter came before the court, investigation history, procedural steps."
    ),
    "findings": (
        "Findings of fact: what the court finds proved, with specific dates, "
        "amounts, addresses and document references."
    ),
    "conclusions": (
        "Legal conclusions: how the court characterises the conduct and which "
        "statutory provisions apply."
    ),
    "sentence": (
        "Sentence and order: custodial term or order for each defendant, plus "
        "any confiscation or disqualification orders."
    ),
}

MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE_TEXT_PATTERN = rf"\d{{1,2}} (?:{MONTH_PATTERN}) \d{{4}}"
DATE_RE = re.compile(rf"\b{DATE_TEXT_PATTERN}\b")
VAT_RE = re.compile(r"\b[A-Z]{2}(?=[A-Z0-9]{8,14}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{8,14}\b")
CASE_REF_RE = re.compile(r"\b(?:CPS/\d{4}/\d{4}|C/\d{4}/\d{1,4}|T\d{9,10}|\d{7}/\d{3})\b")
INITIALS_RE = re.compile(r"\b(?:[A-Z]\.){2,4}\b")
TITLE_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lord)\.? [A-Z][A-Za-z'-]+\b")
ORG_SUFFIX_PATTERN = "|".join(
    sorted((re.escape(suffix) for suffix in COMPANY_SUFFIXES), key=len, reverse=True)
)
ORG_NAME_RE = re.compile(
    rf"\b[A-Z0-9][A-Z0-9&'/-]*(?: [A-Z0-9][A-Z0-9&'/-]*)* "
    rf"(?:{ORG_SUFFIX_PATTERN})\b"
)

RUBRIC_LINE_RE = re.compile(
    r"^\s*-\s*([a-zA-Z][a-zA-Z0-9_ -]{1,40})\s*:\s*([1-5])(?:\s*/\s*5)?\s*$",
    re.MULTILINE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
META_TOKEN_RE = re.compile(
    r"(?im)\b(?:APPROVED:|RUBRICS:|ISSUES:|REVISION:|STRICT COMPLIANCE NOTES:|"
    r"REQUIRED FACTS(?:\s*&\s*ENTITIES)?:|ENTITIES MENTIONED:|LOGICAL ORDER:|FIX:)\b"
)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+")
MARKDOWN_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
MARKDOWN_NUMBERED_RE = re.compile(r"(?m)^\s*\d+\.\s+")
MARKDOWN_RULE_RE = re.compile(r"(?m)^\s*[-*_]{3,}\s*$")
MARKDOWN_BOLD_RE = re.compile(r"\*\*")
PLACEHOLDER_STARS_RE = re.compile(r"\*{4,}")
WORD_COUNT_RE = re.compile(r"(?im)^\s*\(?\s*word count\s*:\s*\d+\s*\)?\s*$")
INCOMPLETE_RANGE_RE = re.compile(r"(?i)\bbetween\s+and\b")
DANGLING_BETWEEN_RE = re.compile(r"(?i)\bbetween(?:\s+[A-Za-z0-9,.-]+){0,5}\s*(?:[.,;:]|$)")
BROKEN_TIMELINE_RE = re.compile(
    r"(?i)\b(?:commenced|started|began)\s+on\s+and\s+(?:continued|lasted)\s+until\b"
)
TRUNCATED_END_RE = re.compile(
    r"(?i)\b(?:and|or|to|of|with|through|including|by|for|from|in|on|at|between)\.?$"
)
VAT_LABEL_RE = re.compile(r"(?i)\bVAT(?:\s+Registration\s+No\.)?\s*:\s*([A-Z0-9]+)\b")
META_SUMMARY_OPENING_RE = re.compile(
    r"(?i)\bthis\s+(?:history|charges|facts|evidence|assessment)\s+section\s+is\s+"
    r"(?:drawn|prepared)\s+strictly\s+from\s+case_memory\b"
)

META_LINE_PREFIXES = (
    "approved:",
    "rubrics:",
    "issues:",
    "revision:",
    "strict compliance notes:",
    "required facts & entities:",
    "required facts and entities:",
    "entities mentioned:",
    "logical order:",
    "fix:",
    "note:",
)
META_LINE_LABELS = frozenset(
    {
        "history",
        "persons",
        "companies",
        "charges",
        "facts",
        "evidence",
        "assessment",
        "required facts & entities",
        "required facts and entities",
        "entities mentioned",
        "logical order",
        "strict compliance notes",
    }
)
SECTION_ENTITY_CHECK = frozenset(
    {
        "persons",
        "companies",
        "history",
        "charges",
        "facts",
        "findings",
        "background",
    }
)

DEFAULT_WORKFLOW_VALIDATORS = {
    "empty_section": True,
    "placeholder_text": True,
    "hidden_reasoning_markup": True,
    "placeholder_markers": True,
    "review_metadata": True,
    "meta_summary_style": True,
    "markdown_formatting": True,
    "incomplete_date_range": True,
    "dangling_between_phrase": True,
    "unresolved_timeline_placeholder": True,
    "partial_vat_identifier": True,
    "repeated_long_sentences": True,
    "repeated_sentence_fragments": True,
    "truncated_sentence": True,
    "required_person_facts": True,
    "required_company_facts": True,
    "known_entity_presence": True,
    "unknown_case_references": True,
    "unknown_dates": True,
    "unknown_amounts": True,
    "unknown_vat_numbers": True,
    "unknown_organisations": True,
    "unknown_titled_people": True,
    "unknown_initials": True,
    "facts_contract": True,
}
