"""Static literals."""
from jinja2 import Environment

SECTION_DESCRIPTIONS = {
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
        "Background: how the matter came before the court, investigation "
        "history, procedural steps."
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

PROSECUTION = "Serious Fraud Office"

COURTS = [
    "Crown Court at Manchester",
    "Crown Court at Birmingham",
    "Crown Court at Leeds",
    "Crown Court at Southwark",
    "Crown Court at Bristol",
    "Crown Court at Newcastle",
    "Crown Court at Liverpool",
    "Crown Court at Sheffield",
]

PERSON_ROLES = [
    "company director",
    "managing director",
    "chief financial officer",
    "chief executive officer",
    "financial controller",
    "procurement officer",
    "operations manager",
    "compliance officer",
    "accountant",
    "consultant",
]

COMPANY_SUFFIXES = [
    "LTD",
    "LIMITED",
    "HOLDINGS LTD",
    "GROUP LTD",
    "INTERNATIONAL LTD",
    "CONSULTING LTD",
    "SERVICES LTD",
    "SOLUTIONS LTD",
]

NATIONALITY_ADJECTIVES = {
    "GB": "British",
    "DE": "German",
    "FR": "French",
    "IT": "Italian",
    "NL": "Dutch",
    "CZ": "Czech",
    "PL": "Polish",
    "ES": "Spanish",
    "PT": "Portuguese",
    "BE": "Belgian",
    "AT": "Austrian",
    "SE": "Swedish",
    "DK": "Danish",
    "FI": "Finnish",
    "HU": "Hungarian",
    "RO": "Romanian",
    "BG": "Bulgarian",
    "GR": "Greek",
    "HR": "Croatian",
    "SK": "Slovak",
    "SI": "Slovenian",
}

EN_LABELS = {
    "file": "File No.",
    "crossref": "Cross-Ref.",
    "date": "Date",
    "born": "born",
    "in": "in",
    "address": "residing at",
    "vat": "VAT Registration No.",
}

EN_SECTIONS = {
    "indictment": {
        "title": "INDICTMENT",
        "section_persons": "SECTION I — PERSONS",
        "section_companies": "SECTION II — COMPANIES",
        "section_history": "SECTION III — PROCEDURAL HISTORY",
        "section_charges": "SECTION IV — CHARGES",
        "section_facts": "SECTION V — STATEMENT OF FACTS",
        "section_evidence": "SECTION VI — EVIDENCE",
        "section_assessment": "SECTION VII — LEGAL ASSESSMENT",
    },
    "court_decision": {
        "title": "JUDGMENT",
        "section_persons": "SECTION I — PARTIES",
        "section_companies": "SECTION II — COMPANIES",
        "section_history": "SECTION III — BACKGROUND",
        "section_charges": "SECTION IV — GROUNDS",
        "section_facts": "SECTION V — FINDINGS OF FACT",
        "section_evidence": "SECTION VI — EVIDENCE REVIEWED",
        "section_assessment": "SECTION VII — CONCLUSIONS AND ORDER",
    },
}

GROUNDTRUTH_HEADER = ["doc_id", "entity_text", "label", "should_propose", "notes"]

TITLE_PREFIXES = ["Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Sir", "Lord"]

INCOMPLETE_SECTION_MARKERS = {
    "[missing section]",
    "[section not generated]",
}

INLINE_TEMPLATE_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
