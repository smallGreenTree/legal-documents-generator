"""Cross-cutting static literals shared by generator domains."""

from jinja2 import Environment

PROSECUTION = "Serious Fraud Office"

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
    "RU": "Russian",
    "UA": "Ukrainian",
    "CN": "Chinese",
    "EG": "Egyptian",
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

TITLE_PREFIXES = ["Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Sir", "Lord"]

INCOMPLETE_SECTION_MARKERS = {
    "[missing section]",
    "[section not generated]",
}

# This environment renders prompt/config text, never HTML.
INLINE_TEMPLATE_ENV = Environment(  # nosec B701
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
