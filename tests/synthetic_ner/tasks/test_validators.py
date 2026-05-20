from src.synthetic_ner.tasks.validators import (
    build_deterministic_fallback_section,
    clean_generated_section_text,
    repair_section_text,
    validate_section_text,
)

MEMORY_TEXT = """# CASE_MEMORY

## Document
- Doc ID: test_doc
- Document type: indictment
- Fraud type: financial_fraud
- Court: Test Crown Court
- Case number: CPS/2025/0001
- Cross reference: C/2025/10
- Filing date: 1 February 2025

## Defendants
- Alice Smith | role: director | nationality: British | address: 1 Test Street
- Bob Jones | role: accountant | nationality: British | address: 2 Test Street

## Organisations
- ACME TRADING LTD | VAT: GB123456789 | address: 3 Test Street
- BETA SERVICES LTD | VAT: GB987654321 | address: 4 Test Street

## Counts
- Fraud by false representation | Fraud Act 2006 | between 1 January 2025 and 31 January 2025.

## Relationship Graph
- Alice Smith instructed Bob Jones to route funds through ACME TRADING LTD.
- ACME TRADING LTD issued records to BETA SERVICES LTD.

## Allowed References
### Case References and Dates
- CPS/2025/0001
- C/2025/10
- 1 February 2025
- 1 January 2025
- 31 January 2025

### Allowed Person Surface Forms
- Alice Smith | allowed forms: Alice Smith; Ms Smith | dob: 1 March 1980
- Bob Jones | allowed forms: Bob Jones; Mr Jones | dob: 2 April 1981

### Allowed Organisations
- ACME TRADING LTD | VAT: GB123456789
- BETA SERVICES LTD | VAT: GB987654321

## Strict Rules
- Use only listed facts.
"""


def test_clean_generated_section_text_removes_metadata_and_thinking():
    raw_text = """
APPROVED: yes
<think>private reasoning</think>
### Heading
Alice Smith acted for ACME TRADING LTD
"""

    assert clean_generated_section_text(raw_text) == (
        "### Heading Alice Smith acted for ACME TRADING LTD."
    )


def test_validate_section_text_flags_unknown_values_and_markdown():
    issues = validate_section_text(
        section_name="facts",
        section_text=(
            "- Alice Smith mentioned UNKNOWN LTD and CPS/2026/9999 "
            "during the alleged scheme."
        ),
        memory_text=MEMORY_TEXT,
        word_target=300,
    )

    assert "Section contains markdown/list formatting; output must be plain prose." in issues
    assert "Section mentions unknown case reference 'CPS/2026/9999'." in issues
    assert "Section mentions unknown organisation 'UNKNOWN LTD'." in issues


def test_repair_section_text_removes_hidden_reasoning_and_unknown_value():
    repaired = repair_section_text(
        section_text="<think>draft</think>Alice Smith dealt with UNKNOWN LTD",
        issues=[
            "Section still contains hidden reasoning markup.",
            "Section mentions unknown organisation 'UNKNOWN LTD'.",
        ],
        memory_text=MEMORY_TEXT,
    )

    assert "<think>" not in repaired
    assert "UNKNOWN LTD" not in repaired
    assert repaired == "Alice Smith dealt with."


def test_build_deterministic_fallback_section_uses_memory_facts():
    fallback = build_deterministic_fallback_section(
        section_name="facts",
        memory_text=MEMORY_TEXT,
        word_target=300,
    )

    assert "Alice Smith" in fallback
    assert "ACME TRADING LTD" in fallback
    assert "1 January 2025 and 31 January 2025" in fallback
