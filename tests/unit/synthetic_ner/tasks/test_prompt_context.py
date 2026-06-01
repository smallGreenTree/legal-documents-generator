from src.synthetic_ner.tasks.document_generation.prompt_context import build_section_context

MEMORY_TEXT = "\n".join(
    [
        "# CASE_MEMORY",
        "",
        "## Document",
        "- Doc ID: test_doc",
        "- Document type: persons_test",
        "- Fraud type: financial_fraud",
        "- Court: Crown Court at Birmingham",
        "- Case number: T202601050",
        "- Cross reference: C/2025/3254",
        "- Filing date: 20 September 2025",
        "",
        "## Defendants",
        (
            "- Ann-Kathrin Dietz | role: procurement officer | "
            "nationality: German | address: Pohlring 36"
        ),
        "",
        "## Collateral",
        "- Paul-Gerhard Gröttner | role: company director | nationality: German",
        "",
        "## Organisations",
        "- PAVAROTTI SERVICES LTD | VAT: IT60686699853 | address: 1 Test Street",
        "",
        "## Counts",
        (
            "- FRAUD BY FALSE REPRESENTATION | section 1 of the Fraud Act 2006 | "
            "causing loss of £559,822."
        ),
        "",
        "## Amounts",
        "- Total alleged loss: £559,822",
        "",
        "## Relationship Graph",
        "- Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD",
        "",
        "## Allowed References",
        "### Case References and Dates",
        "- Case number: T202601050",
        "- Cross reference: C/2025/3254",
        "- Filing date: 20 September 2025",
        "",
        "### Allowed Person Surface Forms",
        (
            "- Ann-Kathrin Dietz | allowed forms: Ann-Kathrin Dietz; A.D; "
            "Dr Dietz | dob: 15 April 1964 | nationality: German"
        ),
        (
            "- Paul-Gerhard Gröttner | allowed forms: Paul-Gerhard Gröttner; "
            "P.G | dob: 4 June 1985 | nationality: German"
        ),
        "",
        "### Allowed Organisations",
        "- PAVAROTTI SERVICES LTD | VAT: IT60686699853",
        "",
        "### Allowed Amounts",
        "- Total alleged loss: £559,822",
        "",
        "### Relationship Facts",
        "- Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD",
        "",
        "## Strict Rules",
        "- Use only listed facts.",
    ]
)


def test_persons_context_contains_only_identity_slice():
    context = build_section_context(MEMORY_TEXT, "persons")

    assert "Ann-Kathrin Dietz" in context
    assert "15 April 1964" in context
    assert "Paul-Gerhard Gröttner" in context
    assert "4 June 1985" in context
    assert "## Person Identity Facts" in context
    assert "PAVAROTTI SERVICES LTD" not in context
    assert "IT60686699853" not in context
    assert "£559,822" not in context
    assert "controlled" not in context
    assert "FRAUD BY FALSE REPRESENTATION" not in context


def test_companies_context_contains_only_organisation_identity_slice():
    context = build_section_context(MEMORY_TEXT, "companies")

    assert "## Organisations" in context
    assert "PAVAROTTI SERVICES LTD" in context
    assert "IT60686699853" in context
    assert "1 Test Street" in context
    assert "Ann-Kathrin Dietz" not in context
    assert "Paul-Gerhard Gröttner" not in context
    assert "£559,822" not in context
    assert "controlled" not in context
    assert "FRAUD BY FALSE REPRESENTATION" not in context


def test_history_context_contains_case_references_and_counts_only():
    context = build_section_context(MEMORY_TEXT, "history")

    assert "### Case References and Dates" in context
    assert "T202601050" in context
    assert "C/2025/3254" in context
    assert "20 September 2025" in context
    assert "## Counts" in context
    assert "FRAUD BY FALSE REPRESENTATION" in context
    assert "Ann-Kathrin Dietz" not in context
    assert "PAVAROTTI SERVICES LTD" not in context
    assert "IT60686699853" not in context
    assert "controlled" not in context
    assert "## Amounts" not in context


def test_charges_context_contains_charge_parties_amounts_and_counts():
    context = build_section_context(MEMORY_TEXT, "charges")

    assert "## Defendants" in context
    assert "Ann-Kathrin Dietz" in context
    assert "## Organisations" in context
    assert "PAVAROTTI SERVICES LTD" in context
    assert "## Counts" in context
    assert "FRAUD BY FALSE REPRESENTATION" in context
    assert "## Amounts" in context
    assert "£559,822" in context
    assert "controlled" not in context
    assert "## Relationship Facts" not in context


def test_facts_context_contains_relationship_counts_and_amounts():
    context = build_section_context(MEMORY_TEXT, "facts")

    assert "## Counts" in context
    assert "FRAUD BY FALSE REPRESENTATION" in context
    assert "## Amounts" in context
    assert "£559,822" in context
    assert "## Relationship Facts" in context
    assert "Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD" in context
    assert "Pohlring 36" not in context
    assert "1 Test Street" not in context
    assert "## Defendants" not in context
    assert "## Organisations" not in context


def test_evidence_context_contains_organisations_and_relationships_without_amounts():
    context = build_section_context(MEMORY_TEXT, "evidence")

    assert "## Organisations" in context
    assert "PAVAROTTI SERVICES LTD" in context
    assert "IT60686699853" in context
    assert "## Relationship Facts" in context
    assert "Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD" in context
    assert "FRAUD BY FALSE REPRESENTATION" not in context
    assert "£559,822" not in context
    assert "## Counts" not in context
    assert "## Amounts" not in context


def test_assessment_context_contains_counts_and_relationships_without_identity_blocks():
    context = build_section_context(MEMORY_TEXT, "assessment")

    assert "## Counts" in context
    assert "FRAUD BY FALSE REPRESENTATION" in context
    assert "## Relationship Facts" in context
    assert "Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD" in context
    assert "£559,822" in context
    assert "Pohlring 36" not in context
    assert "1 Test Street" not in context
    assert "## Defendants" not in context
    assert "## Organisations" not in context
