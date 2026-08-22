from src.synthetic_ner.tasks.document_generation.context.memory import CaseMemoryManager
from src.synthetic_ner.types.document_inputs import DocumentInputs


def test_case_memory_contains_exact_resolved_input_values(tmp_path):
    manager = CaseMemoryManager(tmp_path, summary_chars=120)
    memory_path = manager.create_initial_memory(
        doc_id="doc-1",
        doc_type="indictment",
        fraud_type="procurement_fraud",
        document=_document_inputs(),
        section_order=["facts"],
    )

    memory = memory_path.read_text(encoding="utf-8")
    for expected in (
        "Élodie Müller",
        "ÆTHER LTD",
        "€12,345",
        "CPS/2026/1234",
    ):
        assert expected in memory


def test_case_memory_does_not_infer_missing_counts_or_evidence(tmp_path):
    document = _document_inputs()
    document.counts_list = []
    document.evidence_categories = []
    manager = CaseMemoryManager(tmp_path, summary_chars=120)

    memory_path = manager.create_initial_memory(
        doc_id="doc-1",
        doc_type="indictment",
        fraud_type="procurement_fraud",
        document=document,
        section_order=["facts"],
    )

    memory = memory_path.read_text(encoding="utf-8")
    assert "## Counts\n- none" in memory
    assert "## Evidence Categories\n- none" in memory
    assert "Relationship Graph" not in memory


def test_section_results_do_not_modify_case_memory(tmp_path):
    manager = CaseMemoryManager(tmp_path, summary_chars=120)
    memory_path = manager.create_initial_memory(
        doc_id="doc-1",
        doc_type="indictment",
        fraud_type="procurement_fraud",
        document=_document_inputs(),
        section_order=["facts"],
    )
    memory_before = memory_path.read_bytes()

    manager.append_section_result(
        memory_path,
        section_name="facts",
        section_text="Generated section text.",
        issues=["Example issue."],
    )

    assert memory_path.read_bytes() == memory_before
    run_history = memory_path.with_name("RUN_HISTORY.md").read_text(encoding="utf-8")
    assert "Generated section text." in run_history
    assert "Example issue." in run_history


def _document_inputs() -> DocumentInputs:
    return DocumentInputs(
        defendants=[
            {
                "name": "Élodie Müller",
                "role": "director",
                "nationality": "French",
                "address": "12 Rue Exemple, Paris 75001",
                "dob": "1 January 1980",
                "initials": "É.M.",
                "title_surname": "Ms Müller",
                "short_name": "Élodie",
                "surface_forms_list": ["Élodie Müller"],
            }
        ],
        collateral=[],
        charged_orgs=[
            {
                "name": "ÆTHER LTD",
                "role": "contractor",
                "vat": "FR12345678901",
                "address": "1 Company Street, Paris 75002",
            }
        ],
        associated_orgs=[],
        metadata={
            "court": "Synthetic Court",
            "case_number": "CPS/2026/1234",
            "legal_reference": "1234567/890",
            "cross_ref": "C/2026/1234",
            "filing_date": "2 February 2026",
            "offence_period": None,
        },
        amounts={"total_loss": "€12,345"},
        counts_list=[
            {
                "offence": "PROCUREMENT FRAUD",
                "statute": "Synthetic Act 2026",
                "particulars": "Élodie Müller caused a loss of €12,345.",
            }
        ],
        evidence_categories=[],
    )
