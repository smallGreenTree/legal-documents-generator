import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.synthetic_ner.config import load_app_config
from src.synthetic_ner.constants import EN_LABELS
from src.synthetic_ner.engine import build_section_labels, build_template_environment
from src.synthetic_ner.tasks.document_generation.orchestrator import run_document_graph
from src.synthetic_ner.tasks.groundtruth import (
    generate_groundtruth_for_document,
    read_groundtruth_tsv,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ID = "en_facts_test_financial_fraud_001"
UNKNOWN_AMOUNT_ISSUE = "Section mentions unknown amount '£99,999'."

GENERATED_FACTS_TEXT = (
    "Ann-Kathrin Dietz is described in the recorded relationship facts as controlling "
    "PAVAROTTI SERVICES LTD. The account remains limited to that relationship and to "
    "the company identified in the case memory. It states that the alleged loss was "
    "£99,999, while otherwise keeping the narrative focused on the allowed company, "
    "the named person, and the recorded financial-fraud context. The section does "
    "not add invoices, witnesses, bank accounts, hearings, or extra organisations, "
    "and it treats the relationship as the only factual foundation for this draft."
)


def test_run_document_graph_sends_prompt_and_applies_validator_config(tmp_path, monkeypatch):
    enabled = _run_graph(
        tmp_path / "enabled",
        monkeypatch,
        unknown_amounts=True,
        max_revisions=2,
    )
    disabled = _run_graph(
        tmp_path / "disabled",
        monkeypatch,
        unknown_amounts=False,
        max_revisions=2,
    )

    writer_call = next(call for call in enabled.calls if call["stage"] == "writer")
    writer_calls = [call for call in enabled.calls if call["stage"] == "writer"]
    critic_calls = [call for call in enabled.calls if call["stage"] == "critic"]
    assert "SECTION_CONTEXT:" in writer_call["user_prompt"]
    assert "SECTION_CONTRACT:" in writer_call["user_prompt"]
    assert "Allowed Amounts" in writer_call["user_prompt"]
    assert "{% if" not in writer_call["user_prompt"]
    assert len(writer_calls) == 3
    assert len(critic_calls) == 3
    assert all(call["client_stage"] == "writer" for call in writer_calls)
    assert all(
        call["client_stage"] == "polisher" for call in enabled.calls if call["stage"] == "polisher"
    )
    assert any("_r1" in call["task_id"] for call in enabled.calls)
    assert any("_r2" in call["task_id"] for call in enabled.calls)
    assert not any("_r3" in call["task_id"] for call in enabled.calls)
    assert len([call for call in disabled.calls if call["stage"] == "writer"]) == 1
    assert len([call for call in disabled.calls if call["stage"] == "critic"]) == 1

    assert UNKNOWN_AMOUNT_ISSUE in enabled.report_text
    assert UNKNOWN_AMOUNT_ISSUE not in disabled.report_text
    assert GENERATED_FACTS_TEXT in enabled.document_text
    assert GENERATED_FACTS_TEXT in disabled.document_text
    assert not any(call["stage"] == "groundtruth" for call in enabled.calls)
    assert {path.name for path in enabled.document_dir.iterdir()} == {
        f"{DOC_ID}.txt",
        "document_inputs.json",
        "generation_report.md",
    }


def test_max_revisions_zero_disables_rewrites(tmp_path, monkeypatch):
    generated = _run_graph(
        tmp_path / "no-revisions",
        monkeypatch,
        unknown_amounts=True,
        max_revisions=0,
    )

    writer_calls = [call for call in generated.calls if call["stage"] == "writer"]
    critic_calls = [call for call in generated.calls if call["stage"] == "critic"]
    assert len(writer_calls) == 1
    assert len(critic_calls) == 1
    assert not any("_r1" in call["task_id"] for call in generated.calls)
    assert UNKNOWN_AMOUNT_ISSUE in generated.report_text


def test_generated_report_and_text_are_sufficient_for_groundtruth(tmp_path, monkeypatch):
    generated = _run_graph(tmp_path / "generated", monkeypatch, unknown_amounts=True)

    result = generate_groundtruth_for_document(
        document_dir=generated.document_dir,
        contract_path=PROJECT_ROOT / "groundtruth_contract.yaml",
    )

    assert result["status"] == "completed"
    annotations = read_groundtruth_tsv(generated.document_dir / "groundtruth.tsv")
    keys = {(row.entity_text, row.label) for row in annotations}
    assert ("Ann-Kathrin Dietz", "PERSON") in keys
    assert ("PAVAROTTI SERVICES LTD", "ORG") in keys
    assert ("£99,999", "AMOUNT") not in keys


def _run_graph(
    tmp_path: Path,
    monkeypatch,
    *,
    unknown_amounts: bool,
    max_revisions: int = 2,
):
    context = _build_context(
        tmp_path,
        unknown_amounts=unknown_amounts,
        max_revisions=max_revisions,
    )
    calls = []
    document = _document_inputs()

    def fake_build_model_client(*, stage, routing, tracer):
        del routing, tracer
        return FakeModelClient(stage, calls)

    monkeypatch.setattr(
        "src.synthetic_ner.tasks.document_generation.orchestrator.build_model_client",
        fake_build_model_client,
    )
    run_document_graph(
        context=context,
        document=document,
        doc_id=DOC_ID,
        workflow_run_id=f"test-{unknown_amounts}",
    )
    report_text = (context.output_dir / DOC_ID / "generation_report.md").read_text(encoding="utf-8")
    document_text = (context.output_dir / DOC_ID / f"{DOC_ID}.txt").read_text(encoding="utf-8")
    return SimpleNamespace(
        calls=calls,
        report_text=report_text,
        document_text=document_text,
        document_dir=context.output_dir / DOC_ID,
    )


def _build_context(
    tmp_path: Path,
    *,
    unknown_amounts: bool,
    max_revisions: int,
) -> RuntimeContext:
    tmp_path.mkdir(parents=True)
    app_config = load_app_config(
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "config_case" / "case_1.yaml",
    )
    validators = {**app_config.workflow.validators, "unknown_amounts": unknown_amounts}
    workflow_cfg = replace(
        app_config.workflow,
        max_revisions=max_revisions,
        validators=validators,
    )
    mlflow_cfg = replace(app_config.mlflow, enabled=False)
    template_path = tmp_path / "facts_test.j2"
    template_path.write_text(
        "FACTS TEST\n\n{{ llm_sections[0] }}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    memory_dir = tmp_path / "memory"
    output_dir.mkdir()
    memory_dir.mkdir()

    return RuntimeContext(
        project_root=PROJECT_ROOT,
        app_config=app_config,
        paths=app_config.paths,
        generation_cfg=app_config.generation,
        profile=replace(app_config.profile, doc_type="facts_test"),
        case_cfg=app_config.case,
        mlflow_cfg=mlflow_cfg,
        model_routing_cfg=app_config.model_routing,
        workflow_cfg=workflow_cfg,
        nat_locales=app_config.nationality_locales,
        vat_prefixes=app_config.vat_prefixes,
        doc_type="facts_test",
        fraud_type="financial_fraud",
        output_dir=output_dir,
        memory_dir=memory_dir,
        template_path=template_path,
        template_env=build_template_environment(template_path),
        template_name=template_path.name,
        sections=build_section_labels("facts_test", ["facts"]),
        labels=EN_LABELS,
        section_word_targets={"facts": 90},
        documents=1,
        prose_overrides={},
    )


def _document_inputs() -> DocumentInputs:
    defendant = {
        "name": "Ann-Kathrin Dietz",
        "role": "procurement officer",
        "nationality": "German",
        "address": "Pohlring 36",
        "dob": "15 April 1964",
        "initials": "A.D.",
        "title_surname": "Dr Dietz",
        "short_name": "Ann-Kathrin",
        "surface_forms_list": ["Ann-Kathrin Dietz"],
        "is_defendant": True,
    }
    org = {
        "name": "PAVAROTTI SERVICES LTD",
        "vat": "IT60686699853",
        "address": "1 Test Street",
    }
    return DocumentInputs(
        defendants=[defendant],
        collateral=[],
        charged_orgs=[org],
        associated_orgs=[],
        metadata={
            "court": "Test Synthetic Court",
            "case_number": "T202601050",
            "cross_ref": "C/2025/3254",
            "filing_date": "20 September 2025",
            "offence_period": ("1 January 2025", "31 January 2025"),
        },
        amounts={"total_loss": "£559,822"},
        counts_list=[
            {
                "offence": "FRAUD BY FALSE REPRESENTATION",
                "statute": "section 1 of the Fraud Act 2006",
                "particulars": (
                    "Ann-Kathrin Dietz caused loss of £559,822 through PAVAROTTI SERVICES LTD."
                ),
            }
        ],
    )


class FakeModelClient:
    def __init__(self, stage: str, calls: list[dict]):
        self.stage = stage
        self.calls = calls

    def invoke(self, **kwargs):
        self.calls.append({**kwargs, "client_stage": self.stage})
        if kwargs["stage"] == "writer":
            text = json.dumps(
                {
                    "content": GENERATED_FACTS_TEXT,
                    "facts_used": [
                        "Ann-Kathrin Dietz controlled PAVAROTTI SERVICES LTD",
                    ],
                    "tone": "formal",
                    "legal_risks": [],
                }
            )
        elif kwargs["stage"] == "polisher":
            text = GENERATED_FACTS_TEXT
        else:
            text = json.dumps(
                {
                    "blocking": False,
                    "edits": [],
                    "risk_level": "low",
                    "rubrics": {
                        "grounding": 5,
                        "completeness": 5,
                        "chronology": 5,
                    },
                }
            )
        return SimpleNamespace(text=text, metadata={"stage": self.stage})
