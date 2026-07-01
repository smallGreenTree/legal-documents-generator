import json
from pathlib import Path
from types import SimpleNamespace

from src.synthetic_ner.engine import build_groundtruth_rows
from src.synthetic_ner.prefect_flows.utils import (
    EntityReviewInput,
    QualityDocumentSelectionInput,
    ScenarioReviewInput,
    _apply_case_setup_to_config,
    _build_scenario,
    _case_setup_from_review_response,
    _case_setup_initial_data,
    _document_from_review_json,
    _document_to_payload,
    _entity_review_description,
    _entity_review_scenario_summary,
    _initial_organisation_specs_for_setup,
    _initial_person_specs_for_setup,
    _organisation_setup_review_input_model,
    _organisation_specs_from_review_response,
    _parse_person_specs_yaml,
    _person_setup_review_input_model,
    _person_specs_from_review_response,
    _required_prefilled_input_model,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_prefect_stage_one_limits_scenarios_to_configured_options():
    scenario = _build_scenario(
        project_root=PROJECT_ROOT,
        case_config="config_case/case_1.yaml",
        template="templates/en_indictment.j2",
        documents=None,
        doc_type=None,
        fraud_type="financial_fraud",
        from_schema=None,
    )

    assert scenario["fraud_type"] == "procurement_fraud"
    assert scenario["scenario_options"] == {
        "procurement_fraud": "Procurement fraud and corruption",
        "eu_subsidy_fraud": "Non-procurement expenditure fraud (subsidy fraud)",
    }


def test_case_setup_builds_generated_case_config():
    source = {
        "profile": {
            "doc_type": "indictment",
            "fraud_type": "procurement_fraud",
            "documents": 1,
        },
        "case": {"cast": {"address_surface_forms": 3}},
    }
    scenario = {
        "doc_type": "indictment",
        "fraud_type": "eu_subsidy_fraud",
        "documents": 3,
    }
    person_specs = _parse_person_specs_yaml(
        """
        - group: defendant
          nationality: DE
          title: Dr
          surface_forms: 2
        - group: collateral
          nationality: FR
          title: Ms
          surface_forms: 1
        """
    )
    case_setup = {
        "person_specs": person_specs,
        "charged_orgs": 2,
        "associated_orgs": 1,
        "organisation_specs": [
            {"group": "charged", "country": "DE"},
            {"group": "charged", "country": "IT"},
            {"group": "associated", "country": "FR"},
        ],
    }

    generated = _apply_case_setup_to_config(source, scenario, case_setup)

    assert generated["profile"]["fraud_type"] == "eu_subsidy_fraud"
    assert generated["profile"]["documents"] == 3
    assert generated["case"]["cast"]["charged_orgs"] == 2
    assert generated["case"]["cast"]["associated_orgs"] == 1
    assert generated["case"]["cast"]["organisation_specs"] == [
        {"group": "charged", "country": "DE"},
        {"group": "charged", "country": "IT"},
        {"group": "associated", "country": "FR"},
    ]
    assert "address_surface_forms" not in generated["case"]["cast"]
    assert generated["case"]["cast"]["defendants"] == [
        {"nationality": "DE", "title": "Dr", "surface_forms": 2}
    ]
    assert generated["case"]["cast"]["collateral"] == [
        {"nationality": "FR", "title": "Ms", "surface_forms": 1}
    ]


def test_prefect_stage_one_collects_counts_before_person_rows():
    scenario = _build_scenario(
        project_root=PROJECT_ROOT,
        case_config="config_case/case_1.yaml",
        template="templates/en_indictment.j2",
        documents=None,
        doc_type=None,
        fraud_type=None,
        from_schema=None,
    )

    initial = _case_setup_initial_data(scenario)

    assert initial["scenario"] == "Procurement fraud and corruption"
    assert initial["person_entities"] == 5
    assert initial["charged_orgs"] == 2
    assert initial["associated_orgs"] == 1
    assert "person_1_group" not in initial
    assert "address_surface_forms" not in initial


def test_prefect_stage_one_schema_marks_setup_fields_required():
    scenario = _build_scenario(
        project_root=PROJECT_ROOT,
        case_config="config_case/case_1.yaml",
        template="templates/en_indictment.j2",
        documents=None,
        doc_type=None,
        fraud_type=None,
        from_schema=None,
    )
    input_model = _required_prefilled_input_model(
        ScenarioReviewInput,
        description="test",
        **_case_setup_initial_data(scenario),
        documents=scenario["documents"],
        doc_type=scenario["doc_type"] or "",
    )

    schema = input_model.model_json_schema()

    assert set(schema["required"]) == set(schema["properties"])
    assert all(field.is_required() for field in input_model.model_fields.values())
    assert "scenario_summary" not in schema["properties"]
    assert "person_specs_yaml" not in schema["properties"]
    assert "action" not in schema["properties"]
    assert "case_config" not in schema["properties"]
    assert "template" not in schema["properties"]
    assert "from_schema" not in schema["properties"]
    assert "address_surface_forms" not in schema["properties"]
    assert "generated_case_config" not in schema["properties"]
    assert schema["properties"]["doc_type"]["enum"] == ["indictment", "court_decision"]
    assert "person_1_group" not in schema["properties"]


def test_prefect_stage_one_b_person_setup_has_exact_selected_rows():
    scenario = _build_scenario(
        project_root=PROJECT_ROOT,
        case_config="config_case/case_1.yaml",
        template="templates/en_indictment.j2",
        documents=None,
        doc_type=None,
        fraud_type=None,
        from_schema=None,
    )
    specs = _initial_person_specs_for_setup(scenario, 3)
    input_model = _person_setup_review_input_model(specs)

    schema = input_model.model_json_schema()

    assert "person_1_group" in schema["properties"]
    assert "person_3_surface_forms" in schema["properties"]
    assert "person_4_group" not in schema["properties"]
    assert schema["properties"]["person_1_group"]["enum"] == [
        "defendant",
        "collateral",
    ]
    assert schema["properties"]["person_1_nationality"]["enum"][:4] == [
        "GB",
        "DE",
        "FR",
        "IT",
    ]
    assert set(schema["required"]) == set(schema["properties"])
    assert all(field.is_required() for field in input_model.model_fields.values())


def test_prefect_stage_one_c_organisation_setup_has_exact_selected_rows():
    scenario = _build_scenario(
        project_root=PROJECT_ROOT,
        case_config="config_case/case_1.yaml",
        template="templates/en_indictment.j2",
        documents=None,
        doc_type=None,
        fraud_type=None,
        from_schema=None,
    )
    specs = _initial_organisation_specs_for_setup(
        scenario,
        charged_count=2,
        associated_count=1,
    )
    input_model = _organisation_setup_review_input_model(specs)

    schema = input_model.model_json_schema()

    assert "organisation_1_group" in schema["properties"]
    assert "organisation_3_country" in schema["properties"]
    assert "organisation_4_group" not in schema["properties"]
    assert schema["properties"]["organisation_1_group"]["enum"] == [
        "charged",
        "associated",
    ]
    assert schema["properties"]["organisation_1_country"]["enum"][:4] == [
        "GB",
        "DE",
        "FR",
        "IT",
    ]
    assert set(schema["required"]) == set(schema["properties"])
    assert all(field.is_required() for field in input_model.model_fields.values())


def test_prefect_review_pause_forms_do_not_expose_action():
    entity_model = _required_prefilled_input_model(
        EntityReviewInput,
        description="test",
        document_json="{}",
        refresh_counts=True,
    )
    quality_model = _required_prefilled_input_model(
        QualityDocumentSelectionInput,
        description="test",
        doc_id="doc-1",
        candidate_documents="doc-1",
    )

    assert "action" not in entity_model.model_json_schema()["properties"]
    assert "action" not in quality_model.model_json_schema()["properties"]


def test_entity_review_payload_round_trips_amounts():
    document = DocumentInputs(
        defendants=[
            {
                "name": "Alex Meyer",
                "surface_forms_list": ["Alex Meyer"],
                "is_defendant": True,
            }
        ],
        collateral=[],
        charged_orgs=[{"name": "ACME LTD"}],
        associated_orgs=[],
        metadata={
            "court": "Crown Court at Cardiff",
            "case_number": "T202612345",
            "cross_ref": "C/2026/1234",
            "filing_date": "1 June 2026",
            "offence_period": ("1 January 2026", "31 January 2026"),
        },
        amounts={"total_loss": "£125,000"},
        counts_list=[
            {
                "offence": "Fraud",
                "statute": "Fraud Act 2006",
                "particulars": "Loss of £125,000.",
            }
        ],
        evidence_categories=["invoices"],
    )

    context = SimpleNamespace(
        fraud_type="procurement_fraud",
        doc_type="indictment",
    )

    payload = _document_to_payload(document, context=context)
    reviewed = _document_from_review_json(json.dumps(payload))

    assert payload["scenario"] == {
        "id": "procurement_fraud",
        "label": "Procurement fraud and corruption",
        "doc_type": "indictment",
    }
    assert payload["amounts"] == {"total_loss": "£125,000"}
    assert reviewed.amounts == document.amounts
    assert reviewed.evidence_categories == document.evidence_categories


def test_entity_review_screen_includes_scenario_summary():
    context = SimpleNamespace(
        fraud_type="procurement_fraud",
        doc_type="indictment",
    )

    summary = _entity_review_scenario_summary(context)

    assert "Procurement fraud and corruption" in summary
    assert "procurement_fraud" in summary
    assert "Document type: indictment" in summary


def test_entity_review_description_includes_surface_forms_and_review_prompt():
    context = SimpleNamespace(
        fraud_type="procurement_fraud",
        doc_type="indictment",
    )
    payload = {
        "defendants": [
            {
                "name": "Roswitha Fechner",
                "surface_forms_list": [
                    "Roswitha Fechner",
                    "R.F.",
                    "Mr Fechner",
                ],
            }
        ],
        "collateral": [],
        "charged_orgs": [{"name": "HÖRLE HEIDRICH GROUP LTD"}],
        "associated_orgs": [],
    }

    description = _entity_review_description(context, payload)

    assert "Procurement fraud and corruption" in description
    assert "Generated names:\nDefendants: Roswitha Fechner" in description
    assert "Person surface forms:\nDefendants: Roswitha Fechner:" in description
    assert "Roswitha Fechner, R.F., Mr Fechner" in description
    assert "Review the specifics below." in description


def test_case_setup_reads_prefect_person_rows():
    setup_response = ScenarioReviewInput(
        scenario="Non-procurement expenditure fraud (subsidy fraud)",
        doc_type="indictment",
        person_entities=2,
        charged_orgs=1,
        associated_orgs=0,
    )
    person_model = _person_setup_review_input_model(
        [
            {
                "group": "defendant",
                "nationality": "DE",
                "title": "Dr",
                "surface_forms": 2,
            },
            {
                "group": "collateral",
                "nationality": "FR",
                "title": "",
                "surface_forms": 1,
            },
        ]
    )
    person_response = person_model(
        person_1_group="defendant",
        person_1_nationality="DE",
        person_1_title="Dr",
        person_1_surface_forms=2,
        person_2_group="collateral",
        person_2_nationality="FR",
        person_2_title="No title",
        person_2_surface_forms=1,
    )
    scenario = {"fraud_type": "eu_subsidy_fraud"}
    person_specs = _person_specs_from_review_response(person_response, 2)
    organisation_model = _organisation_setup_review_input_model(
        [{"group": "charged", "country": "DE"}]
    )
    organisation_response = organisation_model(
        organisation_1_group="charged",
        organisation_1_country="DE",
    )
    organisation_specs = _organisation_specs_from_review_response(
        organisation_response,
        1,
    )

    case_setup = _case_setup_from_review_response(
        setup_response,
        scenario,
        person_specs,
        organisation_specs,
    )

    assert case_setup["person_specs"] == [
        {
            "group": "defendant",
            "nationality": "DE",
            "title": "Dr",
            "surface_forms": 2,
        },
        {
            "group": "collateral",
            "nationality": "FR",
            "title": "",
            "surface_forms": 1,
        },
    ]
    assert case_setup["charged_orgs"] == 1
    assert case_setup["associated_orgs"] == 0
    assert case_setup["organisation_specs"] == [{"group": "charged", "country": "DE"}]


def test_address_surface_forms_limit_groundtruth_address_rows():
    rows = build_groundtruth_rows(
        "doc-1",
        defendants=[
            {
                "name": "Olivia Price",
                "initials": "O.P.",
                "title_surname": "Dr Price",
                "surface_forms_list": ["Olivia Price"],
                "dob": "1 January 1980",
                "address": "10 Legal Street, London EC1A 1AA",
                "street": "10 Legal Street",
                "city_postcode": "London EC1A 1AA",
                "is_defendant": True,
            }
        ],
        collateral=[],
        charged_orgs=[],
        associated_orgs=[],
        metadata={
            "court": "Crown Court at Manchester",
            "case_number": "CPS/2026/1234",
            "cross_ref": "C/2026/5678",
            "filing_date": "3 March 2026",
            "offence_period": None,
        },
        counts_list=[],
        address_surface_forms=1,
    )

    assert [row[1] for row in rows if row[2] == "ADDRESS"] == [
        "10 Legal Street, London EC1A 1AA"
    ]
