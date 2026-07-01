import json
from pathlib import Path
from types import SimpleNamespace

from src.synthetic_ner.constants import NATIONALITY_ADJECTIVES
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
    _scenario_review_field_types,
    _used_doc_counters,
)
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_prefect_stage_one_reads_scenarios_from_case_config():
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
    }
    assert scenario["specific_scenario_options"] == {
        "procurement_fraud": "Czechish transport ministry sound surveillance procurement",
    }


def test_case_setup_builds_generated_case_config():
    source = {
        "profile": {
            "doc_type": "indictment",
            "documents": 1,
        },
        "scenario": {
            "id": "procurement_fraud",
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
        "court": "District Court of Example",
        "person_specs": person_specs,
        "charged_orgs": 2,
        "associated_orgs": 1,
        "organisation_specs": [
            {"group": "charged", "role": "Contractor", "country": "DE"},
            {"group": "charged", "role": "Awarded contractor", "country": "IT"},
            {"group": "associated", "role": "Associated organisation", "country": "FR"},
        ],
    }

    generated = _apply_case_setup_to_config(source, scenario, case_setup)

    assert "fraud_type" not in generated["profile"]
    assert generated["scenario"]["id"] == "eu_subsidy_fraud"
    assert generated["profile"]["documents"] == 3
    assert generated["case"]["metadata"]["court"] == "District Court of Example"
    assert generated["case"]["cast"]["charged_orgs"] == 2
    assert generated["case"]["cast"]["associated_orgs"] == 1
    assert generated["case"]["cast"]["organisation_specs"] == [
        {"group": "charged", "role": "Contractor", "country": "DE"},
        {"group": "charged", "role": "Awarded contractor", "country": "IT"},
        {"group": "associated", "role": "Associated organisation", "country": "FR"},
    ]
    assert "address_surface_forms" not in generated["case"]["cast"]
    assert generated["case"]["cast"]["defendants"] == [
        {"nationality": "DE", "title": "Dr", "surface_forms": 2}
    ]
    assert generated["case"]["cast"]["collateral"] == [
        {"nationality": "FR", "title": "Ms", "surface_forms": 1}
    ]


def test_generated_case_yaml_reserves_document_id(tmp_path):
    generated_dir = tmp_path / "config_case" / "generated"
    generated_dir.mkdir(parents=True)
    generated_dir.joinpath("en_indictment_procurement_fraud_001.yaml").write_text(
        "profile: {}\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        project_root=tmp_path,
        output_dir=tmp_path / "output",
        schema_dir=tmp_path / "schemas",
        memory_dir=tmp_path / "memory",
        doc_type="indictment",
        fraud_type="procurement_fraud",
    )

    assert _used_doc_counters(context) == {1}


def test_prefect_stage_one_selects_family_and_specific_scenario_before_rows():
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

    assert initial["scenario_family"] == "Procurement fraud and corruption"
    assert (
        initial["select_scenario"]
        == "Czechish transport ministry sound surveillance procurement"
    )
    assert "Faker placeholders appear in {braces}" in initial[
        "scenario_template_preview"
    ]
    assert "{first_defendant}" in initial["scenario_template_preview"]
    assert "{first_company}" in initial["scenario_template_preview"]
    assert initial["court"] == "Czechish District Court"
    assert "person_entities" not in initial
    assert "charged_orgs" not in initial
    assert "associated_orgs" not in initial
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
        field_types=_scenario_review_field_types(scenario),
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
    assert "person_entities" not in schema["properties"]
    assert "charged_orgs" not in schema["properties"]
    assert "associated_orgs" not in schema["properties"]
    assert "const" not in schema["properties"]["scenario_family"]
    assert "const" not in schema["properties"]["select_scenario"]
    assert schema["$defs"]["ScenarioFamilyChoice"]["enum"] == [
        "Procurement fraud and corruption",
    ]
    assert schema["$defs"]["SpecificScenarioChoice"]["enum"] == [
        "Czechish transport ministry sound surveillance procurement",
    ]
    assert "scenario_template_preview" in schema["properties"]
    assert "court" in schema["properties"]
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
    assert "person_1_role" in schema["properties"]
    assert "person_1_custom_role" in schema["properties"]
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
    assert schema["properties"]["person_1_nationality"]["enum"][-4:] == [
        "RU",
        "UA",
        "CN",
        "EG",
    ]
    assert schema["properties"]["person_1_role"]["enum"] == [
        "Executive Director",
        "Public official",
        "Procurement officer",
        "Tender committee chair",
        "Tender committee member",
        "Company director",
        "Managing director",
        "Beneficial owner",
        "Accountant",
        "Custom role",
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
    assert "organisation_1_role" in schema["properties"]
    assert "organisation_1_custom_role" in schema["properties"]
    assert "organisation_3_country" in schema["properties"]
    assert "organisation_4_group" not in schema["properties"]
    assert schema["properties"]["organisation_1_group"]["enum"] == [
        "charged",
        "associated",
    ]
    assert schema["properties"]["organisation_1_role"]["enum"] == [
        "Sole tenderer / contractor",
        "Contractor",
        "Awarded contractor",
        "Contracting authority",
        "Managing authority",
        "Beneficiary company",
        "Intermediary company",
        "Associated organisation",
        "Custom role",
    ]
    assert schema["properties"]["organisation_1_country"]["enum"][:4] == [
        "GB",
        "DE",
        "FR",
        "IT",
    ]
    assert schema["properties"]["organisation_1_country"]["enum"][-4:] == [
        "RU",
        "UA",
        "CN",
        "EG",
    ]
    assert set(schema["required"]) == set(schema["properties"])
    assert all(field.is_required() for field in input_model.model_fields.values())


def test_new_nationality_choices_have_faker_locales_and_labels():
    case_config = load_config(PROJECT_ROOT / "config_case" / "case_1.yaml")

    assert {key: NATIONALITY_ADJECTIVES[key] for key in ("RU", "UA", "CN", "EG")} == {
        "RU": "Russian",
        "UA": "Ukrainian",
        "CN": "Chinese",
        "EG": "Egyptian",
    }
    assert case_config["nationality_locales"]["RU"] == "ru_RU"
    assert case_config["nationality_locales"]["UA"] == "uk_UA"
    assert case_config["nationality_locales"]["CN"] == "zh_CN"
    assert case_config["nationality_locales"]["EG"] == "ar_EG"
    assert case_config["vat_prefixes"]["RU"] == "RU"
    assert case_config["vat_prefixes"]["UA"] == "UA"
    assert case_config["vat_prefixes"]["CN"] == "CN"
    assert case_config["vat_prefixes"]["EG"] == "EG"


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
            "court": "Test Synthetic Court",
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
        "label": "procurement fraud",
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

    assert "procurement fraud" in summary
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

    assert "procurement fraud" in description
    assert "Generated names:\nDefendants: Roswitha Fechner" in description
    assert "Person surface forms:\nDefendants: Roswitha Fechner:" in description
    assert "Roswitha Fechner, R.F., Mr Fechner" in description
    assert "Review the specifics below." in description


def test_case_setup_reads_prefect_person_rows():
    setup_response = ScenarioReviewInput(
        scenario_family="Procurement fraud and corruption",
        select_scenario="Czechish transport ministry sound surveillance procurement",
        court="District Court of Example",
        doc_type="indictment",
    )
    person_model = _person_setup_review_input_model(
        [
            {
                "group": "defendant",
                "role": "Executive Director",
                "nationality": "DE",
                "title": "Dr",
                "surface_forms": 2,
            },
            {
                "group": "collateral",
                "role": "Custom Ministry Role",
                "nationality": "FR",
                "title": "",
                "surface_forms": 1,
            },
        ]
    )
    person_response = person_model(
        person_1_group="defendant",
        person_1_role="Executive Director",
        person_1_custom_role="",
        person_1_nationality="DE",
        person_1_title="Dr",
        person_1_surface_forms=2,
        person_2_group="collateral",
        person_2_role="Custom role",
        person_2_custom_role="Custom Ministry Role",
        person_2_nationality="FR",
        person_2_title="No title",
        person_2_surface_forms=1,
    )
    scenario = {"fraud_type": "procurement_fraud"}
    person_specs = _person_specs_from_review_response(person_response, 2)
    organisation_model = _organisation_setup_review_input_model(
        [{"group": "charged", "role": "Contractor", "country": "DE"}]
    )
    organisation_response = organisation_model(
        organisation_1_group="charged",
        organisation_1_role="Custom role",
        organisation_1_custom_role="Sole tenderer and awarded contractor",
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
            "role": "Executive Director",
            "nationality": "DE",
            "title": "Dr",
            "surface_forms": 2,
        },
        {
            "group": "collateral",
            "role": "Custom Ministry Role",
            "nationality": "FR",
            "title": "",
            "surface_forms": 1,
        },
    ]
    assert case_setup["court"] == "District Court of Example"
    assert case_setup["charged_orgs"] == 1
    assert case_setup["associated_orgs"] == 0
    assert case_setup["organisation_specs"] == [
        {
            "group": "charged",
            "role": "Sole tenderer and awarded contractor",
            "country": "DE",
        }
    ]


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
            "court": "Test Synthetic Court",
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
