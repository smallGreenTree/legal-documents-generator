from src.synthetic_ner.engine import (
    build_groundtruth_rows,
    filter_groundtruth_rows_for_rendered_text,
)


def test_groundtruth_rows_are_structured_in_required_order():
    rows = build_groundtruth_rows(
        "doc-1",
        defendants=[
            {
                "name": "Olivia Price",
                "initials": "O.P.",
                "title_surname": "Dr Price",
                "surface_forms_list": ["Olivia Price", "O.P.", "Dr Price"],
                "dob": "1 January 1980",
                "address": "10 Legal Street, London EC1A 1AA",
                "street": "10 Legal Street",
                "city_postcode": "London EC1A 1AA",
                "is_defendant": True,
            }
        ],
        collateral=[
            {
                "name": "Daniel Dunn",
                "initials": "D.D.",
                "title_surname": "Mr Dunn",
                "surface_forms_list": ["Daniel Dunn", "D.D.", "Mr Dunn"],
                "dob": "2 February 1981",
                "address": "11 Witness Road, London EC2A 2BB",
                "street": "11 Witness Road",
                "city_postcode": "London EC2A 2BB",
                "is_defendant": False,
            }
        ],
        charged_orgs=[
            {
                "name": "PRICE GROUP LTD",
                "address": "1 Company House, London W1A 1AA",
                "street": "1 Company House",
                "city_postcode": "London W1A 1AA",
                "vat": "GB123456789",
            }
        ],
        associated_orgs=[
            {
                "name": "DUNN HOLDINGS LTD",
                "address": "2 Company House, London W2A 2AA",
                "street": "2 Company House",
                "city_postcode": "London W2A 2AA",
                "vat": "GB987654321",
            }
        ],
        metadata={
            "court": "Crown Court at Manchester",
            "case_number": "CPS/2026/1234",
            "cross_ref": "C/2026/5678",
            "filing_date": "3 March 2026",
            "offence_period": ("4 April 2025", "5 May 2025"),
        },
        counts_list=[
            {
                "particulars": (
                    "Olivia Price caused loss of EUR 10,000 between "
                    "4 April 2025 and 5 May 2025."
                )
            }
        ],
        amounts={
            "total_loss": "£250,000",
            "inflated_invoice_value": "£75,000",
            "transfers": [
                {
                    "from": "PRICE GROUP LTD",
                    "to": "DUNN HOLDINGS LTD",
                    "amount": "£175,000",
                }
            ],
        },
    )

    labels = [row[2] for row in rows if row[2] != "NEGATIVE_CONTROL"]
    assert labels == sorted(labels, key=_required_label_order)
    assert _texts_for_label(rows, "PERSON") == ["Olivia Price", "Daniel Dunn"]
    assert _texts_for_label(rows, "ORG") == ["PRICE GROUP LTD", "DUNN HOLDINGS LTD"]
    assert _texts_for_label(rows, "CASE_REFERENCE") == ["CPS/2026/1234", "C/2026/5678"]
    assert "EUR 10,000" in _texts_for_label(rows, "AMOUNT")
    assert "£250,000" in _texts_for_label(rows, "AMOUNT")
    assert "£75,000" in _texts_for_label(rows, "AMOUNT")
    assert "£175,000" in _texts_for_label(rows, "AMOUNT")
    assert _texts_for_label(rows, "INITIAL") == ["O.P.", "D.D."]
    assert _texts_for_label(rows, "TITLE") == ["Dr Price", "Mr Dunn"]
    assert _texts_for_label(rows, "VAT") == ["GB123456789", "GB987654321"]


def _texts_for_label(rows, label):
    return [row[1] for row in rows if row[2] == label]


def test_groundtruth_rows_are_filtered_to_rendered_text_surfaces():
    rows = [
        ("doc-1", "Olivia Price", "PERSON", "yes", "person"),
        ("doc-1", "1 January 1980", "DATE", "yes", "dob"),
        ("doc-1", "PRICE GROUP LTD", "ORG", "yes", "company"),
        ("doc-1", "Serious Fraud Office", "NEGATIVE_CONTROL", "no", "prosecution"),
        ("doc-1", "Crown Court at Manchester", "NEGATIVE_CONTROL", "no", "court"),
    ]
    rendered_text = (
        "OLIVIA PRICE appeared in proceedings involving PRICE GROUP LTD. "
        "The case was brought by Serious Fraud Office."
    )

    filtered = filter_groundtruth_rows_for_rendered_text(rows, rendered_text)

    assert filtered == [
        ("doc-1", "Olivia Price", "PERSON", "yes", "person"),
        ("doc-1", "PRICE GROUP LTD", "ORG", "yes", "company"),
        ("doc-1", "Serious Fraud Office", "NEGATIVE_CONTROL", "no", "prosecution"),
    ]


def _required_label_order(label):
    return {
        "PERSON": 1,
        "ORG": 2,
        "CASE_REFERENCE": 3,
        "DATE": 4,
        "AMOUNT": 5,
        "INITIAL": 6,
        "TITLE": 7,
        "ADDRESS": 8,
        "VAT": 9,
    }[label]
