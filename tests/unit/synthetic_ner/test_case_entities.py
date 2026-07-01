from src.synthetic_ner.case_entities import (
    clean_company_token,
    clean_person_part,
    make_person,
)


def test_clean_person_part_keeps_greek_and_cyrillic_letters():
    cleaned = clean_person_part("Συμεώνη Кузнецова Иванова! 123")

    assert cleaned == "Συμεώνη Кузнецова Иванова"


def test_clean_company_token_keeps_unicode_letters_and_digits():
    cleaned = clean_company_token("Αρματά Иванов 123 & co.!")

    assert cleaned == "ΑΡΜΑΤΆ ИВАНОВ 123 & CO"


def test_make_person_generates_non_empty_greek_and_cyrillic_names():
    locales = {
        "GR": "el_GR",
        "RU": "ru_RU",
        "BG": "bg_BG",
    }

    for nationality in locales:
        person = make_person(
            nationality,
            title="Dr",
            surface_forms=3,
            nickname_variants=1,
            misspelling_variants=1,
            nat_locales=locales,
        )

        assert person["name"].strip()
        assert person["initials"] != "."
        assert person["title_surname"] != "Dr"
