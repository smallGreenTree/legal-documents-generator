# Synthetic Legal NER Generator

This repo generates synthetic legal documents together with NER ground truth.

The generator works in three layers:

- `case schema` = the case facts
- `prose` = the narrative text written by the model
- `template` = the final document shell

`config.yaml` is now the single authored source of truth for generation inputs:

- `profile` holds document-level run settings
- `case` holds case-specific inputs, either as `auto` generators or explicit records
- templates remain render-only

## How It Works

1. The generator builds or loads the cast and companies.
2. It builds a case schema, or loads one with `--from-schema`.
3. That schema is converted into plain-English constraints for the model.
4. The model writes section prose, one section at a time.
5. A Jinja template renders the final document and inserts the generated prose into fixed slots.

The CLI entrypoint stays in `generator.py`, while the implementation now lives under `src/synthetic_ner/`.

Current package split:

- `src/synthetic_ner/constants.py` for static literals and document defaults
- `src/synthetic_ner/utils.py` for shared helpers and file utilities
- `src/synthetic_ner/case.py` for cast, metadata, and count resolution
- `src/synthetic_ner/schema.py` for case-schema and document-id handling
- `src/synthetic_ner/engine.py` for the core generation flow
- `src/synthetic_ner/cli.py` for argument parsing

## What The Case Schema Does

The case schema is a lightweight knowledge graph for the story of the case.

It stores relationships such as:

- who controlled which company
- who instructed whom
- which company received funds from which other company

Its job is consistency of facts, not formatting. The schema is saved as a JSON file under `schemas/`.

When you run with `--from-schema`, the generator loads an existing schema and uses it as the factual backbone for the next document.

## What The Prose Does

The prose is the model-written narrative.

This is the part controlled by `profile.section_words` in `config.yaml`. The model writes each section separately, using:

- the requested section type
- the case schema context
- the target word count
- entity mention instructions

So the prose is where the document becomes detailed and readable.

## What The Template Does

The template is the deterministic wrapper around the generated prose.

It handles the fixed legal structure, including:

- court header
- file number and dates
- defendant list
- charged company list
- count blocks
- section headings
- insertion points for generated prose

The templates do not create the story. They only format and assemble the document around the generated content.

## Why There Are Two Templates

There are two templates because there are two document types:

- `templates/en_indictment.j2`
- `templates/en_court_decision.j2`

They have different structures.

The indictment template includes:

- prosecution header
- counts
- sections such as `history`, `charges`, `facts`, `evidence`, `assessment`

The court decision template includes:

- judgment caption
- sections such as `background`, `findings`, `conclusions`, `sentence`

The generator selects the template from `profile.doc_type`.

## Short Summary

- schema = what happened
- prose = how it is narrated
- template = how it is formatted

## Configuration Notes

The active run profile lives in `config.yaml`.

Useful fields:

- `doc_type`: chooses the document form, such as `indictment` or `court_decision`
- `fraud_type`: chooses the offense family and statute text
- `documents`: number of documents to generate in one run
- `section_words`: exact target size for each prose section

`section_words` is the preferred size control. If it is omitted, the generator can still fall back to `pages`.

Case-specific inputs now live under `case`:

- `case.metadata`: court, filing references, dates, offence period
- `case.cast`: auto-generation specs for defendants, collateral, and org counts
- `case.defendants` / `case.collateral` / `case.charged_orgs` / `case.associated_orgs`: explicit records that override auto-generation
- `case.schema`: embedded schema, same role as `case_schema.json`
- `case.prose`: explicit prose per section; any section left as `auto` is still generated
- `case.counts`: explicit indictment counts

Keeping these values as `auto` preserves the current generator behavior. Replacing them with explicit values lets one config file hold the entire case definition.

## Commands

Generate with the current profile:

```bash
poetry run python generator.py
```

Generate from an existing schema:

```bash
poetry run python generator.py --from-schema schemas/en_indictment_financial_fraud_002.json
```

Generate multiple documents:

```bash
poetry run python generator.py --documents 3
```

Visualize entity coverage:

```bash
poetry run python visualize.py
```

Check code quality:

```bash
poetry run ruff check .
```
