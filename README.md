# Synthetic Legal NER Generator

This repo generates synthetic legal documents together with NER ground truth.

The generator works in three layers:

- `case schema` = the case facts
- `prose` = the narrative text written by the model
- `template` = the final document shell

`config.yaml` remains the run control file, and workflow prompts now live in
`configs/workflow_prompts.yaml`:

- `profile` holds document-level run settings
- `case` holds case-specific inputs, either as `auto` generators or explicit records
- `workflow.prompts_config_path` points to prompt templates (Jinja text)
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
- `src/synthetic_ner/tasks/` for the LangGraph planner/writer/critic workflow
- `src/synthetic_ner/models/ollama_client.py` for traced Ollama access

At startup, the CLI auto-loads environment variables from `.env` and `.env.langfuse`
if present (without overriding already-exported shell variables). This helps IDE
debugger runs pick up Langfuse credentials.

## LangGraph Workflow

The project now supports a LangGraph-based workflow around the existing
generator primitives.

That workflow adds:

- persistent case memory under `memory/case_{doc_id}/CASE_MEMORY.md`
- planner, writer, critic, and validator task modules
- a generation report saved alongside the final document
- node-level Langfuse spans for every LangGraph node, including section, revision, next-node, and latency metadata
- Langfuse Prompt Management integration (prompts fetched by name with config fallback)

The active workflow mode is controlled by `workflow.mode` in `config.yaml`,
and can be overridden on the CLI with `--workflow-mode classic|langgraph`.

When Langfuse is enabled, the workflow tries to fetch these managed text prompts:

- `synthetic_ner.document_planner_system`
- `synthetic_ner.document_planner_user`
- `synthetic_ner.section_planner_system`
- `synthetic_ner.section_planner_user`
- `synthetic_ner.writer_system`
- `synthetic_ner.writer_user`
- `synthetic_ner.critic_system`
- `synthetic_ner.critic_user`

If a prompt does not exist, the app falls back to `configs/workflow_prompts.yaml` and can auto-seed
the missing prompt in Langfuse.

Optional env vars:

- `LANGFUSE_PROMPT_LABEL` (default label for fetch/seed, e.g. `production`)
- `LANGFUSE_PROMPT_AUTOSEED` (`true` by default; set `false` to disable auto-seeding)

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

Sync prompt templates to Langfuse Prompt Management:

```bash
poetry run python -m src.synthetic_ner.sync_langfuse_prompts
```

Generate from an existing schema:

```bash
poetry run python generator.py --from-schema schemas/en_indictment_financial_fraud_002.json
```

Generate multiple documents:

```bash
poetry run python generator.py --documents 3
```

Generate with the classic non-graph flow:

```bash
poetry run python generator.py --workflow-mode classic
```

Visualize entity coverage:

```bash
poetry run python visualize.py
```

Check code quality:

```bash
poetry run ruff check .
```
