# Code Quality Improvement User Story

## User Story

As a developer maintaining the Synthetic Legal NER Generator, I want the codebase
to pass static quality checks with clear maintainability and complexity thresholds,
so that future changes can be made safely without expanding hard-to-review modules
or hiding risky logic in large coordinator functions.

## Current Static Inspection Baseline

Commands run:

```bash
poetry run ruff check .
poetry run radon mi src tests -s
poetry run radon cc src tests -s -a
poetry run radon raw src tests
poetry run radon hal src tests
```

Current blockers:

- `ruff check .` fails with 12 issues, all in `src/synthetic_ner/tasks/validators.py`.
- Maintainability hotspots:
  - `src/synthetic_ner/tasks/validators.py` - `C (0.00)`
  - `src/synthetic_ner/tasks/tracer.py` - `C (0.96)`
  - `src/synthetic_ner/tasks/orchestrator.py` - `C (1.67)`
  - `src/synthetic_ner/case.py` - `C (5.10)`
- Highest complexity functions:
  - `validate_section_text` - `D (28)`
  - `repair_section_text` - `D (23)`
  - `normalize_person_record` - `D (21)`
  - `TraceStore.get_llm_run_summary` - `C (20)`
  - `_dedupe_repeated_content` - `C (19)`
  - `_validate_facts_contract` - `C (18)`

## Branch Progress

Commands rerun on `codex-code-quality-refactor`:

```bash
poetry run ruff check .
poetry run pytest tests/synthetic_ner/models/test_ollama_think_config.py tests/synthetic_ner/tasks/test_validators.py -q
poetry run radon mi src tests -s
poetry run radon cc src tests -s -a
```

Current result:

- `ruff check .` passes.
- Focused pytest coverage passes with 9 tests.
- `make mi` / `radon mi` reports no `C` grade files under `src/`.
- Remaining `B` maintainability files are focused helper modules:
  - `src/synthetic_ner/case_entities.py` - `B (15.58)`
  - `src/synthetic_ner/tasks/validators.py` - `B (17.94)`
  - `src/synthetic_ner/tasks/trace_metrics.py` - `B (18.15)`

Refactor slices completed:

- Split deterministic validators into contract, memory, fallback, and repetition modules.
- Removed unreachable legacy section nodes from the LangGraph orchestrator.
- Moved generation report formatting out of `orchestrator.py`.
- Moved tracer summary/metadata helpers out of `tracer.py`.
- Moved case person/organisation generation and normalization out of `case.py`.

## Acceptance Criteria

- `poetry run ruff check .` passes.
- `make mi` reports no `C` grade files under `src/`.
- No new function or method should exceed Radon complexity `B`; existing higher
  complexity functions are tracked by `poetry run radon cc src tests -s -a`.
- `src/synthetic_ner/tasks/validators.py` is split into focused modules and has
  maintainability index above `20`.
- `src/synthetic_ner/tasks/orchestrator.py` no longer owns report formatting and
  section execution internals in the same file.
- Existing generation behavior is covered by focused regression tests before and
  after refactors.

## First Refactor Slice

Extract validator responsibilities into focused modules:

- formatting and cleanup checks
- known-value extraction and unknown entity checks
- facts contract validation
- deterministic repair and fallback text generation

Target outcome:

- `validate_section_text` becomes a coordinator with complexity at or below `B`.
- `repair_section_text` becomes a coordinator with complexity at or below `B`.
- Existing `validators.py` imports remain compatible or are replaced in one small
  adapter pass.

## Second Refactor Slice

Split orchestration support code from `orchestrator.py`:

- move section execution and revision loop to `tasks/section_runner.py`
- move generation report formatting to `tasks/generation_report.py`
- keep LangGraph wiring in `orchestrator.py`

Target outcome:

- `orchestrator.py` drops below 700 LOC.
- report formatting and LLM analytics can be tested without building a workflow graph.
- section execution can be tested with fake planner/writer/critic objects.

## Quality Gates To Add

Add Make targets:

```make
complexity:
	poetry run radon cc src tests -s -a

quality:
	poetry run ruff check .
	poetry run radon mi src tests -s
	poetry run radon cc src tests -s -a
```

Use `make quality` before merging refactor work.
