# Changelog

All notable generator changes are recorded here. The current generator version is
the semantic version in `pyproject.toml`.

## 0.1.0 - 2026-05-29

### Added

- Prefect generation deployment for synthetic NER document creation.
- Prefect quality deployment for scoring existing generated documents.
- Single document quality artifact for business-friendly review.
- Deterministic quality score explanation with per-section penalty breakdown.
- Expected section word target visibility in quality tables.
- Langfuse prompt/response references for section inspection.
- Section-level critic rubric display where Langfuse scores are available.
- Semantic generator version stamping in generation and quality reports.

### Changed

- Split Prefect orchestration into generation and quality flows.
- Moved the root Prefect entrypoint beside `main.py`.
- Quality scoring now separates deterministic output health from LLM critic rubrics.

### Scoring

- Deterministic quality score starts at `100`.
- It subtracts validator issue penalties, revision penalties, and short-section penalties.
- LLM critic rubrics remain separate `1-5` semantic/legal quality scores.

### Known Limitations

- Documents generated before version stamping may show `not recorded` for generator version.
- Historical quality reports do not update retroactively.
