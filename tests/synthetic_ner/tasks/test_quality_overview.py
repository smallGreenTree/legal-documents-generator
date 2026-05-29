from types import SimpleNamespace

from src.synthetic_ner.tasks.quality_overview import (
    _fetch_langfuse_observation_metadata,
    _fetch_langfuse_scores,
    _prompt_response_reference_rows,
    _rubric_summary_from_scores,
    build_quality_overview,
    format_duration_ms,
    format_model_workflow_markdown,
    format_run_health_markdown,
    latency_band,
)


def test_duration_and_latency_bands_are_human_readable():
    assert format_duration_ms(24_000) == "24s"
    assert format_duration_ms(78_000) == "1m 18s"
    assert format_duration_ms(222_000) == "3m 42s"
    assert format_duration_ms(23_590_370) == "6h 33m 10s"

    assert latency_band(30_000) == "very fast"
    assert latency_band(30_001) == "fast"
    assert latency_band(90_000) == "fast"
    assert latency_band(90_001) == "slow"
    assert latency_band(180_000) == "slow"
    assert latency_band(180_001) == "very slow"


def test_quality_overview_uses_existing_report_and_quality_score(tmp_path):
    context = SimpleNamespace(
        output_dir=tmp_path / "output",
        memory_dir=tmp_path / "memory",
        schema_dir=tmp_path / "schemas",
    )
    doc_id = "en_indictment_financial_fraud_004"
    _write_existing_artifacts(context, doc_id)

    overview = build_quality_overview(
        context=context,
        doc_id=doc_id,
        quality_report=_quality_report(doc_id),
        rubric_summary={
            "overall": 3.5,
            "lowest_metric": {"metric": "grounding", "score": 2.5},
            "trace_url": "http://localhost:3000/project/x/traces/trace-1",
            "sections": [
                {
                    "section": "persons",
                    "overall": 4.25,
                    "grounding": 4,
                    "completeness": 5,
                    "legal_style": 4,
                    "chronology": 4,
                    "revision": 0,
                },
                {
                    "section": "history",
                    "overall": 2.75,
                    "grounding": 2,
                    "completeness": 3,
                    "legal_style": 3,
                    "chronology": 3,
                    "revision": 2,
                },
            ],
            "prompt_response_refs": [
                {
                    "section": "persons",
                    "text_url": "http://localhost:3000/project/x/traces/trace-1?observation=obs-persons-polish",
                    "critic_url": "http://localhost:3000/project/x/traces/trace-1?observation=obs-persons-r0",
                    "text_links": "[r0 chunk 01](http://localhost:3000/project/x/traces/trace-1?observation=obs-persons-polish)",
                    "critic_link": "[critic r0](http://localhost:3000/project/x/traces/trace-1?observation=obs-persons-r0)",
                }
            ],
            "status": "available",
        },
    )
    run_health = format_run_health_markdown(overview)
    model_workflow = format_model_workflow_markdown(overview)

    assert overview["readiness"] == "Needs Review"
    assert overview["model_workflow"]["total_tokens"] == 3_800
    assert "6m 40s" in model_workflow
    assert "Total model time for document" in model_workflow
    assert "Document generator version | 0.1.0" in run_health
    assert "Document generator reference | generator-v0.1.0" in run_health
    assert "Document generator summary | First versioned test workflow." in run_health
    assert "Document generator git commit | abc123abc123" in run_health
    assert "Quality analyzer version | 0.1.0" in run_health
    assert "Quality analyzer reference | generator-v0.1.0" in run_health
    assert "`Total latency` is the sum of all LLM calls" in model_workflow
    assert "[open trace](http://localhost:3000/project/x/traces/trace-1)" in model_workflow
    assert "writer" in model_workflow
    assert "3.5 / 5" in model_workflow
    assert "grounding (2.5 / 5)" in model_workflow
    assert "## Section Rubrics" in model_workflow
    assert "## Prompt/Response References" in model_workflow
    assert "Text generation prompts/responses" in model_workflow
    assert "obs-persons-polish" in model_workflow
    assert (
        "| persons | 90 | 120 | 120 | 4.25 | 4 | 5 | 4 | 4 | 0 | "
        "[trace](http://localhost:3000/project/x/traces/trace-1) | none |"
    ) in model_workflow
    assert (
        "| history | 66 | 220 | 120 | 2.75 | 2 | 3 | 3 | 3 | 2 | "
        "[trace](http://localhost:3000/project/x/traces/trace-1) | "
        "Section contains repeated sentence fragments. |"
    ) in model_workflow
    assert str(tmp_path) not in run_health
    assert "SECTION_CONTEXT" not in run_health
    assert "LLM Calls" not in model_workflow


def test_rubric_summary_groups_latest_scores_by_section():
    scores = [
        _score("rubric.grounding", 1, "obs-history-r0"),
        _score("rubric.completeness", 2, "obs-history-r0"),
        _score("rubric.grounding", 4, "obs-history-r2"),
        _score("rubric.completeness", 5, "obs-history-r2"),
        _score("rubric.legal_style", 3, "obs-history-r2"),
        _score("rubric.chronology", 4, "obs-history-r2"),
        _score("rubric.grounding", 5, "obs-persons-r0"),
        _score("rubric.completeness", 5, "obs-persons-r0"),
        _score("rubric.legal_style", 4, "obs-persons-r0"),
        _score("rubric.chronology", 4, "obs-persons-r0"),
    ]
    observation_metadata = {
        "obs-history-r0": {"section_name": "history", "revision_round": 0},
        "obs-history-r2": {"section_name": "history", "revision_round": 2},
        "obs-persons-r0": {"section_name": "persons", "revision_round": 0},
    }

    summary = _rubric_summary_from_scores(
        scores,
        observation_metadata,
        trace_url="http://localhost:3000/project/x/traces/trace-1",
    )

    sections = {section["section"]: section for section in summary["sections"]}
    assert sections["history"]["revision"] == 2
    assert sections["history"]["grounding"] == 4
    assert sections["history"]["overall"] == 4
    assert (
        sections["history"]["langfuse_url"]
        == "http://localhost:3000/project/x/traces/trace-1?observation=obs-history-r2"
    )
    assert sections["persons"]["overall"] == 4.5


def test_prompt_response_references_use_latest_polisher_and_critic_observations():
    observation_metadata = {
        "obs-history-writer-r1-c1": {
            "observation_id": "obs-history-writer-r1-c1",
            "section_name": "history",
            "revision_round": 1,
            "stage": "writer",
            "task_id": "writer_history_r1_chunk_01",
        },
        "obs-history-polish-r2-c1": {
            "observation_id": "obs-history-polish-r2-c1",
            "section_name": "history",
            "revision_round": 2,
            "stage": "polisher",
            "task_id": "polish_history_r2_chunk_01",
        },
        "obs-history-polish-r2-c2": {
            "observation_id": "obs-history-polish-r2-c2",
            "section_name": "history",
            "revision_round": 2,
            "stage": "polisher",
            "task_id": "polish_history_r2_chunk_02",
        },
        "obs-history-critic-r1": {
            "observation_id": "obs-history-critic-r1",
            "section_name": "history",
            "revision_round": 1,
            "stage": "critic",
            "task_id": "critic_history_r1",
        },
        "obs-history-critic-r2": {
            "observation_id": "obs-history-critic-r2",
            "section_name": "history",
            "revision_round": 2,
            "stage": "critic",
            "task_id": "critic_history_r2",
        },
    }

    rows = _prompt_response_reference_rows(
        observation_metadata,
        "http://localhost:3000/project/x/traces/trace-1",
    )

    assert rows == [
        {
            "section": "history",
            "text_url": (
                "http://localhost:3000/project/x/traces/trace-1"
                "?observation=obs-history-polish-r2-c1"
            ),
            "critic_url": (
                "http://localhost:3000/project/x/traces/trace-1"
                "?observation=obs-history-critic-r2"
            ),
            "text_links": (
                "[r2 chunk 01](http://localhost:3000/project/x/traces/trace-1"
                "?observation=obs-history-polish-r2-c1), "
                "[r2 chunk 02](http://localhost:3000/project/x/traces/trace-1"
                "?observation=obs-history-polish-r2-c2)"
            ),
            "critic_link": (
                "[critic r2](http://localhost:3000/project/x/traces/trace-1"
                "?observation=obs-history-critic-r2)"
            ),
        }
    ]


def test_langfuse_score_fetch_uses_api_limit_pages():
    client = _FakeLangfuseClient(
        [
            _FakeScorePage([_score("rubric.grounding", 4, "obs-1")], total_pages=2),
            _FakeScorePage([_score("rubric.grounding", 5, "obs-2")], total_pages=2),
        ]
    )

    scores = _fetch_langfuse_scores(client, "trace-1")

    assert [score.value for score in scores] == [4, 5]
    assert client.api.scores.calls == [
        {"trace_id": "trace-1", "limit": 100, "page": 1},
        {"trace_id": "trace-1", "limit": 100, "page": 2},
    ]


def test_langfuse_observation_fetch_uses_legacy_v1_pages():
    client = _FakeLangfuseClient(
        score_pages=[],
        observation_pages=[
            _FakeObservationPage(
                [
                    _observation(
                        "obs-1",
                        "polish_persons_r0_chunk_01",
                        {
                            "section_name": "persons",
                            "revision_round": 0,
                            "stage": "polisher",
                            "task_id": "polish_persons_r0_chunk_01",
                        },
                    )
                ],
                total_pages=2,
            ),
            _FakeObservationPage(
                [
                    _observation(
                        "obs-2",
                        "critic_persons_r0",
                        {
                            "section_name": "persons",
                            "revision_round": 0,
                            "stage": "critic",
                            "task_id": "critic_persons_r0",
                        },
                    )
                ],
                total_pages=2,
            ),
        ],
    )

    observations = _fetch_langfuse_observation_metadata(client, "trace-1")

    assert observations["obs-1"]["stage"] == "polisher"
    assert observations["obs-2"]["task_id"] == "critic_persons_r0"
    assert client.api.legacy.observations_v1.calls == [
        {"trace_id": "trace-1", "limit": 100, "page": 1},
        {"trace_id": "trace-1", "limit": 100, "page": 2},
    ]


def _write_existing_artifacts(context, doc_id):
    output_dir = context.output_dir / doc_id
    memory_dir = context.memory_dir / f"case_{doc_id}"
    output_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    context.schema_dir.mkdir(parents=True)
    (output_dir / f"{doc_id}.txt").write_text("rendered", encoding="utf-8")
    (memory_dir / "CASE_MEMORY.md").write_text("memory", encoding="utf-8")
    (memory_dir / "RUN_HISTORY.md").write_text("history", encoding="utf-8")
    (context.schema_dir / f"{doc_id}.json").write_text("{}", encoding="utf-8")
    (output_dir / "generation_report.md").write_text(
        "\n".join(
            [
                "# Generation Report: en_indictment_financial_fraud_004",
                "",
                "- Generator version: 0.1.0",
                "- Generator version reference: generator-v0.1.0",
                "- Generator version summary: First versioned test workflow.",
                "- Generator version features: Test feature",
                "- Generator version manifest hash: sha256:test",
                "- Generator report schema version: 1.0.0",
                "- Generator git commit: abc123abc123abc123",
                "- Generator git branch: test-branch",
                "- Generator git dirty: false",
                "- Workflow mode: langgraph",
                "- Langfuse trace id: trace-1",
                "- Langfuse trace url: http://localhost:3000/project/x/traces/trace-1",
                "- Total LLM calls: 5",
                "- Total LLM latency ms: 400000",
                "- Empty LLM responses: 0",
                "- Truncated LLM calls: 1",
                "",
                "### Stage Totals",
                "",
                (
                    "| Stage | Calls | Total ms | Avg ms | Prompt Tokens | "
                    "Response Tokens | Empty | Truncated | Errors |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| planner | 2 | 100000 | 50000 | 1000 | 500 | 0 | 0 | 0 |",
                "| writer | 2 | 250000 | 125000 | 2000 | 250 | 0 | 0 | 0 |",
                "| critic | 1 | 50000 | 50000 | 50 | 0 | 0 | 1 | 0 |",
            ]
        ),
        encoding="utf-8",
    )


def _quality_report(doc_id):
    return {
        "doc_id": doc_id,
        "overall_score": 78,
        "verdict": "acceptable",
        "sections": [
            {
                "section": "persons",
                "score": 90,
                "verdict": "good",
                "revision": 0,
                "word_count": 120,
                "expected_words": 120,
                "issues": [],
            },
            {
                "section": "history",
                "score": 66,
                "verdict": "risky",
                "revision": 2,
                "word_count": 220,
                "expected_words": 120,
                "issues": ["Section contains repeated sentence fragments."],
            },
        ],
    }


def _score(name, value, observation_id):
    return SimpleNamespace(
        name=name,
        value=value,
        observation_id=observation_id,
        metadata={},
    )


class _FakeLangfuseClient:
    def __init__(self, score_pages, observation_pages=None):
        self.api = SimpleNamespace(
            scores=_FakeScoresClient(score_pages),
            legacy=SimpleNamespace(
                observations_v1=_FakeObservationsClient(observation_pages or [])
            ),
        )


class _FakeScoresClient:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        return self._pages[kwargs["page"] - 1]


class _FakeScorePage:
    def __init__(self, data, total_pages):
        self.data = data
        self.meta = SimpleNamespace(total_pages=total_pages)


class _FakeObservationsClient:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        return self._pages[kwargs["page"] - 1]


class _FakeObservationPage:
    def __init__(self, data, total_pages):
        self.data = data
        self.meta = SimpleNamespace(total_pages=total_pages)


def _observation(observation_id, name, metadata):
    return SimpleNamespace(id=observation_id, name=name, metadata=metadata)
