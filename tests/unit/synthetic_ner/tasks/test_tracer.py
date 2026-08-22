from types import SimpleNamespace

from src.synthetic_ner.tasks.document_generation.observability.tracer import TraceStore
from src.synthetic_ner.types.app_config import MlflowConfig


class FakeSpan:
    def __init__(self, trace_id: str, span_id: str) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.inputs = None
        self.outputs = None
        self.attributes = {}
        self.ended = False

    def set_inputs(self, value) -> None:
        self.inputs = value

    def set_outputs(self, value) -> None:
        self.outputs = value

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_attributes(self, values) -> None:
        self.attributes.update(values)

    def record_exception(self, exception) -> None:
        self.attributes["exception"] = str(exception)

    def end(self, **kwargs) -> None:
        self.ended = True


class FakeSpanContext:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.span.ended = True


class FakeMlflow:
    def __init__(self) -> None:
        self.spans = []
        self.genai = SimpleNamespace()
        self.trace_updates = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str):
        self.experiment_name = name
        return SimpleNamespace(experiment_id="7")

    def start_span(self, **kwargs) -> FakeSpanContext:
        span = FakeSpan("tr-test", f"span-{len(self.spans) + 1}")
        self.spans.append((kwargs, span))
        return FakeSpanContext(span)

    def update_current_trace(self, **kwargs) -> None:
        self.trace_updates.append(kwargs)


def test_mlflow_spans_are_started_and_completed(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setattr(
        "src.synthetic_ner.tasks.document_generation.observability.tracer.mlflow",
        fake,
    )
    trace_store = TraceStore(
        MlflowConfig(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment_name="ner-platform",
            service_name="synthetic-dataset-generation",
            pipeline_stage="synthetic_dataset_generation",
            trace_name="document-workflow",
            prompt_name_prefix="synthetic_ner",
            prompt_alias="production",
        )
    )

    session = trace_store.start_document_run(
        doc_id="doc-1",
        input_payload={"doc_id": "doc-1"},
        metadata={"doc_id": "doc-1"},
    )
    assert session.trace_id == "tr-test"
    assert session.trace_url == "http://localhost:5000/#/experiments/7/traces/tr-test"
    assert fake.trace_updates == [
        {
            "session_id": "doc-1",
            "tags": {
                "service.name": "synthetic-dataset-generation",
                "pipeline.stage": "synthetic_dataset_generation",
            },
            "metadata": {"service.name": "synthetic-dataset-generation"},
        }
    ]

    handle = trace_store.start_trace(
        doc_id="doc-1",
        task_id="writer_history_r0_chunk_01",
        stage="writer",
        model="qwen",
        prompt="prompt",
    )
    trace_store.record_llm_call(
        handle,
        prompt="prompt",
        response="response",
        metadata={"stage": "writer", "task_id": "writer_history_r0_chunk_01"},
    )

    assert handle.observation is not None
    assert handle.observation.ended
    assert handle.observation.outputs == "response"
