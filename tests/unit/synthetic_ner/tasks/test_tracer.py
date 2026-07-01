from src.synthetic_ner.tasks.document_generation.tracer import TraceStore
from src.synthetic_ner.types.app_config import LangfuseConfig


class FakeObservation:
    def __init__(self) -> None:
        self.ended = False
        self.updates = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True

    def score(self, **kwargs) -> None:
        self.updates.append({"score": kwargs})


class FakeObservationContext:
    def __init__(self, observation: FakeObservation) -> None:
        self.observation = observation

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.flush_count = 0
        self.observations = []

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace-{seed}"

    def get_trace_url(self, *, trace_id: str) -> str:
        return f"http://langfuse.local/trace/{trace_id}"

    def start_as_current_observation(self, **kwargs) -> FakeObservationContext:
        observation = FakeObservation()
        self.observations.append((kwargs, observation))
        return FakeObservationContext(observation)

    def start_observation(self, **kwargs) -> FakeObservation:
        observation = FakeObservation()
        self.observations.append((kwargs, observation))
        return observation

    def flush(self) -> None:
        self.flush_count += 1


def test_langfuse_observations_flush_when_started_and_completed(monkeypatch):
    fake_client = FakeLangfuseClient()
    monkeypatch.setenv("TEST_LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("TEST_LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setattr(
        "src.synthetic_ner.tasks.document_generation.tracer.Langfuse",
        lambda **kwargs: fake_client,
    )
    trace_store = TraceStore(
        LangfuseConfig(
            enabled=True,
            host="http://localhost:3000",
            public_key_env="TEST_LANGFUSE_PUBLIC_KEY",
            secret_key_env="TEST_LANGFUSE_SECRET_KEY",
        )
    )

    trace_store.start_document_run(
        doc_id="doc-1",
        name="document-workflow",
        input_payload={"doc_id": "doc-1"},
        metadata={"doc_id": "doc-1"},
    )
    assert fake_client.flush_count == 1

    handle = trace_store.start_trace(
        doc_id="doc-1",
        task_id="writer_history_r0_chunk_01",
        stage="writer",
        model="qwen",
        prompt="prompt",
    )
    assert fake_client.flush_count == 2

    trace_store.record_llm_call(
        handle,
        prompt="prompt",
        response="response",
        metadata={"stage": "writer", "task_id": "writer_history_r0_chunk_01"},
    )
    assert fake_client.flush_count == 3
    assert handle.observation is not None
    assert handle.observation.ended
