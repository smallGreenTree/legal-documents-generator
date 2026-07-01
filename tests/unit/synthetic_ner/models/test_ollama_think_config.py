from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.synthetic_ner.config import build_app_config, load_app_config
from src.synthetic_ner.models.factory import build_model_client
from src.synthetic_ner.tasks.document_generation.tracer import TraceStore
from src.synthetic_ner.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_config_loads_think_false_for_ollama_and_all_routed_stages():
    app_config = load_app_config(PROJECT_ROOT / "config.yaml")

    assert not hasattr(app_config, "ollama")
    assert app_config.model_routing.stages["planner"].think is False
    assert app_config.model_routing.stages["writer"].think is False
    assert app_config.model_routing.stages["critic"].think is False
    assert app_config.model_routing.stages["planner"].top_p == 0.9
    assert app_config.model_routing.stages["writer"].top_p == 0.9
    assert app_config.model_routing.stages["critic"].top_p == 0.9


def test_inactive_planner_prompts_are_not_required():
    app_config = load_app_config(PROJECT_ROOT / "config.yaml")

    assert app_config.workflow.planner.active is False
    assert app_config.workflow.prompts.document_planner_system == ""
    assert app_config.workflow.prompts.section_planner_user == ""
    assert app_config.workflow.prompts.writer_system.strip()


def test_planner_active_requires_planner_prompts():
    raw_config = load_config(PROJECT_ROOT / "config.yaml")
    case_config = load_config(PROJECT_ROOT / "config_case" / "case_1.yaml")
    raw_config["workflow"]["planner"]["active"] = True

    with pytest.raises(ValueError, match="document_planner_system"):
        build_app_config(
            raw_config,
            case_cfg=case_config,
            config_path=PROJECT_ROOT / "config.yaml",
        )


def test_config_requires_explicit_stage_routes():
    raw_config = load_config(PROJECT_ROOT / "config.yaml")
    case_config = load_config(PROJECT_ROOT / "config_case" / "case_1.yaml")
    del raw_config["model_routing"]["stages"]["writer"]

    with pytest.raises(ValueError, match="writer"):
        build_app_config(
            raw_config,
            case_cfg=case_config,
            config_path=PROJECT_ROOT / "config.yaml",
        )


@pytest.mark.parametrize("stage", ["planner", "writer", "critic"])
def test_ollama_stage_clients_send_think_false(monkeypatch, stage):
    app_config = load_app_config(PROJECT_ROOT / "config.yaml")
    captured_requests = []

    def fake_post(url, **kwargs):
        captured_requests.append({"url": url, **kwargs})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "response": "ok",
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    monkeypatch.setattr(
        "src.synthetic_ner.models.ollama_client.requests.post",
        fake_post,
    )

    provider_config = app_config.model_routing.stages[stage]
    assert provider_config.think is False

    client = build_model_client(
        stage=stage,
        routing=app_config.model_routing,
        tracer=TraceStore(replace(app_config.langfuse, enabled=False)),
    )
    client.invoke(
        doc_id="doc",
        task_id=f"{stage}_smoke",
        stage=stage,
        system_prompt="system",
        user_prompt="user",
        temperature=0.0,
        max_output_tokens=8,
    )

    assert captured_requests
    request_json = captured_requests[0]["json"]
    assert request_json["model"] == provider_config.model
    assert request_json["think"] is False
    assert request_json["options"]["temperature"] == 0.0
    assert request_json["options"]["top_p"] == 0.9
