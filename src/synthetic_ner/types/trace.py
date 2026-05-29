from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.synthetic_ner.types.app_config import WorkflowPromptsConfig


@dataclass(slots=True)
class TraceHandle:
    observation: Any
    metadata: dict[str, Any]


@dataclass(slots=True)
class DocumentTraceSession:
    enabled: bool
    trace_id: str | None
    trace_url: str | None


@dataclass(frozen=True, slots=True)
class NodeExecutionRecord:
    node_name: str
    status: str
    latency_ms: int
    next_node: str | None
    section_name: str | None


@dataclass(frozen=True, slots=True)
class ResolvedWorkflowPrompts:
    prompts: WorkflowPromptsConfig
    prompt_clients: dict[str, Any]
    sync_summary: str
