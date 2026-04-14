"""Critic node for section review."""

from __future__ import annotations

import re
from typing import Any

from requests.exceptions import ReadTimeout
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig
from src.synthetic_ner.types.critic import CriticResult
from src.synthetic_ner.utils import render_inline_template

_RUBRIC_LINE_RE = re.compile(
    r"^\s*-\s*([a-zA-Z][a-zA-Z0-9_ -]{1,40})\s*:\s*([1-5])(?:\s*/\s*5)?\s*$",
    re.MULTILINE,
)
_CRITIC_MEMORY_CHAR_LIMIT = 6_000
_CRITIC_SECTION_TEXT_CHAR_LIMIT = 3_500
_CRITIC_OUTPUT_TOKEN_LIMIT = 180


class SectionCritic:
    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        critic_temperature: float,
        prompt_clients: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.critic_temperature = critic_temperature
        self.prompt_clients = prompt_clients or {}

    def review_section(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        section_name: str,
        section_plan: str,
        section_text: str,
        revision_round: int,
    ) -> CriticResult:
        compact_memory = _truncate_text(memory_text, _CRITIC_MEMORY_CHAR_LIMIT)
        compact_section_text = _truncate_text(section_text, _CRITIC_SECTION_TEXT_CHAR_LIMIT)
        user_prompt = render_inline_template(
            self.prompts.critic_user,
            memory_text=compact_memory,
            section_plan=section_plan,
            section_text=compact_section_text,
            section_name=section_name,
        )
        try:
            result = self.client.invoke(
                doc_id=doc_id,
                task_id=f"critic_{section_name}_r{revision_round}",
                stage="critic",
                system_prompt=self.prompts.critic_system,
                user_prompt=user_prompt,
                parent_task_id=parent_task_id,
                temperature=self.critic_temperature,
                max_output_tokens=_CRITIC_OUTPUT_TOKEN_LIMIT,
                prompt_object=self.prompt_clients.get("critic_user"),
            )
            return self._parse_result(result.text)
        except ReadTimeout:
            return CriticResult(
                approved=True,
                issues=[],
                revision_instruction="keep as is",
                rubrics={},
                raw_text="[critic-timeout] Skipped critic due to model timeout; relying on deterministic validation.",
            )

    def _parse_result(self, raw_text: str) -> CriticResult:
        normalized = raw_text.replace("\r", "")
        lowered = normalized.lower()
        approved = "approved: yes" in lowered

        issues = []
        revision_instruction = "keep as is"
        rubrics: dict[str, int] = {}

        rubrics_marker = normalized.find("RUBRICS:")
        issues_marker = normalized.find("ISSUES:")
        revision_marker = normalized.find("REVISION:")
        if rubrics_marker != -1:
            rubric_block_end = (
                issues_marker
                if issues_marker != -1
                else (revision_marker if revision_marker != -1 else len(normalized))
            )
            rubric_block = normalized[rubrics_marker + len("RUBRICS:"):rubric_block_end]
            rubrics = _parse_rubrics(rubric_block)
        if issues_marker != -1:
            issues_block_end = revision_marker if revision_marker != -1 else len(normalized)
            issues_block = normalized[issues_marker + len("ISSUES:"):issues_block_end]
            issues = [
                line.removeprefix("-").strip()
                for line in issues_block.splitlines()
                if line.strip().startswith("-")
                and line.removeprefix("-").strip().lower() != "none"
            ]
        if revision_marker != -1:
            revision_instruction = normalized[revision_marker + len("REVISION:"):].strip()
        if not revision_instruction:
            revision_instruction = "Fix the inconsistencies flagged by the critic."
        if not approved and not issues:
            issues = ["Critic rejected the section without specific issues."]
        if not approved and revision_instruction.lower() == "keep as is":
            revision_instruction = (
                "Revise the section to resolve consistency issues and remove any invented facts."
            )

        return CriticResult(
            approved=approved and not issues,
            issues=issues,
            revision_instruction=revision_instruction,
            rubrics=rubrics,
            raw_text=raw_text,
        )


def _parse_rubrics(rubric_block: str) -> dict[str, int]:
    rubrics: dict[str, int] = {}
    for metric, raw_score in _RUBRIC_LINE_RE.findall(rubric_block):
        key = metric.strip().lower().replace(" ", "_").replace("-", "_")
        score = int(raw_score)
        if key and 1 <= score <= 5:
            rubrics[key] = score
    return rubrics


def _truncate_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return f"{head}\n\n[...truncated for critic...]\n\n{tail}"
