"""Critic node for section review."""

from __future__ import annotations

from dataclasses import dataclass

from src.synthetic_ner.utils import render_inline_template


@dataclass(slots=True)
class CriticResult:
    approved: bool
    issues: list[str]
    revision_instruction: str
    raw_text: str


class SectionCritic:
    def __init__(
        self,
        *,
        client,
        prompts: dict,
        critic_temperature: float,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.critic_temperature = critic_temperature

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
        user_prompt = render_inline_template(
            self.prompts["critic_user"],
            memory_text=memory_text,
            section_plan=section_plan,
            section_text=section_text,
            section_name=section_name,
        )
        result = self.client.invoke(
            doc_id=doc_id,
            task_id=f"critic_{section_name}_r{revision_round}",
            stage="critic",
            system_prompt=self.prompts["critic_system"],
            user_prompt=user_prompt,
            parent_task_id=parent_task_id,
            temperature=self.critic_temperature,
        )
        return self._parse_result(result.text)

    def _parse_result(self, raw_text: str) -> CriticResult:
        normalized = raw_text.replace("\r", "")
        lowered = normalized.lower()
        approved = "approved: yes" in lowered

        issues = []
        revision_instruction = "keep as is"

        issues_marker = normalized.find("ISSUES:")
        revision_marker = normalized.find("REVISION:")
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

        return CriticResult(
            approved=approved and not issues,
            issues=issues,
            revision_instruction=revision_instruction,
            raw_text=raw_text,
        )
