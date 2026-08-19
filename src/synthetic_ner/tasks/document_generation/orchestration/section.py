"""Writer, review, and polishing lifecycle for a single document section."""

from __future__ import annotations

from pathlib import Path

from src.synthetic_ner.tasks.document_generation.context.memory import CaseMemoryManager
from src.synthetic_ner.tasks.document_generation.context.prompts import build_section_contract
from src.synthetic_ner.tasks.document_generation.stages.critic import SectionCritic
from src.synthetic_ner.tasks.document_generation.stages.polisher import SectionPolisher
from src.synthetic_ner.tasks.document_generation.stages.writer import SectionWriter
from src.synthetic_ner.tasks.document_generation.validation.validators import (
    clean_generated_section_text,
    validate_section_text,
)
from src.synthetic_ner.types.document_generation import SectionWorkflowResult
from src.synthetic_ner.types.document_inputs import DocumentInputs
from src.synthetic_ner.types.runtime_context import RuntimeContext


class SectionWorkflowRunner:
    def __init__(
        self,
        *,
        context: RuntimeContext,
        document: DocumentInputs,
        doc_id: str,
        memory_path: Path,
        memory_manager: CaseMemoryManager,
        writer: SectionWriter,
        polisher: SectionPolisher | None,
        critic: SectionCritic | None,
    ) -> None:
        self.context = context
        self.document = document
        self.doc_id = doc_id
        self.memory_path = memory_path
        self.memory_manager = memory_manager
        self.writer = writer
        self.polisher = polisher
        self.critic = critic

    def run(self, *, memory_text: str, section_name: str) -> SectionWorkflowResult:
        section_contract = build_section_contract(section_name)
        section_text = self.writer.write_section(
            doc_id=self.doc_id,
            parent_task_id=None,
            memory_text=memory_text,
            section_name=section_name,
            case_number=self.document.metadata["case_number"],
            word_target=self.context.section_word_targets[section_name],
        )
        section_text = clean_generated_section_text(section_text)
        issues: list[str] = []
        revision_count = 0
        review_parent_task_id = f"writer_{section_name}_r0"

        while True:
            issues, critic_instruction = self._review_section(
                memory_text=memory_text,
                section_name=section_name,
                section_text=section_text,
                revision_round=revision_count,
                parent_task_id=review_parent_task_id,
            )
            if (
                not issues
                or revision_count >= self.context.workflow_cfg.max_revisions
                or self.polisher is None
            ):
                break

            revision_count += 1
            section_text = self.polisher.polish_section(
                doc_id=self.doc_id,
                parent_task_id=(
                    f"critic_{section_name}_r{revision_count - 1}"
                    if self.critic is not None
                    else f"validator_{section_name}_r{revision_count - 1}"
                ),
                memory_text=memory_text,
                section_name=section_name,
                current_text=section_text,
                revision_instruction=combine_revision_instruction(
                    critic_instruction=critic_instruction,
                    issues=issues,
                ),
                revision_round=revision_count,
            )
            section_text = clean_generated_section_text(section_text)
            review_parent_task_id = f"polish_{section_name}_r{revision_count}"

        final_text, final_issues = self._finalize_section_text(
            section_name=section_name,
            section_text=section_text,
            issues=issues,
        )
        return SectionWorkflowResult(
            section_name=section_name,
            section_contract=section_contract,
            section_text=final_text,
            issues=final_issues,
        )

    def _review_section(
        self,
        *,
        memory_text: str,
        section_name: str,
        section_text: str,
        revision_round: int,
        parent_task_id: str,
    ) -> tuple[list[str], str]:
        issues: list[str] = []
        critic_instruction = ""
        if self.critic is not None:
            review = self.critic.review_section(
                doc_id=self.doc_id,
                parent_task_id=parent_task_id,
                memory_text=memory_text,
                section_name=section_name,
                section_text=section_text,
                revision_round=revision_round,
            )
            issues = list(review.issues)
            critic_instruction = review.revision_instruction
            if review.blocking and not issues:
                issues.append("Critic marked the section as blocking.")

        validator_issues = self._validate_section(
            section_name=section_name,
            section_text=section_text,
            memory_text=memory_text,
        )
        for issue in validator_issues:
            if issue not in issues:
                issues.append(issue)
        return issues, critic_instruction

    def _validate_section(
        self,
        *,
        section_name: str,
        section_text: str,
        memory_text: str,
    ) -> list[str]:
        return validate_section_text(
            section_name=section_name,
            section_text=section_text,
            memory_text=memory_text,
            word_target=self.context.section_word_targets[section_name],
            min_completion_ratio=self.context.workflow_cfg.writer.min_completion_ratio,
            enabled_validators=self.context.workflow_cfg.validators,
        )

    def _finalize_section_text(
        self,
        *,
        section_name: str,
        section_text: str,
        issues: list[str],
    ) -> tuple[str, list[str]]:
        final_text = clean_generated_section_text(section_text)
        final_issues = self._validate_section(
            section_name=section_name,
            section_text=final_text,
            memory_text=self.memory_manager.read_memory(self.memory_path),
        )
        if final_issues:
            print(
                "  Warning : section output still has validation issues "
                f"({section_name}): {'; '.join(final_issues)}"
            )
        combined_issues = list(issues)
        for issue in final_issues:
            if issue not in combined_issues:
                combined_issues.append(issue)
        return final_text, combined_issues


def combine_revision_instruction(*, critic_instruction: str, issues: list[str]) -> str:
    parts = [
        "Revise the existing section using only SECTION_CONTEXT and SECTION_CONTRACT.",
        "Resolve these issues:",
        *(f"- {issue}" for issue in issues),
    ]
    normalized_critic_instruction = critic_instruction.strip()
    if normalized_critic_instruction and normalized_critic_instruction.lower() != "keep as is":
        parts.extend(["", normalized_critic_instruction])
    return "\n".join(parts)
