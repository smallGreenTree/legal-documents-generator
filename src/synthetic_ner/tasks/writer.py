"""Section writer for the LangGraph workflow."""

from __future__ import annotations

from typing import Any

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS
from src.synthetic_ner.tasks.validators import clean_generated_section_text
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig
from src.synthetic_ner.utils import render_inline_template


class SectionWriter:
    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        chunk_words: int,
        context_tail_chars: int,
        writer_temperature: float,
        prompt_clients: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.chunk_words = chunk_words
        self.context_tail_chars = context_tail_chars
        self.writer_temperature = writer_temperature
        self.prompt_clients = prompt_clients or {}

    def write_section(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        document_plan: str,
        section_name: str,
        section_plan: str,
        case_number: str,
        word_target: int,
        revision_instruction: str = "",
        revision_round: int = 0,
    ) -> str:
        chunks = []
        words_so_far = 0
        chunk_index = 1

        while words_so_far < word_target:
            remaining = word_target - words_so_far
            chunk_target = min(self.chunk_words, remaining)
            previous_tail = chunks[-1][-self.context_tail_chars:] if chunks else "n/a"
            user_prompt = render_inline_template(
                self.prompts.writer_user,
                memory_text=memory_text,
                document_plan=document_plan,
                section_plan=section_plan,
                section_name=section_name,
                section_description=SECTION_DESCRIPTIONS.get(section_name, section_name),
                case_number=case_number,
                word_target=chunk_target,
                previous_tail=previous_tail,
                revision_instruction=revision_instruction or "none",
            )
            if revision_instruction.strip():
                user_prompt = (
                    f"{user_prompt}\n\n"
                    "REVISION REQUIREMENTS:\n"
                    f"{revision_instruction.strip()}\n"
                    "- Use only entities, dates, case references, and VAT/reference numbers "
                    "present in CASE_MEMORY.\n"
                    "- Remove any unknown identifiers instead of inventing replacements.\n"
                )
            task_id = (
                f"writer_{section_name}_r{revision_round}_chunk_{chunk_index:02d}"
            )
            result = self.client.invoke(
                doc_id=doc_id,
                task_id=task_id,
                stage="writer",
                system_prompt=self.prompts.writer_system,
                user_prompt=user_prompt,
                parent_task_id=parent_task_id,
                temperature=self.writer_temperature,
                max_output_tokens=_estimate_writer_output_tokens(chunk_target),
                prompt_object=self.prompt_clients.get("writer_user"),
            )
            text = clean_generated_section_text(result.text)
            if not text:
                break
            chunks.append(text)
            words_so_far += len(text.split())
            chunk_index += 1

        if not chunks:
            return "[section not generated]"
        return clean_generated_section_text("\n\n".join(chunks))


def _estimate_writer_output_tokens(chunk_words: int) -> int:
    # Conservative cap to prevent runaway generations while preserving section quality.
    estimated = int(chunk_words * 1.8)
    return max(220, min(1100, estimated))
