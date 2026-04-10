"""Section writer for the LangGraph workflow."""

from __future__ import annotations

import re

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS
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
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.chunk_words = chunk_words
        self.context_tail_chars = context_tail_chars
        self.writer_temperature = writer_temperature

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
            )
            text = re.sub(r"<think>.*?</think>", "", result.text, flags=re.DOTALL).strip()
            if not text:
                break
            chunks.append(text)
            words_so_far += len(text.split())
            chunk_index += 1

        return "\n\n".join(chunks) if chunks else "[section not generated]"
