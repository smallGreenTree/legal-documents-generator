"""Feedback-driven section revision for the LangGraph workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.document_generation.artifacts.partial_sections import (
    PartialSectionStore,
)
from src.synthetic_ner.tasks.document_generation.context.prompts import (
    build_section_context,
    build_section_contract,
)
from src.synthetic_ner.tasks.document_generation.validation.validators import (
    clean_generated_section_text,
)
from src.synthetic_ner.text.templates import render_prompt_template
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig


class SectionPolisher:
    """Revise an existing draft in response to critic and validator feedback."""

    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        temperature: float,
        max_output_tokens: int,
        prompt_clients: dict[str, Any] | None = None,
        partial_output_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_clients = prompt_clients or {}
        self._partial_store = PartialSectionStore(
            partial_output_dir,
            thread_name_prefix="partial-section-polisher",
        )

    def polish_section(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        section_name: str,
        current_text: str,
        revision_instruction: str,
        revision_round: int,
    ) -> str:
        prompt_client = self.prompt_clients.get("polisher_user")
        user_prompt = render_prompt_template(
            self.prompts.polisher_user,
            prompt_client=prompt_client,
            section_context=build_section_context(memory_text, section_name),
            section_contract=build_section_contract(section_name),
            section_name=section_name,
            current_section=current_text,
            revision_instruction=revision_instruction,
        )
        task_id = f"polish_{section_name}_r{revision_round}"
        result = self.client.invoke(
            doc_id=doc_id,
            task_id=task_id,
            stage="polisher",
            system_prompt=self.prompts.polisher_system,
            user_prompt=user_prompt,
            parent_task_id=parent_task_id,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            prompt_object=prompt_client,
        )
        polished_text = clean_generated_section_text(result.text)
        if not polished_text:
            polished_text = clean_generated_section_text(current_text)

        self._partial_store.write(
            doc_id=doc_id,
            section_name=section_name,
            revision_round=revision_round,
            chunk_index=1,
            chunk_text=polished_text,
            combined_text=polished_text,
            task_id=task_id,
            metadata=dict(result.metadata),
            complete=True,
        )
        self._partial_store.flush()
        return polished_text
