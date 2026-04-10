"""Planner nodes for the LangGraph workflow."""

from __future__ import annotations

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig
from src.synthetic_ner.utils import render_inline_template


class Planner:
    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        planner_temperature: float,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.planner_temperature = planner_temperature

    def plan_document(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        doc_type: str,
        fraud_type: str,
        case_number: str,
        section_order: list[str],
    ) -> str:
        section_list = "\n".join(
            f"- {section_name}: {SECTION_DESCRIPTIONS.get(section_name, section_name)}"
            for section_name in section_order
        )
        user_prompt = render_inline_template(
            self.prompts.document_planner_user,
            memory_text=memory_text,
            doc_type=doc_type,
            fraud_type=fraud_type.replace("_", " "),
            case_number=case_number,
            section_list=section_list,
        )
        result = self.client.invoke(
            doc_id=doc_id,
            task_id="planner_document",
            stage="planner",
            system_prompt=self.prompts.document_planner_system,
            user_prompt=user_prompt,
            parent_task_id=parent_task_id,
            temperature=self.planner_temperature,
        )
        return result.text

    def plan_section(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        document_plan: str,
        doc_type: str,
        section_name: str,
        word_target: int,
    ) -> str:
        user_prompt = render_inline_template(
            self.prompts.section_planner_user,
            memory_text=memory_text,
            document_plan=document_plan,
            doc_type=doc_type,
            section_name=section_name,
            section_description=SECTION_DESCRIPTIONS.get(section_name, section_name),
            word_target=word_target,
        )
        result = self.client.invoke(
            doc_id=doc_id,
            task_id=f"planner_{section_name}",
            stage="planner",
            system_prompt=self.prompts.section_planner_system,
            user_prompt=user_prompt,
            parent_task_id=parent_task_id,
            temperature=self.planner_temperature,
        )
        return result.text
