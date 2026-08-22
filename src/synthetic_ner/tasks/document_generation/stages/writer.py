"""Section writer for the LangGraph workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.synthetic_ner.tasks.document_generation.artifacts.partial_sections import (
    PartialSectionStore,
)
from src.synthetic_ner.tasks.document_generation.constants import SECTION_DESCRIPTIONS
from src.synthetic_ner.tasks.document_generation.context.prompts import (
    build_section_context,
    build_section_contract,
)
from src.synthetic_ner.tasks.document_generation.validation.validators import (
    clean_generated_section_text,
)
from src.synthetic_ner.text.templates import render_prompt_template
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig
from src.synthetic_ner.types.document_generation import WriterPacket


class SectionWriter:
    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        writer_temperature: float,
        max_output_tokens: int,
        prompt_clients: dict[str, Any] | None = None,
        partial_output_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.writer_temperature = writer_temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_clients = prompt_clients or {}
        self._partial_store = PartialSectionStore(
            partial_output_dir,
            thread_name_prefix="partial-section-writer",
        )

    def write_section(
        self,
        *,
        doc_id: str,
        parent_task_id: str | None,
        memory_text: str,
        section_name: str,
        case_number: str,
    ) -> str:
        prompt_client = self.prompt_clients.get("writer_user")
        section_contract = build_section_contract(section_name)
        user_prompt = render_prompt_template(
            self.prompts.writer_user,
            prompt_client=prompt_client,
            memory_text=memory_text,
            section_context=build_section_context(memory_text, section_name),
            section_contract=section_contract,
            section_name=section_name,
            section_description=SECTION_DESCRIPTIONS.get(section_name, section_name),
            case_number=case_number,
        )
        task_id = f"writer_{section_name}_r0"
        result = self.client.invoke(
            doc_id=doc_id,
            task_id=task_id,
            stage="writer",
            system_prompt=self.prompts.writer_system,
            user_prompt=user_prompt,
            parent_task_id=parent_task_id,
            temperature=self.writer_temperature,
            max_output_tokens=self.max_output_tokens,
            prompt_object=prompt_client,
        )
        writer_packet = parse_writer_packet(result.text)
        text = clean_generated_section_text(writer_packet.content)
        if not text:
            return "[section not generated]"
        metadata = dict(result.metadata)
        metadata.update(
            {
                "writer_json_valid": writer_packet.valid_json,
                "writer_json_parse_error": writer_packet.parse_error,
                "facts_used_count": len(writer_packet.facts_used),
                "legal_risks_count": len(writer_packet.legal_risks),
                "tone": writer_packet.tone,
            }
        )
        self._partial_store.write(
            doc_id=doc_id,
            section_name=section_name,
            revision_round=0,
            chunk_index=1,
            chunk_text=text,
            combined_text=text,
            task_id=task_id,
            metadata=metadata,
            complete=True,
            writer_packet_json=_writer_packet_json(writer_packet),
        )
        self._partial_store.flush()
        return text


def parse_writer_packet(raw_text: str) -> WriterPacket:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except (json.JSONDecodeError, ValueError) as exc:
        return WriterPacket(
            content=clean_generated_section_text(raw_text),
            facts_used=[],
            tone="formal neutral legal prose",
            legal_risks=["Writer did not return valid JSON."],
            raw_text=raw_text,
            valid_json=False,
            parse_error=str(exc),
        )

    content = payload.get("content")
    facts_used = payload.get("facts_used")
    tone = payload.get("tone")
    legal_risks = payload.get("legal_risks")
    return WriterPacket(
        content=clean_generated_section_text(content if isinstance(content, str) else ""),
        facts_used=_string_list(facts_used),
        tone=(
            tone.strip() if isinstance(tone, str) and tone.strip() else "formal neutral legal prose"
        ),
        legal_risks=_string_list(legal_risks),
        raw_text=raw_text,
        valid_json=True,
    )


def _writer_packet_json(packet: WriterPacket) -> str:
    payload = {
        "content": packet.content,
        "facts_used": packet.facts_used,
        "tone": packet.tone,
        "legal_risks": packet.legal_risks,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in writer response.")
    return text[start : end + 1]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
