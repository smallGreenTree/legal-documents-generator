"""Section writer for the LangGraph workflow."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from src.synthetic_ner.constants import SECTION_DESCRIPTIONS
from src.synthetic_ner.tasks.prompt_context import (
    build_section_context,
    build_section_contract,
)
from src.synthetic_ner.tasks.validators import clean_generated_section_text
from src.synthetic_ner.types.app_config import WorkflowPromptsConfig
from src.synthetic_ner.utils import render_prompt_template


@dataclass(slots=True)
class WriterPacket:
    content: str
    facts_used: list[str]
    tone: str
    legal_risks: list[str]
    raw_text: str
    valid_json: bool
    parse_error: str | None = None


class SectionWriter:
    def __init__(
        self,
        *,
        client,
        prompts: WorkflowPromptsConfig,
        chunk_words: int,
        context_tail_chars: int,
        writer_temperature: float,
        max_output_tokens: int,
        min_output_tokens: int,
        output_token_multiplier: float,
        prompt_clients: dict[str, Any] | None = None,
        partial_output_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.chunk_words = chunk_words
        self.context_tail_chars = context_tail_chars
        self.writer_temperature = writer_temperature
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        self.output_token_multiplier = output_token_multiplier
        self.prompt_clients = prompt_clients or {}
        self.partial_output_dir = partial_output_dir
        self._partial_writer = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="partial-section-writer",
        )
        self._partial_write_futures: list[Future] = []
        self._partial_write_lock = Lock()

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
        temperature: float | None = None,
    ) -> str:
        chunks = []
        words_so_far = 0
        chunk_index = 1
        effective_temperature = (
            self.writer_temperature if temperature is None else temperature
        )

        while words_so_far < word_target:
            remaining = word_target - words_so_far
            chunk_target = min(self.chunk_words, remaining)
            previous_tail = chunks[-1][-self.context_tail_chars:] if chunks else "n/a"
            prompt_client = self.prompt_clients.get("writer_user")
            section_contract = build_section_contract(section_name)
            user_prompt = render_prompt_template(
                self.prompts.writer_user,
                prompt_client=prompt_client,
                memory_text=memory_text,
                section_context=build_section_context(memory_text, section_name),
                section_contract=section_contract,
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
                temperature=effective_temperature,
                max_output_tokens=_estimate_writer_output_tokens(
                    chunk_target,
                    max_output_tokens=self.max_output_tokens,
                    min_output_tokens=self.min_output_tokens,
                    output_token_multiplier=self.output_token_multiplier,
                ),
                prompt_object=prompt_client,
            )
            writer_packet = parse_writer_packet(result.text)
            polisher_prompt_client = self.prompt_clients.get("polisher_user")
            polisher_prompt = render_prompt_template(
                self.prompts.polisher_user,
                prompt_client=polisher_prompt_client,
                section_context=build_section_context(memory_text, section_name),
                section_contract=section_contract,
                writer_json=_writer_packet_json(writer_packet),
                section_name=section_name,
            )
            polished_result = self.client.invoke(
                doc_id=doc_id,
                task_id=f"polish_{section_name}_r{revision_round}_chunk_{chunk_index:02d}",
                stage="polisher",
                system_prompt=self.prompts.polisher_system,
                user_prompt=polisher_prompt,
                parent_task_id=task_id,
                temperature=0.0,
                max_output_tokens=_estimate_writer_output_tokens(
                    chunk_target,
                    max_output_tokens=self.max_output_tokens,
                    min_output_tokens=self.min_output_tokens,
                    output_token_multiplier=self.output_token_multiplier,
                ),
                prompt_object=polisher_prompt_client,
            )
            text = clean_generated_section_text(polished_result.text)
            if not text:
                text = clean_generated_section_text(writer_packet.content)
            if not text:
                break
            chunks.append(text)
            metadata = dict(polished_result.metadata)
            metadata.update(
                {
                    "writer_json_valid": writer_packet.valid_json,
                    "writer_json_parse_error": writer_packet.parse_error,
                    "facts_used_count": len(writer_packet.facts_used),
                    "legal_risks_count": len(writer_packet.legal_risks),
                    "tone": writer_packet.tone,
                }
            )
            self._write_partial_section(
                doc_id=doc_id,
                section_name=section_name,
                revision_round=revision_round,
                chunk_index=chunk_index,
                chunk_text=text,
                combined_text=clean_generated_section_text("\n\n".join(chunks)),
                task_id=f"polish_{section_name}_r{revision_round}_chunk_{chunk_index:02d}",
                metadata=metadata,
                complete=True,
                writer_packet=writer_packet,
            )
            words_so_far += len(text.split())
            chunk_index += 1

        self._flush_partial_writes()
        if not chunks:
            return "[section not generated]"
        return clean_generated_section_text("\n\n".join(chunks))

    def _write_partial_section(
        self,
        *,
        doc_id: str,
        section_name: str,
        revision_round: int,
        chunk_index: int,
        chunk_text: str,
        combined_text: str,
        task_id: str,
        metadata: dict[str, Any],
        complete: bool,
        writer_packet: WriterPacket | None = None,
    ) -> None:
        if self.partial_output_dir is None:
            return

        with self._partial_write_lock:
            self._partial_write_futures.append(
                self._partial_writer.submit(
                    self._write_partial_section_sync,
                    doc_id=doc_id,
                    section_name=section_name,
                    revision_round=revision_round,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    combined_text=combined_text,
                    task_id=task_id,
                    metadata=metadata,
                    complete=complete,
                    writer_packet=writer_packet,
                )
            )

    def _write_partial_section_sync(
        self,
        *,
        doc_id: str,
        section_name: str,
        revision_round: int,
        chunk_index: int,
        chunk_text: str,
        combined_text: str,
        task_id: str,
        metadata: dict[str, Any],
        complete: bool,
        writer_packet: WriterPacket | None = None,
    ) -> None:
        if self.partial_output_dir is None:
            return

        revision_dir = (
            self.partial_output_dir
            / doc_id
            / "sections"
            / section_name
            / f"r{revision_round}"
        )
        revision_dir.mkdir(parents=True, exist_ok=True)
        (revision_dir / f"chunk_{chunk_index:02d}.txt").write_text(
            chunk_text.rstrip() + "\n",
            encoding="utf-8",
        )
        (revision_dir / "combined.txt").write_text(
            combined_text.rstrip() + "\n",
            encoding="utf-8",
        )
        manifest = {
            "doc_id": doc_id,
            "section_name": section_name,
            "revision_round": revision_round,
            "latest_chunk_index": chunk_index,
            "latest_task_id": task_id,
            "word_count": len(combined_text.split()),
            "complete": complete,
            "metadata": {
                "model": metadata.get("model"),
                "streaming": metadata.get("streaming", False),
                "latency_ms": metadata.get("latency_ms"),
                "tokens_prompt": metadata.get("tokens_prompt"),
                "tokens_response": metadata.get("tokens_response"),
                "done_reason": metadata.get("done_reason"),
                "output_budget": metadata.get("output_budget"),
            },
        }
        (revision_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if writer_packet is not None:
            (revision_dir / f"writer_packet_{chunk_index:02d}.json").write_text(
                _writer_packet_json(writer_packet) + "\n",
                encoding="utf-8",
            )

    def _flush_partial_writes(self) -> None:
        with self._partial_write_lock:
            futures, self._partial_write_futures = self._partial_write_futures, []
        for future in futures:
            future.result()


def _estimate_writer_output_tokens(
    chunk_words: int,
    *,
    max_output_tokens: int,
    min_output_tokens: int,
    output_token_multiplier: float,
) -> int:
    estimated = int(chunk_words * output_token_multiplier)
    return max(min_output_tokens, min(max_output_tokens, estimated))


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
            tone.strip()
            if isinstance(tone, str) and tone.strip()
            else "formal neutral legal prose"
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
    return text[start:end + 1]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
