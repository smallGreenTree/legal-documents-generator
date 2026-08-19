"""Shared persistence for recoverable section-generation outputs."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any


class PartialSectionStore:
    def __init__(self, root: Path | None, *, thread_name_prefix: str) -> None:
        self.root = root
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._futures: list[Future] = []
        self._lock = Lock()

    def write(
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
        writer_packet_json: str | None = None,
    ) -> None:
        if self.root is None:
            return

        with self._lock:
            self._futures.append(
                self._executor.submit(
                    self._write_sync,
                    doc_id=doc_id,
                    section_name=section_name,
                    revision_round=revision_round,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    combined_text=combined_text,
                    task_id=task_id,
                    metadata=metadata,
                    complete=complete,
                    writer_packet_json=writer_packet_json,
                )
            )

    def flush(self) -> None:
        with self._lock:
            futures, self._futures = self._futures, []
        for future in futures:
            future.result()

    def _write_sync(
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
        writer_packet_json: str | None,
    ) -> None:
        if self.root is None:
            return

        revision_dir = self.root / doc_id / "sections" / section_name / f"r{revision_round}"
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
        if writer_packet_json is not None:
            (revision_dir / f"writer_packet_{chunk_index:02d}.json").write_text(
                writer_packet_json + "\n",
                encoding="utf-8",
            )
