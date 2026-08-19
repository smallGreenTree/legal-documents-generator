"""Create and inspect case document identifiers."""

from pathlib import Path


def doc_id_prefix(doc_type: str, fraud_type: str) -> str:
    return f"en_{doc_type}_{fraud_type}_"


def make_doc_id(doc_type: str, fraud_type: str, counter: int) -> str:
    return f"{doc_id_prefix(doc_type, fraud_type)}{counter:03d}"


def counter_from_doc_id(doc_id: str, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    if not isinstance(doc_id, str) or not doc_id.startswith(prefix):
        raise ValueError(f"Document ID must start with '{prefix}', got {doc_id!r}")

    suffix = doc_id[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"Document ID must end with digits, got {doc_id!r}")
    return int(suffix)


def next_counter(output_dir: Path, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    numbers = (
        [
            int(directory.name.removeprefix(prefix))
            for directory in output_dir.iterdir()
            if directory.is_dir()
            and directory.name.startswith(prefix)
            and directory.name.removeprefix(prefix).isdigit()
        ]
        if output_dir.exists()
        else []
    )
    return (max(numbers) + 1) if numbers else 1
