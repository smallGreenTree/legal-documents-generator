from dataclasses import dataclass


@dataclass(slots=True)
class CriticResult:
    approved: bool
    issues: list[str]
    revision_instruction: str
    rubrics: dict[str, int]
    raw_text: str
