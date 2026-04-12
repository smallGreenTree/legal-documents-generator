from dataclasses import dataclass


@dataclass(slots=True)
class CriticResult:
    approved: bool
    issues: list[str]
    revision_instruction: str
    raw_text: str

