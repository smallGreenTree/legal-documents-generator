from dataclasses import dataclass


@dataclass(slots=True)
class CriticEdit:
    target: str
    action: str
    reason: str
    replacement: str


@dataclass(slots=True)
class CriticResult:
    approved: bool
    issues: list[str]
    revision_instruction: str
    rubrics: dict[str, int]
    raw_text: str
    edits: list[CriticEdit]
    blocking: bool
    risk_level: str
