import ast
from pathlib import Path

from src.synthetic_ner.types.document_generation import (
    AllowedFacts,
    GenerationComponents,
    SectionWorkflowResult,
    WriterPacket,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOCUMENT_GENERATION_ROOT = PROJECT_ROOT / "src" / "synthetic_ner" / "tasks" / "document_generation"


def test_document_generation_dataclasses_live_in_types():
    declarations = []
    for path in DOCUMENT_GENERATION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _has_dataclass_decorator(node):
                declarations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name}")

    assert declarations == []
    assert {
        model.__module__
        for model in (AllowedFacts, GenerationComponents, SectionWorkflowResult, WriterPacket)
    } == {"src.synthetic_ner.types.document_generation"}


def test_document_generation_constants_are_centralized():
    declarations = []
    for path in DOCUMENT_GENERATION_ROOT.rglob("*.py"):
        if path.name == "constants.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            for name in _assigned_names(node):
                if name.isupper() or (name.startswith("_") and name[1:].isupper()):
                    declarations.append(f"{path.relative_to(PROJECT_ROOT)}:{name}")

    assert declarations == []


def _has_dataclass_decorator(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        candidate = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(candidate, ast.Name) and candidate.id == "dataclass":
            return True
    return False


def _assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []
