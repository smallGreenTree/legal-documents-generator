#!/usr/bin/env python
"""Smoke-test that a configured model can follow a drafting prompt contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_FIELDS = {"content", "facts_used", "tone", "legal_risks"}
WORD_RE = re.compile(r"\b[\w'/-]+\b")

SMOKE_SYSTEM_PROMPT = """
You are a precise legal data synthesis specialist creating synthetic evidence
sections for criminal NER/anonymization test datasets.

Return exactly one valid JSON object and nothing else. The object must have
exactly these fields: content, facts_used, tone, legal_risks.
""".strip()

SMOKE_USER_PROMPT = """
SECTION_CONTEXT:
Section: evidence
Purpose: Write a natural evidence section about a synthetic money laundering
scenario. Use only the facts listed here.

Case reference: CPS/2026/1442
Cross reference: C/2025/3188
Charged period: 3 February 2025 to 19 September 2025
Defendants: Amelia Cross and Luca Rinaldi
Charged organisations: ORION MATERIALS LTD and VEGA SUPPLIES LTD
Associated organisation: HELIOS REFUNDS LTD
Total alleged loss: GBP 187,420
Inflated invoice value: GBP 62,500
Transfer amount: GBP 74,300 from ORION MATERIALS LTD to HELIOS REFUNDS LTD
Transfer amount: GBP 113,120 from VEGA SUPPLIES LTD to HELIOS REFUNDS LTD

Evidence categories:
- Invoice and purchase-order records concern material orders placed between
  the charged organisations.
- Delivery and warehouse records concern the absence of the materials said to
  have been purchased.
- Civil claim and refund records concern a non-delivery claim and the return
  of funds through the charged organisations.
- Payment records concern funds moved between the charged organisations and
  HELIOS REFUNDS LTD during the charged period.

SECTION_CONTRACT:
Write connected English legal prose only. The content field must describe the
evidence categories and what each category tends to prove. Use the case
references, dates, organisations, transfer amounts, and total alleged loss only
where they help explain relevance. Do not invent exhibit labels, emails,
witnesses, bank account numbers, invoice numbers, VAT numbers, full addresses,
or unlisted entities. Do not include headings, bullets, markdown, or
meta-commentary inside content.

Output requirements:
- content: 180-260 words, connected paragraphs only
- facts_used: exact or close facts from SECTION_CONTEXT that you actually used
- tone: short description of the tone
- legal_risks: drafting risks avoided
""".strip()

REQUIRED_CONTENT_FACTS = (
    "CPS/2026/1442",
    "C/2025/3188",
    "ORION MATERIALS LTD",
    "VEGA SUPPLIES LTD",
    "HELIOS REFUNDS LTD",
    "GBP 187,420",
    "GBP 74,300",
    "GBP 113,120",
)

REQUIRED_TERM_GROUPS = (
    ("invoice and purchase-order records", ("invoice", "purchase")),
    ("delivery and warehouse records", ("delivery", "warehouse")),
    ("civil claim and refund records", ("civil", "refund")),
    ("payment records", ("payment",)),
)

FORBIDDEN_CONTENT_TERMS = (
    "as an ai",
    "bank account",
    "compliance",
    "contract",
    "email",
    "exhibit",
    "i cannot",
    "invoice number",
    "section_context",
    "section_contract",
    "the content field",
    "this response",
    "vat",
    "witness",
)

ENGLISH_MARKERS = (
    "the",
    "and",
    "of",
    "to",
    "records",
    "evidence",
    "funds",
    "during",
    "alleged",
    "organisations",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a strict drafting prompt.")
    parser.add_argument("--config", default="config.yaml", help="Root config path.")
    parser.add_argument(
        "--case-config",
        default="config_case/case_1.yaml",
        help="Case config path used to load the full app config.",
    )
    parser.add_argument(
        "--stage",
        default="writer",
        choices=("planner", "writer", "critic"),
        help="Configured model route to test. Defaults to writer.",
    )
    parser.add_argument("--min-words", type=int, default=180)
    parser.add_argument("--max-words", type=int, default=260)
    parser.add_argument("--max-output-tokens", type=int, default=850)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Print the model response when validation fails.",
    )
    return parser


def main() -> int:
    from src.synthetic_ner.config import load_app_config

    args = build_parser().parse_args()
    project_root = Path.cwd()
    app_config = load_app_config(
        _resolve_path(project_root, args.config),
        _resolve_path(project_root, args.case_config),
    )
    provider = app_config.model_routing.stages[args.stage]
    if provider.provider != "ollama":
        print(f"FAIL {args.stage}: unsupported provider {provider.provider!r}", file=sys.stderr)
        return 1

    result = _generate(provider, args)
    if not result["ok"]:
        print(f"FAIL {args.stage}: {result['error']}", file=sys.stderr)
        return 1

    text = str(result["response"]).strip()
    failures = _validate_response(text, min_words=args.min_words, max_words=args.max_words)
    if failures:
        print("Prompt-contract smoke test failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        if args.show_response:
            print("\nModel response:\n" + text, file=sys.stderr)
        return 1

    payload = json.loads(text)
    words = _word_count(payload["content"])
    print(
        f"ok prompt-contract: stage={args.stage} model={provider.model} "
        f"words={words} tokens={result.get('tokens_response') or 'n/a'}"
    )
    return 0


def _generate(provider: Any, args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": args.temperature,
        "num_predict": args.max_output_tokens,
    }
    if provider.num_ctx is not None:
        options["num_ctx"] = provider.num_ctx
    if provider.top_p is not None:
        options["top_p"] = provider.top_p

    request_json: dict[str, Any] = {
        "model": provider.model,
        "prompt": f"[SYSTEM]\n{SMOKE_SYSTEM_PROMPT}\n\n[USER]\n{SMOKE_USER_PROMPT}\n",
        "stream": False,
        "options": options,
    }
    if provider.think is not None:
        request_json["think"] = provider.think

    try:
        response = requests.post(
            f"{provider.base_url.rstrip('/')}/api/generate",
            json=request_json,
            timeout=args.timeout or provider.timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}

    payload = response.json()
    text = str(payload.get("response") or "").strip()
    if not text:
        return {"ok": False, "error": "empty response"}
    return {
        "ok": True,
        "response": text,
        "tokens_response": payload.get("eval_count"),
    }


def _validate_response(text: str, *, min_words: int, max_words: int) -> list[str]:
    failures: list[str] = []
    if not text.startswith("{") or not text.endswith("}"):
        failures.append("response must be exactly one JSON object with no wrapper text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return failures + [f"response is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return failures + ["response JSON must be an object"]

    failures.extend(_validate_payload_shape(payload))
    content = payload.get("content")
    if isinstance(content, str):
        failures.extend(_validate_content(content, min_words=min_words, max_words=max_words))
    return failures


def _validate_payload_shape(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if set(payload) != REQUIRED_FIELDS:
        failures.append(f"JSON fields must be exactly {sorted(REQUIRED_FIELDS)}")
    if not isinstance(payload.get("content"), str) or not payload.get("content", "").strip():
        failures.append("content must be a non-empty string")
    if not _is_string_list(payload.get("facts_used"), minimum=6):
        failures.append("facts_used must contain at least six strings")
    if not isinstance(payload.get("tone"), str) or not payload.get("tone", "").strip():
        failures.append("tone must be a non-empty string")
    if not _is_string_list(payload.get("legal_risks"), minimum=2):
        failures.append("legal_risks must contain at least two strings")
    return failures


def _validate_content(content: str, *, min_words: int, max_words: int) -> list[str]:
    failures: list[str] = []
    lower = content.lower()
    words = _word_count(content)
    if words < min_words or words > max_words:
        failures.append(f"content word count {words} is outside {min_words}-{max_words}")
    if len(re.findall(r"[.!?](?:\s|$)", content)) < 4:
        failures.append("content must contain at least four complete sentences")
    failures.extend(_missing_required_content(lower))
    failures.extend(_formatting_failures(content))
    if sum(1 for marker in ENGLISH_MARKERS if marker in lower) < 7:
        failures.append("content does not look like connected English legal prose")
    return failures


def _missing_required_content(lower_content: str) -> list[str]:
    failures: list[str] = []
    for fact in REQUIRED_CONTENT_FACTS:
        if fact.lower() not in lower_content:
            failures.append(f"content is missing required fact: {fact}")
    for label, terms in REQUIRED_TERM_GROUPS:
        if not all(term in lower_content for term in terms):
            failures.append(f"content is missing {label}")
    for term in FORBIDDEN_CONTENT_TERMS:
        if term in lower_content:
            failures.append(f"content contains forbidden term: {term}")
    return failures


def _formatting_failures(content: str) -> list[str]:
    failures: list[str] = []
    if re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", content):
        failures.append("content must not contain bullets or numbered lists")
    if "```" in content or "\n#" in content:
        failures.append("content must not contain markdown")
    if content.lstrip().upper().startswith(("SECTION", "EVIDENCE")):
        failures.append("content must start with substantive prose, not a heading")
    return failures


def _word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def _is_string_list(value: Any, *, minimum: int) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


if __name__ == "__main__":
    raise SystemExit(main())
