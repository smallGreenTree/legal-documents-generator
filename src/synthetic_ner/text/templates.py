"""Render inline and workflow prompt templates."""

from typing import Any

from src.synthetic_ner.core.constants import INLINE_TEMPLATE_ENV


def render_inline_template(template: str, **context) -> str:
    return INLINE_TEMPLATE_ENV.from_string(template).render(**context)


def render_prompt_template(
    template: str,
    *,
    prompt_client: Any | None = None,
    **context,
) -> str:
    del prompt_client
    return render_inline_template(template, **context)
