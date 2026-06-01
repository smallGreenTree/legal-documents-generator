from src.synthetic_ner.utils import render_prompt_template


class PromptClient:
    def compile(self, **context):
        del context
        return "compiled by external prompt client"


def test_render_prompt_template_evaluates_jinja_blocks_with_prompt_client():
    template = """Hello {{ name }}.
{% if revision_instruction and revision_instruction != "none" -%}
REVISION:
{{ revision_instruction }}
{% endif -%}
Done.
"""

    rendered = render_prompt_template(
        template,
        prompt_client=PromptClient(),
        name="writer",
        revision_instruction="none",
    )

    assert "Hello writer." in rendered
    assert "REVISION:" not in rendered
    assert "{% if" not in rendered
    assert "compiled by external prompt client" not in rendered
