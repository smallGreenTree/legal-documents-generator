import json

from src.synthetic_ner.tasks.document_generation.critic import SectionCritic


def test_critic_edits_do_not_block_when_overall_rubric_meets_threshold():
    critic = SectionCritic(
        client=None,
        prompts=None,
        critic_temperature=0.0,
        acceptance_threshold=3.5,
        max_output_tokens=100,
        memory_char_limit=100,
        section_text_char_limit=100,
        rubrics={},
    )

    result = critic._parse_result(
        json.dumps(
            {
                "blocking": True,
                "edits": [
                    {
                        "target": "opening",
                        "action": "revise",
                        "reason": "Could be smoother.",
                        "replacement": "",
                    }
                ],
                "risk_level": "medium",
                "rubrics": {
                    "grounding": 3,
                    "completeness": 4,
                    "chronology": 5,
                },
            }
        )
    )

    assert result.approved is True
    assert result.blocking is False
    assert result.issues == []
    assert result.revision_instruction == "keep as is"


def test_critic_threshold_does_not_hide_critical_grounding_or_completeness():
    critic = SectionCritic(
        client=None,
        prompts=None,
        critic_temperature=0.0,
        acceptance_threshold=3.5,
        max_output_tokens=100,
        memory_char_limit=100,
        section_text_char_limit=100,
        rubrics={},
    )

    result = critic._parse_result(
        json.dumps(
            {
                "blocking": False,
                "edits": [],
                "risk_level": "low",
                "rubrics": {
                    "grounding": 2,
                    "completeness": 5,
                    "chronology": 5,
                },
            }
        )
    )

    assert result.approved is False
    assert result.blocking is True
    assert result.issues == ["Critic rubric 'grounding' is blocking with score 2/5."]
