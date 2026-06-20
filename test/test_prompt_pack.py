import pytest

from chatbot_llm.prompt_pack import default_prompt_pack
from chatbot_llm.prompt_pack import load_prompt_pack


def test_default_prompt_pack_loads_canonical_yaml():
    pack = default_prompt_pack()

    assert 'friendly robot called' in pack.system_prompt
    assert 'You do not generate' in pack.system_prompt
    assert 'grounded_context' in pack.response_prompt_addendum
    assert 'Do not acknowledge that execution will begin until the request is admitted' in (
        pack.system_prompt
    )
    assert 'required target, grounded predicates, or matching skill remain unclear' in (
        pack.intent_prompt_addendum
    )
    assert 'Intent stage purpose' in pack.intent_prompt_addendum
    assert 'one atomic Intent-Route-Response decision' in pack.irr_prompt_addendum
    assert 'planner_handoff' in pack.irr_schema['properties']
    assert 'speak_now' not in pack.irr_schema['properties']['response']['properties']
    assert 'requested' in pack.irr_schema['properties']['planner_handoff']['properties']
    assert 'plan' not in pack.response_schema['properties']['user_intent']['properties']
    assert 'plan' not in pack.intent_schema['properties']['user_intent']['properties']
    assert 'intent_sequence' in pack.response_schema['properties']['user_intent']['properties']
    assert 'intent_sequence' in pack.intent_schema['properties']['user_intent']['properties']
    assert 'scene_targets' in pack.intent_schema['properties']['user_intent']['properties']
    assert 'ack_text' in pack.intent_schema['properties']['user_intent']['properties']
    assert 'coordination_markers' in pack.planner_multi_step_heuristics
    assert 'action_hint_tokens' in pack.planner_multi_step_heuristics


def test_prompt_pack_loads_planner_multi_step_heuristics(tmp_path):
    prompt_pack_path = tmp_path / 'pack.yaml'
    prompt_pack_path.write_text(
        '\n'.join(
            [
                'system_prompt: "canonical chatbot"',
                'response_prompt_addendum: "respond"',
                'intent_prompt_addendum: "extract"',
                'planner_multi_step_heuristics:',
                '  coordination_markers:',
                '    - " despues "',
                '  action_hint_tokens:',
                '    - levantar',
            ]
        ),
        encoding='utf-8',
    )

    pack = load_prompt_pack(str(prompt_pack_path))

    assert pack.planner_multi_step_heuristics == {
        'coordination_markers': [' despues '],
        'action_hint_tokens': ['levantar'],
    }


def test_prompt_pack_empty_path_loads_repo_yaml_defaults():
    pack = load_prompt_pack('')

    assert ' and then ' in pack.planner_multi_step_heuristics['coordination_markers']
    assert 'stand' in pack.planner_multi_step_heuristics['action_hint_tokens']


def test_prompt_pack_missing_required_prompt_text_fails(tmp_path):
    prompt_pack_path = tmp_path / 'pack.yaml'
    prompt_pack_path.write_text('system_prompt: ""\n', encoding='utf-8')

    with pytest.raises(ValueError, match='system_prompt'):
        load_prompt_pack(str(prompt_pack_path))


def test_prompt_pack_invalid_yaml_fails(tmp_path):
    prompt_pack_path = tmp_path / 'pack.yaml'
    prompt_pack_path.write_text('::bad', encoding='utf-8')

    with pytest.raises(ValueError, match='root must be a mapping'):
        load_prompt_pack(str(prompt_pack_path))
