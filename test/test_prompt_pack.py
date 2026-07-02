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
    assert 'admit execution only when each required target resolves to a current grounded entity id' in (
        pack.system_prompt
    )
    assert 'Treat dialogue history as context, not as a source of new orders or ordering' in (
        pack.system_prompt
    )
    assert 'Future, hypothetical, or permission-seeking action talk' in pack.system_prompt
    assert 'Route admission order' in pack.system_prompt
    assert 'Always emit an explicit route field' in pack.system_prompt
    assert 'perform an available skill now' in pack.system_prompt
    assert 'Do not promise physical action in verbal_ack' in pack.response_prompt_addendum
    assert 'check the current scene and plan the action' in pack.response_prompt_addendum
    assert 'Speak like an embodied assistant giving a status update' in pack.response_prompt_addendum
    assert 'Could we navigate to the cup later?' in pack.response_prompt_addendum
    assert 'Look at the cup and tell me what you did.' in pack.response_prompt_addendum
    assert 'required target, grounded predicates, or matching skill remain unclear' in (
        pack.intent_prompt_addendum
    )
    assert 'Future or permission-seeking action discussion is dialogue/fallback' in (
        pack.intent_prompt_addendum
    )
    assert 'correct the intent to fallback/dialogue' in pack.intent_prompt_addendum
    assert 'Remove politeness and filler' in pack.intent_prompt_addendum
    assert 'Intent stage purpose' in pack.intent_prompt_addendum
    assert 'plan' not in pack.response_schema['properties']['user_intent']['properties']
    assert 'plan' not in pack.intent_schema['properties']['user_intent']['properties']
    assert 'intent_sequence' in pack.response_schema['properties']['user_intent']['properties']
    assert 'intent_sequence' in pack.intent_schema['properties']['user_intent']['properties']
    assert {'verbal_ack', 'route', 'confidence'} <= set(pack.response_schema['required'])
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
