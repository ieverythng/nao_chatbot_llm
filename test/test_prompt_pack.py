from chatbot_llm.prompt_pack import default_prompt_pack
from chatbot_llm.prompt_pack import load_prompt_pack


def test_default_prompt_pack_execution_contract_stays_plan_free():
    pack = default_prompt_pack()

    assert 'top-level plan field' in pack.response_prompt_addendum
    assert 'user_intent.plan' in pack.response_prompt_addendum
    assert 'top-level plan field' in pack.intent_prompt_addendum
    assert 'user_intent.plan' in pack.intent_prompt_addendum
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
