from chatbot_llm.prompt_pack import default_prompt_pack


def test_default_prompt_pack_execution_contract_stays_plan_free():
    pack = default_prompt_pack()

    assert 'top-level plan field' in pack.response_prompt_addendum
    assert 'user_intent.plan' in pack.response_prompt_addendum
    assert 'top-level plan field' in pack.intent_prompt_addendum
    assert 'user_intent.plan' in pack.intent_prompt_addendum
    assert 'plan' not in pack.response_schema['properties']['user_intent']['properties']
    assert 'plan' not in pack.intent_schema['properties']['user_intent']['properties']
    assert 'scene_targets' in pack.intent_schema['properties']['user_intent']['properties']
    assert 'ack_text' in pack.intent_schema['properties']['user_intent']['properties']
