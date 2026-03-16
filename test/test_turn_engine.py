from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.turn_engine import DialogueTurnEngine


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return ''
        return self._responses.pop(0)


def make_config(intent_mode: str = 'rules') -> ChatbotConfig:
    return ChatbotConfig(
        server_url='http://localhost:11434/api/chat',
        model='llama3.2:1b',
        api_key='',
        system_prompt='You are {robot_name}.',
        enabled=True,
        intent_model='llama3.2:1b',
        request_timeout_sec=20.0,
        first_request_timeout_sec=60.0,
        intent_request_timeout_sec=10.0,
        context_window_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        fallback_response='fallback',
        max_history_messages=12,
        robot_name='NAO',
        persona_prompt_path='',
        response_prompt_addendum='Respond briefly.',
        intent_prompt_addendum='Infer intent.',
        environment_description='No specific objects described.',
        response_schema={'type': 'object'},
        intent_schema={'type': 'object'},
        identity_reminder_every_n_turns=6,
        intent_detection_mode=intent_mode,
        prompt_pack_path='',
        use_skill_catalog=False,
        skill_catalog_packages=[],
        skill_catalog_max_entries=0,
        skill_catalog_max_chars=0,
        knowledge_enabled=False,
        knowledge_query_service_name='/kb/query',
        knowledge_query_timeout_sec=0.5,
        knowledge_default_patterns=['?s ?p ?o'],
        knowledge_default_vars=['?s', '?p', '?o'],
        knowledge_default_models=[],
        knowledge_max_results=40,
        knowledge_max_chars=3000,
    )


def test_turn_engine_rules_mode_generates_motion_reply():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport([]),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='please stand up',
        history=[],
        user_id='user1',
    )

    assert result.success is True
    assert result.intent == 'posture_stand'
    assert result.intent_source == 'rules'
    assert result.intent_confidence == 1.0
    assert result.verbal_ack == 'Sure. I am switching to a standing posture.'
    assert result.updated_history == [
        'user:please stand up',
        'assistant:Sure. I am switching to a standing posture.',
    ]


def test_turn_engine_llm_mode_uses_two_stage_json_outputs():
    transport = FakeTransport(
        [
            '{"verbal_ack":"Sure. I am turning my head to the left."}',
            '{"user_intent":{"type":"head_look_left"},"intent_confidence":0.85}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback'),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='look left',
        history=['assistant:How can I help you?'],
        user_id='user1',
    )

    assert result.success is True
    assert result.intent == 'head_look_left'
    assert result.intent_source == 'llm_intent'
    assert result.intent_confidence == 0.85
    assert result.user_intent == {'type': 'head_look_left'}
    assert result.verbal_ack == 'Sure. I am turning my head to the left.'
    assert result.updated_history[-2:] == [
        'user:look left',
        'assistant:Sure. I am turning my head to the left.',
    ]


def test_turn_engine_includes_knowledge_snapshot_in_both_llm_stages():
    transport = FakeTransport(
        [
            '{"verbal_ack":"The mug is on the table."}',
            '{"user_intent":{"type":"help"},"intent_confidence":0.4}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm'),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='where is the mug',
        history=[],
        user_id='user1',
        knowledge_snapshot='mug isOn table',
    )

    assert result.success is True
    assert len(transport.calls) == 2
    assert 'Knowledge base snapshot:\nmug isOn table' in transport.calls[0]['messages'][0]['content']
    assert 'Knowledge base snapshot:\nmug isOn table' in transport.calls[1]['messages'][0]['content']
