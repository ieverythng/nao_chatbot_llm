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


def make_config(
    intent_mode: str = 'rules',
    *,
    planner_mode_enabled: bool = False,
) -> ChatbotConfig:
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
        think=False,
        response_max_tokens=64,
        intent_max_tokens=64,
        fallback_response='fallback',
        max_history_messages=20,
        scene_memory_turns=4,
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
        planner_mode_enabled=planner_mode_enabled,
        planner_request_topic='/planner/request',
        planner_request_intent='planner_request',
        planner_scene_summary_topic='/scene/summary',
        planner_world_model_snapshot_topic='/world_model/enriched_snapshot',
        planner_world_model_text_topic='/world_model/enriched_text',
        knowledge_enabled=False,
        knowledge_query_service_name='/kb/query',
        knowledge_query_timeout_sec=0.5,
        knowledge_default_query_groups=[
            'myself sees ?entity && ?entity rdf:type ?type',
        ],
        knowledge_default_patterns=['myself sees ?entity', '?entity rdf:type ?type'],
        knowledge_default_vars=['?entity', '?type'],
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
    assert result.route == 'execution'
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
    assert transport.calls[0]['think'] is False
    assert transport.calls[1]['think'] is False
    assert transport.calls[0]['max_tokens'] == 64
    assert transport.calls[1]['max_tokens'] == 64
    assert result.user_intent == {'type': 'head_look_left'}
    assert result.route == 'execution'
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
    assert 'Live symbolic scene state from KnowledgeCore for this turn:' in transport.calls[0]['messages'][0]['content']
    assert 'Knowledge snapshot:\nmug isOn table' in transport.calls[0]['messages'][0]['content']
    assert 'Live symbolic scene state from KnowledgeCore for this turn:' in transport.calls[1]['messages'][0]['content']
    assert 'Knowledge snapshot:\nmug isOn table' in transport.calls[1]['messages'][0]['content']


def test_turn_engine_prompt_explicitly_mentions_recent_history():
    transport = FakeTransport(
        [
            '{"verbal_ack":"Yes, I can see a person."}',
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
        user_text='is that the same person as before?',
        history=[
            'user:can you see anyone?',
            'assistant:Yes, I can see a person.',
            'user:what can you see besides the person?',
            'assistant:I cannot confirm any object yet.',
        ],
        user_id='user1',
        knowledge_snapshot='Entities currently seen by the robot: anonymous person dhgef (Human)',
    )

    assert result.success is True
    assert (
        'Recent conversation history is included in the messages above.'
        in transport.calls[0]['messages'][0]['content']
    )
    assert 'kb_query_visible_people' in transport.calls[1]['messages'][0]['content']
    assert transport.calls[0]['messages'][1]['content'] == 'can you see anyone?'
    assert transport.calls[0]['messages'][2]['content'] == 'Yes, I can see a person.'


def test_turn_engine_planner_mode_uses_single_response_stage_for_execution():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I will look left and then sit down.",'
                '"route":"execution",'
                '"user_intent":{"type":"fallback","ack_mode":"say",'
                '"goal":"look left and then sit down"},'
                '"confidence":0.72}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='look left and then sit down',
        history=[],
        user_id='user1',
        knowledge_snapshot='person_1 rdf:type Person',
    )

    assert result.success is True
    assert len(transport.calls) == 1
    assert result.route == 'execution'
    assert result.intent == 'fallback'
    assert result.intent_source == 'llm_response_route'
    assert result.intent_confidence == 0.72
    assert result.user_intent['type'] == 'fallback'
    assert result.user_intent['goal'] == 'look left and then sit down'
    assert 'plan' not in result.user_intent
    assert (
        'Planner-mode routing requirements:'
        in transport.calls[0]['messages'][0]['content']
    )
    assert 'planner_llm owns all' in transport.calls[0]['messages'][0]['content']
    assert 'Knowledge snapshot:\nperson_1 rdf:type Person' in transport.calls[0]['messages'][0]['content']


def test_turn_engine_planner_mode_infers_execution_route_without_second_call():
    transport = FakeTransport(
        [
            '{"verbal_ack":"I can do that.","user_intent":{"type":"posture_stand"},"confidence":0.51}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='please stand up',
        history=[],
        user_id='user1',
    )

    assert len(transport.calls) == 1
    assert result.route == 'execution'
    assert result.intent == 'posture_stand'
    assert result.intent_source == 'llm_response_inferred_route'


def test_turn_engine_planner_mode_does_not_use_rules_when_llm_response_fails():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='can you look around and tell me what you see',
        history=[],
        user_id='user1',
    )

    assert result.success is False
    assert result.verbal_ack == 'fallback'
    assert result.intent_source == 'llm_response_failed'
    assert result.route == 'dialogue'


def test_turn_engine_forwards_think_flag_to_transport():
    transport = FakeTransport(
        ['{"verbal_ack":"Sure.","user_intent":{"type":"help"},"confidence":0.4}']
    )
    config = make_config(intent_mode='llm', planner_mode_enabled=True)
    config = ChatbotConfig(**{**config.__dict__, 'think': True})
    engine = DialogueTurnEngine(
        config=config,
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    engine.execute_turn(user_text='help me', history=[], user_id='user1')

    assert transport.calls[0]['think'] is True


def test_turn_engine_forwards_response_and_intent_token_caps_to_transport():
    transport = FakeTransport(
        [
            '{"verbal_ack":"Sure."}',
            '{"user_intent":{"type":"help"},"intent_confidence":0.4}',
        ]
    )
    config = make_config(intent_mode='llm')
    config = ChatbotConfig(
        **{
            **config.__dict__,
            'response_max_tokens': 32,
            'intent_max_tokens': 48,
        }
    )
    engine = DialogueTurnEngine(
        config=config,
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    engine.execute_turn(user_text='help me', history=[], user_id='user1')

    assert transport.calls[0]['max_tokens'] == 32
    assert transport.calls[1]['max_tokens'] == 48


def test_turn_engine_does_not_speak_json_encoded_ack_payload():
    transport = FakeTransport(
        [
            '"{\\"verbal_ack\\": \\"Moving my head down now.\\", '
            '\\"user_intent\\": {\\"type\\": \\"head_look_down\\"}}"',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='move your head down',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == 'Moving my head down now.'


def test_turn_engine_extracts_ack_from_loose_json_like_text():
    transport = FakeTransport(
        [
            '```json\n{\n  "verbal_ack": "Sure, I am tilting my head down.",\n'
            '  "route": "execution",\n  "user_intent": {"type": "move_head"}\n}\n```',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='move your head down',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == 'Sure, I am tilting my head down.'


def test_turn_engine_extracts_ack_from_wrapped_response_field():
    transport = FakeTransport(
        [
            (
                '{"response":"{\\"verbal_ack\\":\\"Standing up now.\\",'
                '\\"route\\":\\"execution\\",'
                '\\"user_intent\\":{\\"type\\":\\"posture_change\\"}}"}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='stand up',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == 'Standing up now.'


def test_turn_engine_ignores_plan_fields_from_response_payloads():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Standing up now.",'
                '"route":"execution",'
                '"plan":[{"type":"skill","name":"perform_motion"}],'
                '"user_intent":{"type":"posture_stand",'
                '"plan":[{"type":"skill","name":"perform_motion"}]}}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='stand up',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == 'Standing up now.'
    assert result.user_intent == {'type': 'posture_stand'}
    assert result.updated_history == ['user:stand up', 'assistant:Standing up now.']


def test_turn_engine_falls_back_when_json_has_no_safe_ack():
    transport = FakeTransport(
        [
            '{"route":"execution","user_intent":{"type":"posture_stand"}}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='stand up',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == 'fallback'
    assert result.updated_history == ['user:stand up', 'assistant:fallback']
