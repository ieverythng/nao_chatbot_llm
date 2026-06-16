from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.turn_engine import DialogueTurnEngine
import pytest


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
        preflight_enabled=True,
        preflight_required=False,
        preflight_timeout_sec=45.0,
        preflight_attempts=1,
        preflight_realistic_enabled=False,
        preflight_keepalive_interval_sec=0.0,
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
        planner_multi_step_heuristics={
            'coordination_markers': [' and then ', ' then '],
            'action_hint_tokens': ['stand', 'sit', 'look', 'move', 'head'],
        },
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
        grounded_context_include_state_t0=False,
        turn_trace_enabled=True,
        turn_trace_topic='/chatbot_llm/turn_trace',
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


def test_turn_engine_includes_grounded_context_in_both_llm_stages():
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
        knowledge_snapshot='Grounded context JSON:\n```json\n{"entities":[]}\n```',
    )

    assert result.success is True
    assert len(transport.calls) == 2
    assert 'Grounded context for this turn:' in transport.calls[0]['messages'][0]['content']
    assert 'Grounded context:\nGrounded context JSON:' in transport.calls[0]['messages'][0]['content']
    assert 'Grounded context for this turn:' in transport.calls[1]['messages'][0]['content']
    assert 'Grounded context:\nGrounded context JSON:' in transport.calls[1]['messages'][0]['content']


def test_turn_engine_prompt_exposes_updated_grounded_relation_for_followups():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Its name is TITAS.","route":"knowledge_query",'
                '"user_intent":{"type":"kb_query_visible_objects"}}'
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
        user_text='What is its name?',
        history=['user:What can you see?', 'assistant:I can see a book.'],
        user_id='user1',
        knowledge_snapshot=(
            'Grounded context JSON:\n'
            '```json\n'
            '{"entities":[{"id":"book_znpbs","label":"book","kind":"object",'
            '"class":"Book","visible":true,"relations":[{"predicate":"dbp:name",'
            '"object":"TITAS"}]}]}\n'
            '```'
        ),
    )

    prompt = transport.calls[0]['messages'][0]['content']
    assert result.verbal_ack == 'Its name is TITAS.'
    assert '"id":"book_znpbs"' in prompt
    assert '"predicate":"dbp:name"' in prompt
    assert '"object":"TITAS"' in prompt
    assert 'Resolve pronouns such as "it"' in prompt
    assert 'salient stable' in prompt
    assert 'names, colors, and locations' in prompt


@pytest.mark.parametrize('user_text', ['What about now?', 'What can you see?'])
def test_turn_engine_prompt_keeps_current_grounded_facts_for_scene_followups(user_text):
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I can see a book named TITAS.",'
                '"route":"knowledge_query",'
                '"user_intent":{"type":"kb_query_visible_objects"}}'
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
        user_text=user_text,
        history=['user:What can you see?', 'assistant:I can see a book.'],
        user_id='user1',
        knowledge_snapshot=(
            'Grounded context JSON:\n'
            '```json\n'
            '{"entities":[{"id":"book_znpbs","label":"book","kind":"object",'
            '"class":"Book","visible":true,"relations":[{"predicate":"dbp:name",'
            '"object":"TITAS"}]}]}\n'
            '```'
        ),
    )

    assert result.verbal_ack == 'I can see a book named TITAS.'
    assert '"predicate":"dbp:name"' in transport.calls[0]['messages'][0]['content']
    assert '"object":"TITAS"' in transport.calls[0]['messages'][0]['content']


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
                '"user_intent":{"type":"fallback",'
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
    assert 'Grounded context:\nperson_1 rdf:type Person' in transport.calls[0]['messages'][0]['content']


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


def test_turn_engine_planner_mode_keeps_capability_question_dialogue_only():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I can move, look around, find objects, and navigate.",'
                '"user_intent":{"type":"navigate_to"},"confidence":0.0}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='I am tired. What can you do?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'


def test_turn_engine_planner_mode_keeps_fake_skill_question_dialogue_only():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Yes, I have fake skills such as navigating, finding objects, '
                'and scanning my environment.","confidence":0.0}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Perfect, do you have any fake skills?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'


def test_turn_engine_keeps_prior_execution_question_dialogue_only():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I moved my head in four directions: left, right, up, and down.",'
                '"user_intent":{"type":"head_look_left"},"confidence":0.0}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='How many directions did you move your head?',
        history=[
            'user:move your head in all directions',
            'assistant:I moved my head left, right, up, and down.',
        ],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.intent_source == 'llm_response_inferred_route'
    assert result.user_intent.get('type') == 'fallback'


def test_turn_engine_planner_mode_promotes_dialogue_wave_ack_to_execution():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will wave at you again.",'
                '"route":"dialogue","confidence":0.0}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Do it again!',
        history=[],
        user_id='user1',
    )

    assert len(transport.calls) == 1
    assert result.route == 'execution'
    assert result.intent == 'wave_greet'
    assert result.intent_source == 'llm_response_route'
    assert result.verbal_ack == 'Sure, I will wave at you again.'


def test_turn_engine_planner_mode_hands_execution_to_planner_when_llm_response_fails():
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
    assert result.verbal_ack == 'I will try that now.'
    assert result.intent_source == 'llm_response_failed_execution_handoff'
    assert result.user_intent == {
        'type': 'fallback',
        'goal_text': 'can you look around and tell me what you see',
    }
    assert result.route == 'execution'


def test_turn_engine_planner_mode_hands_scan_timeout_to_planner():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm_with_rules_fallback', planner_mode_enabled=True),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Can you scan the room for a person?',
        history=[],
        user_id='user1',
    )

    assert result.intent_source == 'llm_response_failed_execution_handoff'
    assert result.user_intent == {
        'type': 'fallback',
        'goal_text': 'Can you scan the room for a person?',
    }


def test_turn_engine_renders_planner_completion_for_system_payload_with_llm():
    transport = FakeTransport(
        [
            '{"verbal_ack":"I scanned the room and found one person."}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"planner_completion":{"goal_text":"scan the room",'
            '"result_summary":"found one person","requested_intents":["scan"]}}'
        ),
        history=['system:planner finished goal_1'],
        user_id='__system__',
    )

    assert result.success is True
    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.intent_source == 'planner_completion'
    assert result.verbal_ack == 'I scanned the room and found one person.'
    assert result.updated_history[-1] == 'assistant:I scanned the room and found one person.'
    assert len(transport.calls) == 1
    assert 'You are NAO.' in transport.calls[0]['messages'][0]['content']
    assert 'Respond briefly.' in transport.calls[0]['messages'][0]['content']
    assert 'Planner completion wording task:' in transport.calls[0]['messages'][0]['content']
    assert 'planner_completion' in transport.calls[0]['messages'][1]['content']


def test_turn_engine_renders_execution_report_for_system_payload_with_llm():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I navigated to the cup and found two blueberries '
                'nearby."}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"navigate to the cup and report other objects",'
            '"requested_intents":["navigate_to","inspect_scene","report_result"],'
            '"dialogue_context":["user:Navigate to the cup and tell me what else you see.",'
            '"assistant:Sure, I will navigate to the cup and look around."],'
            '"scene_targets":["cup"],'
            '"grounded_context":{"entities":[{"id":"cup_1","label":"gold cup"}]},'
            '"requested_summary":"I completed destination navigation to cup_1.",'
            '"steps":[{"id":"step_1","name":"navigate_to","type":"skill",'
            '"args":{"target":"cup_1"},'
            '"status":"succeeded","result_summary":"I navigated to the cup.",'
            '"result_payload":{"skill":"navigate_to","status":"succeeded","target":"cup"}},'
            '{"id":"step_2","name":"scan","type":"skill","status":"succeeded",'
            '"result_summary":"I found two blueberries.",'
            '"result_payload":{"skill":"scan","objects":[{"label":"blueberry"},'
            '{"label":"blueberry"}]}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.success is True
    assert result.route == 'dialogue'
    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I navigated to the cup and found two blueberries nearby.'
    assert len(transport.calls) == 1
    assert 'You are NAO.' in transport.calls[0]['messages'][0]['content']
    assert 'Respond briefly.' in transport.calls[0]['messages'][0]['content']
    assert 'Execution report wording task:' in transport.calls[0]['messages'][0]['content']
    assert 'dialogue_context' in transport.calls[0]['messages'][1]['content']
    assert 'grounded_context' in transport.calls[0]['messages'][1]['content']
    assert 'requested_summary' in transport.calls[0]['messages'][1]['content']
    assert '"target":"cup_1"' in transport.calls[0]['messages'][1]['content']
    assert 'execution_report' in transport.calls[0]['messages'][1]['content']


def test_turn_engine_execution_report_fallback_uses_step_summaries():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"steps":['
            '{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I navigated to the cup."},'
            '{"name":"scan","status":"succeeded",'
            '"result_summary":"I found two blueberries."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I navigated to the cup. I found two blueberries.'


def test_turn_engine_planner_completion_fallback_without_llm_response():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='{"planner_completion":{"result_summary":"Head motion completed."}}',
        history=[],
        user_id='__system__',
    )

    assert result.success is True
    assert result.intent_source == 'planner_completion'
    assert result.verbal_ack == 'Head motion completed.'
    assert result.route == 'dialogue'


def test_turn_engine_renders_planner_dialogue_for_system_payload_with_llm():
    transport = FakeTransport(
        [
            '{"verbal_ack":"Could you specify which cup you mean?"}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"planner_dialogue":{"act":"ask_clarification","reason":"target ambiguous",'
            '"text_hint":"I need a little more detail before continuing."}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.success is True
    assert result.route == 'dialogue'
    assert result.intent_source == 'planner_dialogue'
    assert result.verbal_ack == 'Could you specify which cup you mean?'
    assert len(transport.calls) == 1
    assert 'You are NAO.' in transport.calls[0]['messages'][0]['content']
    assert 'Respond briefly.' in transport.calls[0]['messages'][0]['content']
    assert 'Planner dialogue wording task:' in transport.calls[0]['messages'][0]['content']
    assert 'planner_dialogue' in transport.calls[0]['messages'][1]['content']


def test_turn_engine_planner_dialogue_fallback_without_llm_response():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='{"planner_dialogue":{"act":"ask_for_help","reason":"please hold the object steady"}}',
        history=[],
        user_id='__system__',
    )

    assert result.success is True
    assert result.intent_source == 'planner_dialogue'
    assert result.verbal_ack == 'I need help to continue this task.'
    assert result.route == 'dialogue'


def test_turn_engine_planner_dialogue_fallback_never_exposes_raw_planner_text():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"planner_dialogue":{"act":"progress_update",'
            '"reason":"INTERNAL PLANNER DETAIL","text_hint":"RAW MODEL OUTPUT"}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.verbal_ack == 'I am working on it now.'
    assert 'INTERNAL PLANNER DETAIL' not in result.verbal_ack
    assert 'RAW MODEL OUTPUT' not in result.verbal_ack


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


def test_turn_engine_preserves_intent_sequence_metadata():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will move my head right and report what I see.",'
                '"route":"execution",'
                '"user_intent":{"type":"head_look_right",'
                '"intent_sequence":["head_look_right","inspect_scene","report_result"],'
                '"goal":"move head right and report visible objects"}}'
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
        user_text='move your head right and tell me what you see',
        history=[],
        user_id='user1',
    )

    assert result.user_intent == {
        'type': 'head_look_right',
        'goal': 'move head right and report visible objects',
        'intent_sequence': ['head_look_right', 'inspect_scene', 'report_result'],
    }


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


def test_turn_engine_honors_explicit_dialogue_route_when_ack_promises_execution():
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I cannot confirm the book now. I will scan the area to locate it.",'
                '"route":"dialogue","user_intent":{"type":"fallback"}}'
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
        user_text='Can you navigate to the book?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent_source == 'llm_response_route'


def test_turn_engine_honors_explicit_dialogue_route_for_future_action_discussion() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Yes, we can plan that later when you ask me to do it.",'
                '"route":"dialogue","user_intent":{"type":"navigate_to"},"confidence":0.86}'
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
        user_text='Could we navigate to the probe cup later?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == 'navigate_to'
    assert result.intent_source == 'llm_response_route'


def test_turn_engine_routes_immediate_look_at_report_as_execution() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I understand. I will look at the probe cup and report back.",'
                '"user_intent":{"type":"kb_query_visible_objects"},"confidence":0.0}'
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
        user_text='Look at the probe cup and then tell me what you did.',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'


def test_turn_engine_honors_explicit_dialogue_route_over_execution_skill_intent() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will wave now.",'
                '"route":"dialogue","user_intent":{"type":"wave_greet"}}'
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
        user_text='Please wave and greet me.',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == 'wave_greet'


def test_turn_engine_keeps_social_greeting_on_dialogue_route() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will look for a blueberry.",'
                '"route":"execution","user_intent":{"type":"fallback"}}'
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
        user_text='hey!',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'


def test_turn_engine_keeps_named_greeting_on_dialogue_route() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Hello! How can I help you today?",'
                '"route":"execution","user_intent":{"type":"greet"}}'
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
        user_text='Hey Pop!',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == 'greet'


def test_turn_engine_planner_mode_forces_kb_query_route_for_visibility_question() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will check that now.",'
                '"route":"execution","user_intent":{"type":"inspect_scene"}}'
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
        user_text='Can you tell me if you see anyone right now?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'knowledge_query'
    assert result.intent == 'kb_query_visible_people'
    assert result.user_intent.get('type', '') == 'kb_query_visible_people'
    assert 'fresh scan' not in result.verbal_ack.lower()


def test_turn_engine_planner_mode_forces_kb_query_route_for_people_wording() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will check that now.",'
                '"route":"execution","user_intent":{"type":"inspect_scene"}}'
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
        user_text='Do you see any people?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'knowledge_query'
    assert result.intent == 'kb_query_visible_people'
    assert result.user_intent.get('type', '') == 'kb_query_visible_people'


def test_turn_engine_planner_mode_sanitizes_execution_ack_result_leak() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will look around and report what I can see. '
                '(performs a scan) I can currently see one person in the scene.",'
                '"route":"execution","user_intent":{"type":"inspect_scene"}}'
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
        user_text='can you scan for people?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.verbal_ack == 'Sure, I will look around and report what I can see.'
    assert '(performs' not in result.updated_history[-1]
    assert 'currently see' not in result.updated_history[-1]


def test_turn_engine_planner_mode_sanitizes_past_tense_scan_ack() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I have scanned the scene and found one person.",'
                '"route":"execution","user_intent":{"type":"inspect_scene"}}'
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
        user_text='scan the room',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.verbal_ack == 'Okay, I will do that.'
    assert result.updated_history[-1] == 'assistant:Okay, I will do that.'


def test_turn_engine_planner_mode_sanitizes_failure_style_execution_ack() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I cannot navigate to the phone because I do not have a clear path.",'
                '"route":"execution","user_intent":{"type":"navigate_to"}}'
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
        user_text='navigate to the phone',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.verbal_ack == 'Okay, I will try that now.'
    assert result.updated_history[-1] == 'assistant:Okay, I will try that now.'


def test_turn_engine_planner_mode_keeps_execution_route_when_scan_is_explicit() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will scan and report.",'
                '"route":"execution","user_intent":{"type":"inspect_scene"}}'
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
        user_text='Can you scan the scene and tell me if you see anyone?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'


def test_turn_engine_knowledge_query_does_not_duplicate_scan_offer() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I can currently see one person. If you want, I can run a fresh scan to confirm.",'
                '"route":"knowledge_query","user_intent":{"type":"kb_query_visible_people"}}'
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
        user_text='Do you see anyone right now?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'knowledge_query'
    assert result.verbal_ack.lower().count('fresh scan') == 1
