from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.turn_engine import DialogueTurnEngine
from chatbot_llm.turn_engine import _system_task_response_addendum
from chatbot_llm.prompt_builders import build_intent_prompt
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
    turn_pipeline_mode: str = 'response_first',
    response_prompt_addendum: str = 'Respond briefly.',
    intent_prompt_addendum: str = 'Infer intent.',
    fallback_response: str = 'fallback',
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
        fallback_response=fallback_response,
        max_history_messages=20,
        scene_memory_turns=4,
        robot_name='NAO',
        persona_prompt_path='',
        response_prompt_addendum=response_prompt_addendum,
        intent_prompt_addendum=intent_prompt_addendum,
        environment_description='No specific objects described.',
        response_schema={'type': 'object'},
        intent_schema={'type': 'object'},
        planner_multi_step_heuristics={
            'coordination_markers': [' and then ', ' then '],
            'action_hint_tokens': ['stand', 'sit', 'look', 'move', 'head'],
        },
        identity_reminder_every_n_turns=6,
        intent_detection_mode=intent_mode,
        turn_pipeline_mode=turn_pipeline_mode,
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
    assert 'Grounded context for this turn:\nGrounded context JSON:' in transport.calls[0]['messages'][0]['content']
    assert 'Grounded context:\n' not in transport.calls[0]['messages'][0]['content']
    assert 'Grounded context for this turn:\nGrounded context JSON:' in transport.calls[1]['messages'][0]['content']
    assert 'Grounded context:\n' not in transport.calls[1]['messages'][0]['content']


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
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            response_prompt_addendum=(
                'Response style:\n'
                '- Resolve pronouns such as "it" from recent dialogue focus '
                'and current grounded entities.'
            ),
        ),
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
        config=make_config(
            intent_mode='llm',
            intent_prompt_addendum=(
                'Knowledge-query labels: kb_query_visible_people, '
                'kb_query_visible_objects, kb_query_scene_change.'
            ),
        ),
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
        'Planner-mode response fields:'
        in transport.calls[0]['messages'][0]['content']
    )
    assert 'executable plans after this response' in transport.calls[0]['messages'][0]['content']
    assert 'Route policy, response style, and examples' in transport.calls[0]['messages'][0]['content']
    assert 'Grounded context for this turn:\nperson_1 rdf:type Person' in transport.calls[0]['messages'][0]['content']


def test_turn_engine_intent_first_locks_route_before_response_wording():
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"look_at","goal_text":"look at the cup",'
                '"scene_targets":["cup"]},"intent_confidence":0.91}'
            ),
            (
                '{"verbal_ack":"Sure, I will check the current scene and look at the cup.",'
                '"route":"execution","confidence":0.86}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='look at the cup',
        history=[],
        user_id='user1',
        knowledge_snapshot='Grounded context JSON:\n```json\n{"entities":[]}\n```',
    )

    assert len(transport.calls) == 2
    assert result.route == 'execution'
    assert result.intent == 'look_at'
    assert result.intent_source == 'llm_intent_route_lock'
    assert result.user_intent['goal_text'] == 'look at the cup'
    response_prompt = transport.calls[1]['messages'][0]['content']
    assert 'Locked route context:' in response_prompt
    assert '"route":"execution"' in response_prompt
    assert 'Do not change it' in response_prompt


def test_turn_engine_intent_first_prevents_wave_particle_execution_regression():
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"wave_greet","goal_text":"explain wave particle duality"},'
                '"intent_confidence":0.74}'
            ),
            (
                '{"verbal_ack":"Wave-particle duality is the idea that quantum objects '
                'can show wave-like and particle-like behavior.","route":"dialogue",'
                '"confidence":0.82}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='What is wave-particle duality?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.intent_source == 'llm_intent_route_lock'
    assert result.user_intent.get('type') == 'fallback'
    assert 'wave-particle duality' in result.verbal_ack.lower()
    response_prompt = transport.calls[1]['messages'][0]['content']
    assert '"route":"dialogue"' in response_prompt


def test_turn_engine_intent_first_locks_scene_question_to_knowledge_query():
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"inspect_scene","goal_text":"describe visible objects"},'
                '"intent_confidence":0.67}'
            ),
            (
                '{"verbal_ack":"I can currently see one cup on the table.",'
                '"route":"knowledge_query","confidence":0.88}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='What can you see now?',
        history=[],
        user_id='user1',
        knowledge_snapshot='Scene digest: Objects (1): cup x1 [on table]',
    )

    assert result.route == 'knowledge_query'
    assert result.intent == 'kb_query_visible_objects'
    assert result.user_intent['type'] == 'kb_query_visible_objects'
    assert '"route":"knowledge_query"' in transport.calls[1]['messages'][0]['content']


def test_turn_engine_intent_first_repairs_low_confidence_kb_route_with_concrete_action():
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"kb_query_visible_objects",'
                '"goal_text":"describe visible objects"},"intent_confidence":0.0}'
            ),
            (
                '{"verbal_ack":"I understand. I will check the scene and plan the task.",'
                '"route":"execution","confidence":0.76}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Look at the probe cup and tell me what you did.',
        history=[],
        user_id='user1',
        knowledge_snapshot='Scene digest: Objects (1): probe cup [on table]',
    )

    assert result.route == 'execution'
    assert result.intent == 'look_at'
    assert result.user_intent['type'] == 'look_at'
    assert result.user_intent['goal_text'] == 'Look at the probe cup and tell me what you did.'
    assert '"route":"execution"' in transport.calls[1]['messages'][0]['content']


def test_turn_engine_intent_first_repairs_kb_mutation_mislabeled_as_query():
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"kb_query_visible_objects",'
                '"goal_text":"describe visible objects"},"intent_confidence":0.0}'
            ),
            (
                '{"verbal_ack":"I will update the knowledge base with that fact.",'
                '"route":"execution","confidence":0.82}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Can you add another person with the name Watson to your knowledge base!',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.intent == 'kb_add'
    assert result.user_intent['type'] == 'kb_add'


def test_turn_engine_intent_first_rejects_dialogue_route_action_promise():
    backend_outage_text = 'I am having trouble reaching my language model right now.'
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"help","goal_text":"discuss a future plan"},'
                '"intent_confidence":0.66}'
            ),
            (
                '{"verbal_ack":"Sure, I will navigate to the probe cup now.",'
                '"route":"execution","confidence":0.84}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
            fallback_response=backend_outage_text,
        ),
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
    assert result.intent == ''
    assert result.verbal_ack != backend_outage_text
    assert result.verbal_ack == 'I can talk about that without starting a robot action.'


def test_turn_engine_intent_first_rejects_knowledge_query_action_promise():
    backend_outage_text = 'I am having trouble reaching my language model right now.'
    transport = FakeTransport(
        [
            (
                '{"user_intent":{"type":"inspect_scene","goal_text":"describe visible objects"},'
                '"intent_confidence":0.66}'
            ),
            (
                '{"verbal_ack":"I will walk to the cup and inspect it now.",'
                '"route":"execution","confidence":0.84}'
            ),
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            turn_pipeline_mode='intent_first',
            fallback_response=backend_outage_text,
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='What can you see now?',
        history=[],
        user_id='user1',
        knowledge_snapshot='Scene digest: Objects (1): cup x1 [on table]',
    )

    assert result.route == 'knowledge_query'
    assert result.intent == 'kb_query_visible_objects'
    assert result.verbal_ack != backend_outage_text
    assert result.verbal_ack == 'I will answer from the current grounded context.'


def test_turn_engine_planner_mode_repairs_missing_execution_route_without_second_call():
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
    assert result.intent_source == 'llm_response_route_repair'


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
    assert result.intent_source == 'llm_response_route_repair'
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
    assert 'System wording mode:' in transport.calls[0]['messages'][0]['content']
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


def test_turn_engine_execution_report_words_person_delivery_as_to_not_on():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I found the kitchen cup, which is white, '
                    'and brought it to the person named Alex. I have placed '
                    'the cup on Alex."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"bring the kitchen cup to the person named ALEX",'
            '"grounded_context":{"entities":[{"id":"codex_recipient_person",'
            '"kind":"person","class":"Human","relations":[{"predicate":"dbp:name",'
            '"object":"Alex"}]}]},'
            '"steps":[{"name":"place_object","status":"succeeded",'
            '"args":{"target":"codex_kitchen_cup","destination":"codex_recipient_person"},'
            '"result_summary":"I placed the cup with Alex.",'
            '"result_payload":{"target":"codex_kitchen_cup",'
            '"destination":"codex_recipient_person"}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack.endswith('I have delivered the cup to Alex.')
    assert 'on Alex' not in result.verbal_ack


def test_turn_engine_execution_report_preserves_surface_placement_wording():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            ['{"verbal_ack":"I picked up the apple and placed it on the table."}']
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"pick up the apple and place it on the table",'
            '"grounded_context":{"entities":[{"id":"table_zmrkd","kind":"object",'
            '"class":"Table","relations":[{"predicate":"dbp:name","object":"table"}]}]},'
            '"steps":[{"name":"place_object","status":"succeeded",'
            '"args":{"target":"apple_ajrte","destination":"table_zmrkd"},'
            '"result_summary":"I placed the apple on the table.",'
            '"result_payload":{"target":"apple_ajrte","destination":"table_zmrkd"}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I picked up the apple and placed it on the table.'


def test_turn_engine_execution_report_rejects_future_report_closure():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I looked at codex_probe_cup and will report '
                    'back on what I observed."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"steps":['
            '{"name":"look_at","status":"succeeded",'
            '"result_summary":"I looked at codex_probe_cup."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I looked at codex_probe_cup.'


def test_turn_engine_execution_report_rejects_single_skill_double_summary():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I have completed the wave gesture. '
                    'I performed a friendly wave at you."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"steps":['
            '{"name":"wave_greet","status":"succeeded",'
            '"result_summary":"I performed a friendly wave at you."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I performed a friendly wave at you.'


def test_turn_engine_execution_report_prompt_distinguishes_intermediate_role():
    transport = FakeTransport(
        [
            '{"verbal_ack":"I have navigated to the apple."}',
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
            '{"execution_report":{"goal_text":"walk to every object and report each stop",'
            '"report_role":"intermediate",'
            '"latest_result_summary":"I completed destination navigation to codex_probe_apple.",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I completed destination navigation to codex_probe_apple."}],'
            '"future_steps":[{"name":"navigate_to","args":{"target":"codex_probe_book"}},'
            '{"name":"report_result","args":{}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    prompt = transport.calls[0]['messages'][0]['content']
    payload = transport.calls[0]['messages'][1]['content']
    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I have navigated to the apple.'
    assert 'If report_role is "intermediate"' in prompt
    assert '"report_role":"intermediate"' in payload
    assert 'future_steps' in payload


def test_turn_engine_execution_report_rejects_next_object_without_future_navigation():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I have arrived at the phone. It is silver '
                    'and on the table. I will now move to the next object."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"walk to every object and report each stop",'
            '"report_role":"intermediate",'
            '"latest_result_summary":"I completed destination navigation to codex_probe_phone.",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I arrived at the phone."}],'
            '"future_steps":[{"name":"wave_greet","args":{}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I completed destination navigation to codex_probe_phone.'


def test_turn_engine_execution_report_rejects_stale_intermediate_arrival():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I have arrived at the first object, which is '
                    'the codex_probe_apple. It is red and located on the table. '
                    'Now I am walking to the next object, codex_probe_book."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"walk to every object and report each stop",'
            '"report_role":"intermediate",'
            '"latest_result_summary":"I completed destination navigation to codex_probe_book.",'
            '"latest_result_payload":{"target":"codex_probe_book"},'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"args":{"target":"codex_probe_apple"},'
            '"result_summary":"I arrived at the apple.",'
            '"result_payload":{"target":"codex_probe_apple"}},'
            '{"name":"navigate_to","status":"succeeded",'
            '"args":{"target":"codex_probe_book"},'
            '"result_summary":"I completed destination navigation to codex_probe_book.",'
            '"result_payload":{"target":"codex_probe_book"}}],'
            '"future_steps":[{"name":"navigate_to","args":{"target":"codex_probe_cup"}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I completed destination navigation to codex_probe_book.'


def test_turn_engine_execution_report_rejects_intermediate_next_movement_preview():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I have arrived at the first object, '
                    'codex_probe_apple. I am now ready to move to the next object."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"walk to every object and report each stop",'
            '"report_role":"intermediate",'
            '"latest_result_summary":"I completed destination navigation to codex_probe_apple.",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I completed destination navigation to codex_probe_apple."}],'
            '"future_steps":[{"name":"navigate_to","args":{"target":"codex_probe_book"}}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I completed destination navigation to codex_probe_apple.'


def test_turn_engine_execution_report_intermediate_fallback_prefers_latest_summary():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"report_role":"intermediate",'
            '"latest_result_summary":"I arrived at the book.",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I arrived at the apple."},'
            '{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I arrived at the book."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I arrived at the book.'


def test_turn_engine_execution_report_rejects_quoted_goal_completion():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I finished: Look at the probe cup and then '
                    'tell me what you did."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"Look at the probe cup and then tell me what you did.",'
            '"steps":[{"name":"look_at","status":"succeeded"}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == 'I looked at the target.'


def test_turn_engine_execution_report_rejects_placeholder_scene_report():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I navigated to the probe cup and then looked '
                    'around. I can report the current scene summary."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"Navigate to the probe cup and then tell me what else you see.",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"result_summary":"I navigated to the probe cup."},'
            '{"name":"scan","status":"succeeded",'
            '"result_summary":"I found a silver phone, a red apple, and a blue book."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == (
        'I navigated to the probe cup. I found a silver phone, a red apple, and a blue book.'
    )


def test_turn_engine_execution_report_fallback_synthesizes_multi_object_chain():
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='rules'),
        transport=FakeTransport(
            [
                (
                    '{"verbal_ack":"I completed destination navigation to codex_probe_phone. '
                    'I looked around and can report the current scene summary."}'
                )
            ]
        ),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text=(
            '{"execution_report":{"goal_text":"walk to every object and report each arrival",'
            '"steps":[{"name":"navigate_to","status":"succeeded",'
            '"args":{"target":"codex_probe_apple"},'
            '"result_summary":"I completed destination navigation to codex_probe_apple."},'
            '{"name":"report_result","status":"succeeded",'
            '"result_summary":"I have arrived at the apple."},'
            '{"name":"navigate_to","status":"succeeded",'
            '"args":{"target":"codex_probe_book"},'
            '"result_summary":"I completed destination navigation to codex_probe_book."},'
            '{"name":"report_result","status":"succeeded",'
            '"result_summary":"I have arrived at the book."},'
            '{"name":"navigate_to","status":"succeeded",'
            '"args":{"target":"codex_probe_phone"},'
            '"result_summary":"I completed destination navigation to codex_probe_phone."},'
            '{"name":"scan","status":"succeeded",'
            '"result_summary":"I looked around and can report the current scene summary."}]}}'
        ),
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'execution_report'
    assert result.verbal_ack == (
        'I walked to the apple, the book, and the phone and reported each arrival.'
    )


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


def test_turn_engine_renders_planner_completion_without_interactive_route_policy():
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
        history=[],
        user_id='__system__',
    )

    assert result.intent_source == 'planner_completion'
    assert 'Respond briefly.' in transport.calls[0]['messages'][0]['content']
    assert 'System wording mode:' in transport.calls[0]['messages'][0]['content']
    assert 'Planner completion wording task:' in transport.calls[0]['messages'][0]['content']


def test_system_task_response_addendum_keeps_relevant_response_guidance_only():
    base_addendum = """
Keep responses short and natural for text-to-speech.

Perception and knowledge policy:
- grounded_context is a JSON object that contains the robot's current world state.
- Do not invent objects, people, colors, names, or locations.

Response style:
- A verbal acknowledgement (verbal_ack) must be provided for all routes.
- Speak like an embodied assistant giving a status update.
- If route="dialogue":
  - Ensure to provide friendly and natural wording.
- If route="execution":
  - Provide a short and factual verbal acknowledgement.

Response confidence:
- confidence reflects how sure you are.

Route contrast examples:
- User: "Could we navigate to the cup later?"
""".strip()

    text = _system_task_response_addendum(
        base_addendum,
        'Execution report wording task:\n- Use only facts from the payload.',
    )

    assert 'Keep responses short and natural for text-to-speech.' in text
    assert 'Perception and knowledge policy:' in text
    assert 'Speak like an embodied assistant giving a status update.' in text
    assert 'Could we navigate to the cup later?' not in text
    assert 'If route="execution":' not in text
    assert 'Execution report wording task:' in text


def test_intent_prompt_builder_keeps_policy_in_prompt_pack_addendum():
    prompt = build_intent_prompt(
        robot_name='Pop',
        user_id='user1',
        system_prompt='Base system prompt.',
        environment_description='Lab.',
        knowledge_snapshot='',
        intent_prompt_addendum=(
            'Canonical intent labels from pack:\n'
            '- look_at\n'
            '- kb_add\n'
            '- fallback'
        ),
        skill_catalog_text='',
        persona_prompt='',
    )

    assert 'Base system prompt.' in prompt
    assert 'Canonical intent labels from pack:' in prompt
    assert 'Intent labels, route policy, and examples are defined by the configured' in prompt
    assert 'Canonical intent labels:\n- posture_stand' not in prompt


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


def test_turn_engine_uses_route_safe_ack_when_json_has_no_safe_ack():
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

    assert result.verbal_ack == 'Okay, I will try that now.'
    assert result.updated_history == ['user:stand up', 'assistant:Okay, I will try that now.']


def test_turn_engine_json_without_ack_does_not_claim_backend_outage():
    backend_outage_text = 'I am having trouble reaching my language model right now.'
    transport = FakeTransport(
        [
            '{"route":"dialogue","user_intent":{"type":"help"}}',
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            fallback_response=backend_outage_text,
        ),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='What is your favourite movie?',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack != backend_outage_text
    assert result.verbal_ack == 'I could not understand the request clearly enough. Could you rephrase it?'


def test_turn_engine_empty_transport_response_keeps_backend_outage_wording():
    backend_outage_text = 'I am having trouble reaching my language model right now.'
    engine = DialogueTurnEngine(
        config=make_config(
            intent_mode='llm',
            planner_mode_enabled=True,
            fallback_response=backend_outage_text,
        ),
        transport=FakeTransport(['']),
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='What is your favourite movie?',
        history=[],
        user_id='user1',
    )

    assert result.verbal_ack == backend_outage_text


def test_turn_engine_repairs_dialogue_route_when_ack_promises_execution():
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

    assert result.route == 'execution'
    assert result.intent_source == 'llm_response_route_repair'


def test_turn_engine_routes_explicit_kb_add_as_execution() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will add this to my knowledge base.",'
                '"route":"dialogue","user_intent":{"type":"fallback"}}'
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
        user_text=(
            'Add this to your knowledge base: codex_chat_marker rdf:type Cube, '
            'codex_chat_marker dbp:name NOVA.'
        ),
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.intent == 'kb_add'
    assert result.user_intent.get('type') == 'kb_add'


def test_turn_engine_keeps_kb_question_non_mutating() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I remember that the probe cup is gold.",'
                '"route":"execution","user_intent":{"type":"kb_add"}}'
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
        user_text='What do you remember about the probe cup?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'knowledge_query'
    assert result.intent != 'kb_add'
    assert result.user_intent.get('type') == 'kb_query_visible_objects'


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
    assert result.intent == ''
    assert result.intent_source == 'llm_response_route_repair'


def test_turn_engine_repairs_missing_route_for_future_action_discussion() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Yes, we can plan that later when you ask me to do it.",'
                '"user_intent":{"type":"navigate_to"},"confidence":0.86}'
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
    assert result.intent_source == 'llm_response_route_repair'


def test_turn_engine_repairs_explicit_execution_route_for_future_action_discussion() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Yes, we can plan that later when you ask me to do it.",'
                '"route":"execution","user_intent":{"type":"navigate_to"},'
                '"confidence":0.86}'
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
    assert result.intent == ''
    assert result.intent_source == 'llm_response_route_repair'


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


def test_turn_engine_fills_missing_bring_intent_for_planner_mode() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will bring the apple and the book to Alex.",'
                '"route":"execution","user_intent":{},"confidence":0.0}'
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
        user_text='Bring the apple and the book to Alex, then report what you did.',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.intent == 'bring_object'
    assert result.user_intent['type'] == 'bring_object'


def test_turn_engine_blocks_execution_for_absent_named_person_target() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I understand. I will bring every object from the work table '
                'to the person named BLAKE and report back.",'
                '"route":"execution",'
                '"user_intent":{"type":"bring_object",'
                '"goal":"Bring every object from the work table to the person named BLAKE.",'
                '"scene_targets":["codex_lab_table_section","BLAKE"]},'
                '"confidence":0.9}'
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
        user_text='Bring every object from the work table to the person named BLAKE.',
        history=[],
        user_id='user1',
        knowledge_snapshot=(
            'Grounded context JSON:\n```json\n'
            '{"entities":[{"id":"person_alex","kind":"person",'
            '"relations":[{"predicate":"dbp:name","object":"ALEX"}]}],'
            '"locations":[{"id":"codex_lab_table_section","kind":"location_group",'
            '"contains":[{"id":"codex_lab_cup","kind":"object"}]}]}'
            '\n```'
        ),
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.user_intent['type'] == 'fallback'
    assert result.user_intent['route_conflict'] == {
        'requested_person': 'BLAKE',
        'reason': 'missing_named_person_in_grounded_context',
    }
    assert 'cannot confirm that person' in result.verbal_ack.lower()
    assert 'bring every object' not in result.verbal_ack.lower()


def test_turn_engine_allows_execution_for_present_named_person_target() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will bring the apple to ALEX.",'
                '"route":"execution",'
                '"user_intent":{"type":"bring_object","object":"apple","recipient":"ALEX"},'
                '"confidence":0.9}'
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
        user_text='Bring the apple to the person named ALEX.',
        history=[],
        user_id='user1',
        knowledge_snapshot=(
            'Grounded context JSON:\n```json\n'
            '{"entities":[{"id":"person_alex","kind":"person",'
            '"relations":[{"predicate":"dbp:name","object":"ALEX"}]}]}'
            '\n```'
        ),
    )

    assert result.route == 'execution'
    assert result.intent == 'bring_object'


def test_turn_engine_repairs_dialogue_route_over_execution_skill_intent() -> None:
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

    assert result.route == 'execution'
    assert result.intent == 'wave_greet'


def test_turn_engine_keeps_wave_particle_question_on_dialogue_route() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"The wave-particle relation is described by '
                'the de Broglie equation.","route":"dialogue"}'
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
        user_text='Wave-particle equation?',
        history=[
            'user:speed of light?',
            'assistant:The speed of light in a vacuum is approximately 299,792,458 meters per second.',
        ],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent != 'wave_greet'
    assert result.user_intent == {}


def test_turn_engine_keeps_wave_particle_duality_question_dialogue_when_route_missing() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Wave-particle duality is a concept in quantum '
                'mechanics."}'
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
        user_text='What is wave-particle duality?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.user_intent == {}


def test_turn_engine_keeps_wave_particle_duality_dialogue_despite_bad_skill_intent() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Wave-particle duality describes quantum behavior.",'
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
        user_text='What is wave-particle duality?',
        history=['assistant:I can move my head and wave.'],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.user_intent.get('type') == 'fallback'


def test_turn_engine_keeps_personal_preference_question_dialogue_only() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Okay, I will try that now.",'
                '"route":"execution","user_intent":{"type":"help"}}'
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
        user_text='What is your favorite movie?',
        history=['assistant:I arrived at the apple.'],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == ''
    assert result.user_intent == {}


def test_turn_engine_keeps_advice_request_dialogue_despite_action_words_in_answer() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"I can suggest a few plans: go to a park, visit a museum, '
                'or meet friends for coffee.",'
                '"route":"dialogue","user_intent":{"type":"help"},"confidence":0.82}'
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
        user_text='Any ideas for plans this weekend?',
        history=['assistant:I can help with planning.'],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert result.intent == 'help'
    assert result.intent_source == 'llm_response_route'
    assert result.user_intent.get('type') == 'help'


def test_turn_engine_clamps_long_dialogue_response_for_speech_transport() -> None:
    long_answer = ' '.join(['Time dilation means moving clocks run slower.'] * 80)
    transport = FakeTransport(
        [
            '{"verbal_ack":"%s","route":"dialogue"}' % long_answer,
        ]
    )
    engine = DialogueTurnEngine(
        config=make_config(intent_mode='llm', planner_mode_enabled=True),
        transport=transport,
        logger=None,
        skill_catalog_text='',
    )

    result = engine.execute_turn(
        user_text='Can you explain time dilation in relativity?',
        history=[],
        user_id='user1',
    )

    assert result.route == 'dialogue'
    assert len(result.verbal_ack) <= 950
    assert result.verbal_ack.endswith('I can continue if you want more detail.')


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


def test_turn_engine_planner_mode_sanitizes_step_trace_execution_ack() -> None:
    transport = FakeTransport(
        [
            (
                '{"verbal_ack":"Sure, I will walk to each object one by one, '
                "let you know when I arrive, and then proceed to the next. Let's begin.\\n\\n"
                'I am now walking to the apple. I have arrived at the apple. '
                'I am now walking to the book. I have arrived at the book.",'
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
        user_text='walk to every object',
        history=[],
        user_id='user1',
    )

    assert result.route == 'execution'
    assert result.verbal_ack == (
        "Sure, I will walk to each object one by one, let you know when I arrive, "
        "and then proceed to the next. Let's begin."
    )
    assert 'I am now walking' not in result.updated_history[-1]
    assert 'I have arrived' not in result.updated_history[-1]


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
