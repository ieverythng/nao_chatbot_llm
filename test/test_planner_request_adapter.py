import json

from chatbot_llm.planner_request_adapter import build_planner_request_intent
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import should_route_intents_through_planner
from chatbot_llm.planner_request_adapter import Intent
from chatbot_llm.turn_engine import TurnExecutionResult


def _make_result(**overrides):
    payload = {
        'success': True,
        'verbal_ack': 'I will bring the cup.',
        'updated_history': [
            'user:hello',
            'assistant:Hello.',
            'user:bring me the cup',
            'assistant:I will bring the cup.',
            'user:thanks',
            'assistant:You are welcome.',
            'user:bring me the cup now',
        ],
        'intent': 'bring_object',
        'intent_source': 'llm_intent',
        'intent_confidence': 0.8,
        'user_intent': {'type': 'bring_object', 'object': 'cup'},
        'route': 'execution',
    }
    payload.update(overrides)
    return TurnExecutionResult(**payload)


def test_build_planner_request_payload_derives_scene_targets_and_bounds_context():
    payload = build_planner_request_payload(
        turn_id='turn_1',
        user_text='bring me the cup',
        turn_result=_make_result(),
        knowledge_context='cup isOn table',
    )

    assert payload == {
        'request_id': 'turn_1',
        'goal_id': 'goal_turn_1',
        'parent_goal_id': '',
        'supersedes_goal_id': '',
        'request_kind': 'new_goal',
        'user_text': 'bring me the cup',
        'normalized_intents': ['bring_object'],
        'ack_text': 'I will bring the cup.',
        'ack_mode': 'say',
        'scene_targets': ['cup'],
        'dialogue_context': [
            'assistant:Hello.',
            'user:bring me the cup',
            'assistant:I will bring the cup.',
            'user:thanks',
            'assistant:You are welcome.',
            'user:bring me the cup now',
        ],
        'requested_plan': [],
        'grounded_context': {
            'knowledge_snapshot': {'summary_text': 'cup isOn table'},
            'scene_summary': {},
            'world_model_snapshot': {},
            'world_model_text': '',
        },
        'planner_mode': 'default',
        'interaction_mode': 'speech',
        'dialogue_turn_id': 'turn_1',
    }


def test_build_planner_request_intent_encodes_expected_message_shape():
    msg = build_planner_request_intent(
        turn_id='turn_2',
        user_text='look left',
        source_user_id='user1',
        turn_result=_make_result(
            verbal_ack='I will look left.',
            intent='head_look_left',
            intent_confidence=0.6,
            user_intent={'type': 'head_look_left', 'ack_mode': 'auto'},
        ),
        knowledge_context='',
        planner_request_intent='planner_request',
    )

    assert msg.intent == 'planner_request'
    assert msg.source == 'user1'
    assert msg.confidence == 0.6
    payload = json.loads(msg.data)
    assert payload['normalized_intents'] == ['head_look_left']
    assert payload['ack_mode'] == 'auto'
    assert payload['goal_id'] == 'goal_turn_2'
    assert payload['request_kind'] == 'new_goal'


def test_build_planner_request_payload_splits_explicit_scene_targets_string():
    payload = build_planner_request_payload(
        turn_id='turn_3',
        user_text='check the cup and book',
        turn_result=_make_result(
            user_intent={
                'type': 'inspect_scene',
                'scene_targets': 'cup, book',
            }
        ),
        knowledge_context='',
    )

    assert payload['scene_targets'] == ['cup', 'book']


def test_build_planner_request_payload_ignores_capitalized_motion_objects_as_scene_targets():
    payload = build_planner_request_payload(
        turn_id='turn_4',
        user_text='stand up',
        turn_result=_make_result(
            intent='posture_stand',
            user_intent={
                'type': 'posture_stand',
                'object': 'Stand',
            },
        ),
        knowledge_context='',
    )

    assert payload['scene_targets'] == []


def test_build_planner_request_payload_marks_multi_step_turns_for_planner_mode():
    payload = build_planner_request_payload(
        turn_id='turn_multi',
        user_text='move your head up and then sit down',
        turn_result=_make_result(
            intent='head_look_up',
            user_intent={'type': 'head_look_up'},
        ),
        knowledge_context='',
    )

    assert payload['normalized_intents'] == ['head_look_up']
    assert payload['planner_mode'] == 'multi_step'


def test_build_planner_request_payload_preserves_requested_plan_and_filters_ack_step():
    payload = build_planner_request_payload(
        turn_id='turn_hint',
        user_text='stand up and then tell me when you are done',
        turn_result=_make_result(
            verbal_ack='I will stand up and let you know.',
            intent='fallback',
            user_intent={
                'type': 'fallback',
                'ack_text': 'I will stand up and let you know.',
                'plan': [
                    {
                        'type': 'say',
                        'name': 'say',
                        'args': {'text': 'I will stand up and let you know.'},
                    },
                    {
                        'type': 'skill',
                        'name': 'perform_motion',
                        'args': {'object': 'stand'},
                    },
                    {
                        'type': 'say',
                        'name': 'say',
                        'args': {'text': 'I am standing now.'},
                    },
                ],
            },
        ),
        knowledge_context='',
    )

    assert payload['normalized_intents'] == ['posture_stand']
    assert payload['requested_plan'] == [
        {
            'type': 'skill',
            'name': 'perform_motion',
            'args': {'object': 'stand'},
        },
        {
            'type': 'say',
            'name': 'say',
            'args': {'text': 'I am standing now.'},
        },
    ]


def test_should_route_intents_through_planner_only_for_execution_intents():
    greet_intent = Intent()
    greet_intent.intent = Intent.GREET
    motion_intent = Intent()
    motion_intent.intent = Intent.PERFORM_MOTION

    assert should_route_intents_through_planner([]) is False
    assert should_route_intents_through_planner([greet_intent]) is False
    assert should_route_intents_through_planner([greet_intent, motion_intent]) is True


def test_should_route_intents_through_planner_normalizes_intent_names():
    greet_intent = Intent()
    greet_intent.intent = ' GREET '
    motion_intent = Intent()
    motion_intent.intent = ' perform_motion '

    assert should_route_intents_through_planner([greet_intent]) is False
    assert should_route_intents_through_planner([motion_intent]) is True


def test_build_planner_request_payload_reuses_active_goal_for_cancel_request():
    payload = build_planner_request_payload(
        turn_id='turn_9',
        user_text='stop that',
        turn_result=_make_result(
            intent='cancel_request',
            user_intent={'type': 'cancel_request'},
        ),
        knowledge_context='',
        active_goal_id='goal_existing',
    )

    assert payload['request_kind'] == 'cancel_request'
    assert payload['goal_id'] == 'goal_existing'


def test_should_route_intents_through_planner_for_fallback_with_plan() -> None:
    result = _make_result(
        intent='fallback',
        user_intent={
            'type': 'fallback',
            'plan': [
                {
                    'type': 'skill',
                    'name': 'perform_motion',
                    'args': {'object': 'stand'},
                },
                {
                    'type': 'say',
                    'name': 'say',
                    'args': {'text': 'I am standing now.'},
                },
            ],
        },
    )

    assert should_route_intents_through_planner([], turn_result=result) is True


def test_should_route_intents_through_planner_for_explicit_execution_route() -> None:
    result = _make_result(
        intent='',
        user_intent={},
        route='execution',
    )

    assert should_route_intents_through_planner([], turn_result=result) is True
