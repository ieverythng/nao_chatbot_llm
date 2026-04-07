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
        'grounded_context': {'knowledge_snapshot': 'cup isOn table'},
        'planner_mode': 'default',
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
    assert json.loads(msg.data)['normalized_intents'] == ['head_look_left']
    assert json.loads(msg.data)['ack_mode'] == 'auto'


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
