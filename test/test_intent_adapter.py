import json

from chatbot_llm.intent_adapter import build_response_intents
from hri_actions_msgs.msg import Intent


def test_build_response_intents_maps_motion_to_perform_motion():
    intents = build_response_intents(
        resolved_intent='posture_stand',
        user_intent={'type': 'posture_stand'},
        source_user_id='user1',
        verbal_ack='Sure.',
        raw_input='please stand up',
        confidence=0.9,
    )

    assert len(intents) == 1
    assert intents[0].intent == Intent.PERFORM_MOTION
    assert json.loads(intents[0].data) == {
        'object': 'stand',
        'ack_text': 'Sure.',
        'ack_mode': 'say',
        'plan': [
            {
                'type': 'skill',
                'name': 'perform_motion',
                'args': {'object': 'stand'},
            }
        ],
    }


def test_build_response_intents_maps_greet_with_response_hint():
    intents = build_response_intents(
        resolved_intent='greet',
        user_intent={'type': 'greet'},
        source_user_id='user1',
        verbal_ack='Hello there!',
        raw_input='hello',
        confidence=0.8,
    )

    assert len(intents) == 1
    assert intents[0].intent == Intent.GREET
    assert json.loads(intents[0].data) == {
        'ack_text': 'Hello there!',
        'ack_mode': 'say',
        'suggested_response': 'Hello there!',
    }


def test_build_response_intents_ignores_response_only_intents():
    intents = build_response_intents(
        resolved_intent='identity',
        user_intent={'type': 'identity'},
        source_user_id='user1',
        verbal_ack='I am your NAO chatbot backend.',
        raw_input='who are you?',
        confidence=1.0,
    )

    assert intents == []


def test_build_response_intents_preserves_kb_query_intents():
    intents = build_response_intents(
        resolved_intent='kb_query_visible_people',
        user_intent={'type': 'kb_query_visible_people', 'goal': 'visible_people'},
        source_user_id='user1',
        verbal_ack='I can currently see one person.',
        raw_input='who can you see?',
        confidence=0.6,
    )

    assert len(intents) == 1
    assert intents[0].intent == 'kb_query_visible_people'
    assert json.loads(intents[0].data) == {
        'goal': 'visible_people',
        'suggested_response': 'I can currently see one person.',
    }


def test_build_response_intents_fallbacks_to_raw_user_input():
    intents = build_response_intents(
        resolved_intent='fallback',
        user_intent={},
        source_user_id='user1',
        verbal_ack='I am not fully sure what you meant.',
        raw_input='tell me something complicated',
        confidence=0.0,
    )

    assert len(intents) == 1
    assert intents[0].intent == Intent.RAW_USER_INPUT
    assert json.loads(intents[0].data) == {
        'input': 'tell me something complicated',
        'suggested_response': 'I am not fully sure what you meant.',
        'ack_text': 'I am not fully sure what you meant.',
        'ack_mode': 'say',
    }


def test_build_response_intents_preserves_explicit_scene_targets_and_plan():
    intents = build_response_intents(
        resolved_intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'object': 'cup',
            'scene_targets': ['cup'],
            'plan': [
                {
                    'type': 'skill',
                    'name': 'bring_object',
                    'args': {'object': 'cup'},
                }
            ],
        },
        source_user_id='user1',
        verbal_ack='I will bring the cup.',
        raw_input='bring me the cup',
        confidence=0.7,
    )

    assert len(intents) == 1
    assert intents[0].intent == Intent.BRING_OBJECT
    assert json.loads(intents[0].data) == {
        'object': 'cup',
        'ack_text': 'I will bring the cup.',
        'ack_mode': 'say',
        'scene_targets': ['cup'],
        'plan': [
            {
                'type': 'skill',
                'name': 'bring_object',
                'args': {'object': 'cup'},
            }
        ],
    }
