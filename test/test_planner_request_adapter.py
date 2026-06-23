import json

from chatbot_llm.planner_request_adapter import build_planner_request_intent
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import normalize_goal_text
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
        'goal_text': 'bring me the cup',
        'normalized_intents': ['bring_object'],
        'scene_targets': ['cup'],
        'dialogue_context': [
            'assistant:Hello.',
            'user:bring me the cup',
            'assistant:I will bring the cup.',
            'user:thanks',
            'assistant:You are welcome.',
            'user:bring me the cup now',
        ],
        'grounded_context': {'entities': []},
        'planner_mode': 'default',
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
            user_intent={'type': 'head_look_left'},
        ),
        knowledge_context='',
        planner_request_intent='planner_request',
    )

    assert msg.intent == 'planner_request'
    assert msg.source == 'user1'
    assert msg.confidence == 0.6
    assert msg.priority == 128
    payload = json.loads(msg.data)
    assert payload['normalized_intents'] == ['head_look_left']
    assert payload['goal_id'] == 'goal_turn_2'
    assert payload['request_kind'] == 'new_goal'
    assert 'user_text' not in payload


def test_build_planner_request_payload_preserves_intent_sequence_hints():
    payload = build_planner_request_payload(
        turn_id='turn_sequence',
        user_text='move your head in all directions',
        turn_result=_make_result(
            intent='head_look_left',
            user_intent={
                'type': 'head_look_left',
                'goal': 'move the head in all directions',
                'intent_sequence': [
                    'head_look_left',
                    'head_look_right',
                    'head_look_up',
                    'head_look_down',
                ],
            },
        ),
        knowledge_context='',
    )

    assert payload['normalized_intents'] == [
        'head_look_left',
        'head_look_right',
        'head_look_up',
        'head_look_down',
    ]
    assert payload['goal_text'] == 'move the head in all directions'


def test_build_planner_request_intent_uses_confidence_floor_for_execution_route():
    msg = build_planner_request_intent(
        turn_id='turn_floor',
        user_text='look around',
        source_user_id='user1',
        turn_result=_make_result(intent_confidence=0.0, route='execution'),
        knowledge_context='',
    )

    assert msg.confidence == 0.5
    assert msg.priority == 128


def test_build_planner_request_payload_keeps_richer_grounded_context() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_grounded',
        user_text='look at the cup',
        turn_result=_make_result(),
        knowledge_context='cup isOn table',
        grounded_context={
            'scene_summary': {'objects': [{'label': 'cup'}]},
            'state_t0': {'entities': [{'id': 'cup_1'}]},
        },
    )

    assert payload['grounded_context'] == {
        'entities': [
            {
                'id': 'cup_1',
                'label': 'cup_1',
                'kind': 'object',
                'visible': True,
            }
        ]
    }


def test_build_planner_request_payload_derives_structured_kb_references() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_refs',
        user_text='inspect the cup',
        turn_result=_make_result(),
        knowledge_context='- cup_1 is currently classified as Cup\n- person_1 is a Human',
    )

    assert payload['grounded_context'] == {
        'entities': [
            {
                'id': 'cup_1',
                'label': 'cup_1',
                'kind': 'object',
                'class': 'Cup',
                'visible': True,
            },
            {
                'id': 'person_1',
                'label': 'person_1',
                'kind': 'person',
                'class': 'Human',
                'visible': True,
            },
        ]
    }


def test_build_planner_request_payload_sanitizes_assistant_json_history() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_sanitize',
        user_text='please stand up',
        turn_result=_make_result(
            updated_history=[
                'user:move your head right now',
                (
                    'assistant:{"verbal_ack":"Okay, turning my head to the right now.",'
                    '"route":"execution","user_intent":{"type":"move_head"}}'
                ),
                'user:please stand up',
            ],
            verbal_ack='Sure. I am switching to a standing posture.',
            intent='posture_stand',
            user_intent={'type': 'posture_stand'},
        ),
        knowledge_context='',
    )

    assert payload['dialogue_context'] == [
        'user:move your head right now',
        'assistant:Okay, turning my head to the right now.',
        'user:please stand up',
    ]


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


def test_build_planner_request_payload_prefers_explicit_goal_text() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_goal',
        user_text='please do the thing we discussed',
        turn_result=_make_result(
            user_intent={
                'type': 'inspect_scene',
                'goal_text': 'inspect the cup and report completion',
            }
        ),
        knowledge_context='',
    )

    assert payload['goal_text'] == 'inspect the cup and report completion'
    assert 'user_text' not in payload


def test_normalize_goal_text_strips_leading_filler_and_trailing_punctuation() -> None:
    assert (
        normalize_goal_text('Can you now change the color of the book to purple!')
        == 'change the color of the book to purple'
    )
    assert (
        normalize_goal_text('Hey Pop, could you please look at the cup for me')
        == 'look at the cup for me'
    )
    assert normalize_goal_text('inspect the cup and report completion') == (
        'inspect the cup and report completion'
    )
    assert normalize_goal_text('') == ''


def test_goal_text_fallback_normalizes_verbatim_user_text() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_verbatim',
        user_text='Can you now change the color of the book to purple!',
        turn_result=_make_result(user_intent={'type': 'look_at'}),
        knowledge_context='',
    )

    assert payload['goal_text'] == 'change the color of the book to purple'


def test_build_planner_request_payload_ignores_plan_hints_from_chatbot_result():
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

    assert payload['normalized_intents'] == ['fallback']
    assert 'requested_plan' not in payload
    assert 'interaction_mode' not in payload


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


def test_build_planner_request_payload_supersedes_active_goal_for_new_goal():
    payload = build_planner_request_payload(
        turn_id='turn_10',
        user_text='now scan for people',
        turn_result=_make_result(
            intent='inspect_scene',
            user_intent={'type': 'inspect_scene'},
        ),
        knowledge_context='',
        active_goal_id='goal_existing',
    )

    assert payload['request_kind'] == 'new_goal'
    assert payload['goal_id'] == 'goal_turn_10'
    assert payload['supersedes_goal_id'] == 'goal_existing'


def test_should_route_intents_through_planner_ignores_fallback_plan_hint() -> None:
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
        route='dialogue',
    )

    assert should_route_intents_through_planner([], turn_result=result) is False


def test_should_route_intents_through_planner_for_explicit_execution_route() -> None:
    result = _make_result(
        intent='',
        user_intent={},
        route='execution',
    )

    assert should_route_intents_through_planner([], turn_result=result) is True


def test_should_not_route_fake_skill_question_through_planner() -> None:
    result = _make_result(
        verbal_ack='Yes, I have fake skills such as navigating and finding objects.',
        intent='',
        user_intent={},
        route='execution',
    )

    assert (
        should_route_intents_through_planner(
            [],
            turn_result=result,
            user_text='Perfect, do you have any fake skills?',
        )
        is False
    )


def test_build_planner_request_payload_uses_yaml_style_multi_step_heuristics():
    payload = build_planner_request_payload(
        turn_id='turn_custom_multi',
        user_text='raise your arm despues sit down',
        turn_result=_make_result(
            intent='fallback',
            user_intent={'type': 'fallback'},
        ),
        knowledge_context='',
        multi_step_heuristics={
            'coordination_markers': [' despues '],
            'action_hint_tokens': ['raise', 'sit'],
        },
    )

    assert payload['planner_mode'] == 'multi_step'


def test_build_planner_request_payload_does_not_guess_when_heuristics_do_not_match():
    payload = build_planner_request_payload(
        turn_id='turn_custom_default',
        user_text='move your head up and then sit down',
        turn_result=_make_result(
            intent='head_look_up',
            user_intent={'type': 'head_look_up'},
        ),
        knowledge_context='',
        multi_step_heuristics={
            'coordination_markers': [' despues '],
            'action_hint_tokens': ['raise'],
        },
    )

    assert payload['planner_mode'] == 'default'
