import json

from chatbot_llm.planner_request_adapter import build_planner_request_intent
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import dialogue_turn_id
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
        'intent_source': 'llm_intent_retry_exhausted',
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
        'grounded_context': {
            'entities': [],
            'counts': {
                'entities': 0,
                'people': 0,
                'objects': 0,
                'locations': 0,
            },
        },
        'planner_mode': 'default',
        'dialogue_turn_id': 'turn_1',
    }


def test_build_planner_request_payload_does_not_reconstruct_healthy_llm_selection() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_missing_declared_selection',
        user_text='Bring the book to Alex.',
        turn_result=_make_result(
            intent_source='llm_intent',
            user_intent={
                'type': 'bring_object',
                'goal_text': 'bring the book to Alex',
                'scene_targets': ['book_1', 'person_alex'],
            },
        ),
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_1', 'kind': 'object'},
                {'id': 'person_alex', 'kind': 'person'},
            ],
        },
    )

    assert 'target_selection' not in payload


def test_build_planner_request_payload_salvages_selection_after_intent_retry() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_exhausted_selection_retry',
        user_text='Bring the book to Alex.',
        turn_result=_make_result(
            intent_source='llm_response_route+llm_intent_retry_exhausted',
            user_intent={
                'type': 'bring_object',
                'goal_text': 'bring the book to Alex',
                'scene_targets': ['book_1', 'person_alex'],
            },
        ),
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_1', 'kind': 'object'},
                {'id': 'person_alex', 'kind': 'person'},
            ],
        },
    )

    assert payload['target_selection']['member_ids'] == ['book_1']
    assert payload['target_selection']['recipient_id'] == 'person_alex'


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


def test_build_planner_request_payload_generates_goal_for_dialogue_local_turn_id():
    payload = build_planner_request_payload(
        turn_id='__default__',
        user_text='bring me the cup',
        turn_result=_make_result(),
        knowledge_context='',
    )

    assert payload['goal_id'].startswith('goal_')
    assert payload['goal_id'] != 'goal___default__'
    assert payload['dialogue_turn_id'] == '__default__'


def test_build_planner_request_payload_preserves_scoped_dialogue_turn_goal_id():
    payload = build_planner_request_payload(
        turn_id='__default__:01020304:1',
        user_text='bring me the cup',
        turn_result=_make_result(),
        knowledge_context='',
    )

    assert payload['goal_id'] == 'goal_default___01020304_1'
    assert payload['dialogue_turn_id'] == '__default__:01020304:1'


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


def test_return_to_person_is_robot_navigation_not_object_delivery() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_return_to_alex',
        user_text='Sit, stand, pick up MIDAS, return to ALEX, and summarize.',
        turn_result=_make_result(
            intent='bring_object',
            intent_source='llm_response_route+llm_intent',
            user_intent={
                'type': 'bring_object',
                'goal_text': 'sit, stand, pick up MIDAS, return to ALEX, and summarize',
                'intent_sequence': [
                    'posture_sit',
                    'posture_stand',
                    'pick_object',
                    'navigate_to',
                    'bring_object',
                    'report_result',
                ],
                'scene_targets': ['book_midas', 'person_alex'],
                'target_selection': {
                    'selection_kind': 'explicit_members',
                    'operation': 'deliver',
                    'member_ids': ['book_midas'],
                    'recipient_id': 'person_alex',
                    'report_policy': 'final',
                },
            },
        ),
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_midas', 'label': 'MIDAS', 'kind': 'object'},
                {
                    'id': 'person_alex',
                    'label': 'ALEX',
                    'kind': 'person',
                    'relations': [{'predicate': 'dbp:name', 'object': 'ALEX'}],
                },
            ]
        },
    )

    assert payload['normalized_intents'] == [
        'posture_sit',
        'posture_stand',
        'pick_object',
        'navigate_to',
        'report_result',
    ]
    assert payload['scene_targets'] == ['book_midas', 'person_alex']
    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'visit',
        'source_location_id': '',
        'member_ids': ['person_alex'],
        'recipient_id': '',
        'ordering': 'sequential',
        'report_policy': 'final',
    }


def test_return_it_to_person_remains_object_delivery() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_return_it_to_alex',
        user_text='Pick up MIDAS and return it to ALEX.',
        turn_result=_make_result(
            intent='bring_object',
            intent_source='llm_response_route+llm_intent',
            user_intent={
                'type': 'bring_object',
                'intent_sequence': ['pick_object', 'bring_object'],
                'scene_targets': ['book_midas', 'person_alex'],
                'target_selection': {
                    'selection_kind': 'explicit_members',
                    'operation': 'deliver',
                    'member_ids': ['book_midas'],
                    'recipient_id': 'person_alex',
                    'report_policy': 'none',
                },
            },
        ),
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_midas', 'label': 'MIDAS', 'kind': 'object'},
                {
                    'id': 'person_alex',
                    'label': 'ALEX',
                    'kind': 'person',
                    'relations': [{'predicate': 'dbp:name', 'object': 'ALEX'}],
                },
            ]
        },
    )

    assert payload['normalized_intents'] == ['pick_object', 'bring_object']
    assert payload['target_selection']['operation'] == 'deliver'
    assert payload['target_selection']['member_ids'] == ['book_midas']
    assert payload['target_selection']['recipient_id'] == 'person_alex'


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
        ],
        'counts': {
            'entities': 1,
            'people': 0,
            'objects': 1,
            'locations': 0,
        },
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
        ],
        'counts': {
            'entities': 2,
            'people': 1,
            'objects': 1,
            'locations': 0,
        },
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


def test_build_planner_request_payload_preserves_authoritative_target_selection() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_grouped',
        user_text='Bring every object from the work table to ALEX.',
        turn_result=_make_result(
            intent='bring_object',
            user_intent={
                'type': 'bring_object',
                'goal_text': 'bring every object from the work table to ALEX',
                'target_selection': {
                    'selection_kind': 'location_members',
                    'operation': 'deliver',
                    'source_location_id': 'work_table',
                    'member_ids': ['cup_1', 'book_1'],
                    'recipient_id': 'person_1',
                    'ordering': 'none',
                    'report_policy': 'final',
                },
            },
        ),
        knowledge_context='',
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'work_table',
        'member_ids': ['cup_1', 'book_1'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'final',
    }


def test_build_planner_request_payload_repairs_invalid_retry_exhausted_selection() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_retry_exhausted_delivery',
        user_text='Bring every object from the kitchen to ALEX and report back.',
        turn_result=_make_result(
            intent='bring_object',
            intent_source='llm_response_route+llm_intent_retry_exhausted',
            user_intent={
                'type': 'bring_object',
                'goal_text': 'bring every object from the kitchen to ALEX and report back',
                'target_selection': {
                    'selection_kind': 'location_members',
                    'operation': 'deliver',
                    'source_location_id': 'kitchen',
                    'member_ids': ['cup_1', 'kitchen_table'],
                    'recipient_id': 'person_alex',
                    'ordering': 'sequential',
                    'report_policy': 'final',
                },
            },
        ),
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'kitchen_table', 'label': 'table', 'kind': 'object'},
                {'id': 'person_alex', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [
                {
                    'id': 'kitchen',
                    'label': 'kitchen',
                    'contains': [{'id': 'cup_1', 'kind': 'object'}],
                },
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'kitchen',
        'member_ids': ['cup_1'],
        'recipient_id': 'person_alex',
        'ordering': 'none',
        'report_policy': 'final',
    }


def test_build_planner_request_payload_derives_grouped_delivery_from_structured_fields() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'object': 'work table',
            'recipient': 'ALEX',
            'goal_text': 'deliver the grounded work table objects',
            'intent_sequence': ['bring_object', 'report_result'],
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_grouped',
        user_text='Bring the work table objects to ALEX.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'book_1', 'label': 'book', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [
                {
                    'id': 'work_table',
                    'label': 'work table',
                    'contains': [
                        {'id': 'cup_1', 'kind': 'object'},
                        {'id': 'book_1', 'kind': 'object'},
                    ],
                }
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'work_table',
        'member_ids': ['book_1', 'cup_1'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'final',
    }


def test_build_planner_request_payload_resolves_grouped_delivery_from_goal_entities() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'bring every object from the work table to ALEX and report back',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_repaired_grouped',
        user_text='Bring every object from the work table to ALEX and report back.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'book_1', 'label': 'book', 'kind': 'object'},
                {
                    'id': 'person_1',
                    'label': 'anonymous_person',
                    'kind': 'person',
                    'relations': [{'predicate': 'dbp:name', 'object': 'ALEX'}],
                },
            ],
            'locations': [
                {
                    'id': 'lab_table',
                    'label': 'table',
                    'aliases': ['work table'],
                    'contains': [
                        {'id': 'cup_1', 'kind': 'object'},
                        {'id': 'book_1', 'kind': 'object'},
                    ],
                }
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'lab_table',
        'member_ids': ['book_1', 'cup_1'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'final',
    }


def test_build_planner_request_payload_resolves_unique_visible_book_and_person() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={'type': 'bring_object'},
    )

    payload = build_planner_request_payload(
        turn_id='turn_unique_book_person',
        user_text='Bring the book to the person.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {
                    'id': 'book_sirmq',
                    'label': 'book',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                },
                {
                    'id': 'anonymous_person_hejea',
                    'label': 'anonymous_person_hejea',
                    'kind': 'person',
                    'class': 'Human',
                    'visible': True,
                },
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'deliver',
        'source_location_id': '',
        'member_ids': ['book_sirmq'],
        'recipient_id': 'anonymous_person_hejea',
        'ordering': 'none',
        'report_policy': 'none',
    }


def test_build_planner_request_payload_assigns_multiple_scene_objects_to_one_recipient() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'scene_targets': ['phone_1', 'phone_2', 'person_1'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_multiple_objects_one_person',
        user_text='Bring the phones to the person.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'phone_1', 'label': 'phone', 'kind': 'object'},
                {'id': 'phone_2', 'label': 'phone', 'kind': 'object'},
                {'id': 'person_1', 'label': 'anonymous_person_1', 'kind': 'person'},
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'deliver',
        'source_location_id': '',
        'member_ids': ['phone_1', 'phone_2'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'none',
    }


def test_build_planner_request_payload_expands_quantified_object_class_targets() -> None:
    result = _make_result(
        intent='pick_object',
        user_intent={
            'type': 'pick_object',
            'goal_text': 'grab all the phones',
            'scene_targets': ['phone'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_all_phones',
        user_text='Now grab all the phones!',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'phone_1', 'label': 'phone', 'kind': 'object', 'class': 'Cellular Telephone'},
                {'id': 'phone_2', 'label': 'phone', 'kind': 'object', 'class': 'Cellular Telephone'},
                {'id': 'apple_1', 'label': 'apple', 'kind': 'object', 'class': 'Apple'},
                {'id': 'person_1', 'label': 'person', 'kind': 'person', 'class': 'Human'},
            ],
        },
    )

    assert payload['scene_targets'] == ['phone_1', 'phone_2']


def test_build_planner_request_payload_allows_grounded_delivery_destination() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'bring the apple to the house',
            'scene_targets': ['apple', 'house'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_apple_house',
        user_text='Can you bring the apple to the house?',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'apple_1', 'label': 'apple', 'kind': 'object', 'class': 'Apple'},
                {'id': 'phone_1', 'label': 'phone', 'kind': 'object', 'class': 'Phone'},
                {
                    'id': 'house_1',
                    'label': 'house',
                    'kind': 'object',
                    'class': 'Container',
                    'relations': [{'predicate': 'oro:contains', 'object': 'phone_1'}],
                },
            ],
            'locations': [
                {
                    'id': 'house_1',
                    'label': 'house',
                    'class': 'Container',
                    'contains': [
                        {'id': 'phone_1', 'label': 'phone', 'kind': 'object'},
                    ],
                },
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'deliver',
        'source_location_id': '',
        'member_ids': ['apple_1'],
        'recipient_id': 'house_1',
        'ordering': 'none',
        'report_policy': 'none',
    }


def test_build_planner_request_payload_expands_those_objects_to_unique_person() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={'type': 'bring_object'},
    )

    payload = build_planner_request_payload(
        turn_id='turn_those_objects',
        user_text='Bring those objects to the person we were talking about.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'apple_1', 'label': 'apple', 'kind': 'object'},
                {'id': 'phone_1', 'label': 'phone', 'kind': 'object'},
                {'id': 'phone_2', 'label': 'phone', 'kind': 'object'},
                {
                    'id': 'anonymous_person_jdefb',
                    'label': 'anonymous_person_jdefb',
                    'kind': 'person',
                },
            ]
        },
    )

    assert payload['target_selection']['member_ids'] == [
        'apple_1',
        'phone_1',
        'phone_2',
    ]
    assert payload['target_selection']['recipient_id'] == 'anonymous_person_jdefb'


def test_build_planner_request_payload_keeps_multiple_people_ambiguous() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={'type': 'bring_object'},
    )

    payload = build_planner_request_payload(
        turn_id='turn_ambiguous_people',
        user_text='Bring the book to the person.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_1', 'label': 'book', 'kind': 'object'},
                {'id': 'person_1', 'label': 'anonymous_person_1', 'kind': 'person'},
                {'id': 'person_2', 'label': 'anonymous_person_2', 'kind': 'person'},
            ],
        },
    )

    assert 'target_selection' not in payload


def test_build_planner_request_payload_allows_direct_navigation_to_person() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'scene_targets': ['anonymous_person_edcca'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_navigate_person',
        user_text='Walk to the person edcca.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {
                    'id': 'anonymous_person_edcca',
                    'label': 'anonymous_person_edcca',
                    'kind': 'person',
                    'class': 'Human',
                }
            ]
        },
    )

    assert payload['target_selection']['member_ids'] == ['anonymous_person_edcca']
    assert payload['target_selection']['operation'] == 'visit'


def test_build_planner_request_payload_allows_direct_navigation_to_location() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'scene_targets': ['kitchen'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_navigate_location',
        user_text='Walk to the kitchen.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object', 'class': 'Cup'},
            ],
            'locations': [
                {
                    'id': 'kitchen',
                    'label': 'kitchen',
                    'contains': [
                        {'id': 'cup_1', 'kind': 'object', 'class': 'Cup'},
                    ],
                },
            ],
        },
    )

    assert payload['target_selection']['member_ids'] == ['kitchen']
    assert payload['target_selection']['operation'] == 'visit'


def test_build_planner_request_payload_resolves_generic_person_target_from_suffix() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'goal_text': 'navigate to the person jdefb and report arrival',
            'scene_targets': ['person'],
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_navigate_person_suffix',
        user_text='Walk to the person jdefb and tell me when you arrive.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {
                    'id': 'anonymous_person_jdefb',
                    'label': 'anonymous_person_jdefb',
                    'kind': 'person',
                    'class': 'Human',
                }
            ]
        },
    )

    assert payload['scene_targets'] == ['person']
    assert payload['target_selection']['member_ids'] == ['anonymous_person_jdefb']


def test_build_planner_request_payload_preserves_goal_for_person_clarification() -> None:
    result = _make_result(
        intent='',
        user_intent={'goal': 'Person eidab!'},
        updated_history=[
            'user:Bring the book to the person.',
            (
                'system:{"planner_dialogue":{"act":"ask_clarification",'
                '"await_user_response":true,"goal_id":"goal_pending",'
                '"context":{"goal_text":"Bring the book to the person.",'
                '"requested_intents":["bring_object"]}}}'
            ),
            'assistant:Which person should I use?',
            'user:Person eidab!',
        ],
    )

    payload = build_planner_request_payload(
        turn_id='turn_clarification_answer',
        user_text='Person eidab!',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'book_sirmq', 'label': 'book', 'kind': 'object'},
                {
                    'id': 'anonymous_person_eidab',
                    'label': 'anonymous_person_eidab',
                    'kind': 'person',
                },
            ],
        },
        active_goal_id='goal_pending',
    )

    assert payload['request_kind'] == 'clarification_answer'
    assert payload['goal_id'] == 'goal_pending'
    assert payload['supersedes_goal_id'] == ''
    assert payload['goal_text'] == (
        'Bring the book to the person. Clarification answer: Person eidab'
    )
    assert payload['normalized_intents'] == ['bring_object']
    assert payload['scene_targets'] == ['anonymous_person_eidab']
    assert payload['target_selection']['member_ids'] == ['book_sirmq']
    assert payload['target_selection']['recipient_id'] == 'anonymous_person_eidab'


def test_build_planner_request_payload_rehydrates_route_conflict_delivery() -> None:
    result = _make_result(
        intent='',
        verbal_ack='I will bring both phones to person lznze.',
        user_intent={},
        updated_history=[
            'user:Can you bring both phones to the person iznze?',
            'assistant:I cannot confirm that person in the current grounded context.',
            'user:I meant person lznze!',
            'assistant:I will bring both phones to person lznze.',
        ],
        route='execution',
    )

    payload = build_planner_request_payload(
        turn_id='turn_route_conflict_answer',
        user_text='I meant person lznze!',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'phone_ojxio', 'label': 'phone', 'kind': 'object'},
                {'id': 'phone_qcgnr', 'label': 'phone', 'kind': 'object'},
                {
                    'id': 'sim_person_lznze',
                    'label': 'sim_person_lznze',
                    'kind': 'person',
                },
            ]
        },
        pending_execution_context={
            'goal_text': 'Bring both phones to the person iznze',
            'intent': 'bring_object',
            'scene_targets': ['phone'],
            'requested_person': 'iznze',
        },
    )

    assert payload['request_kind'] == 'new_goal'
    assert payload['goal_text'] == (
        'Bring both phones to the person sim_person_lznze'
    )
    assert payload['normalized_intents'] == ['bring_object']
    assert payload['scene_targets'] == [
        'phone_ojxio',
        'phone_qcgnr',
        'sim_person_lznze',
    ]
    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'deliver',
        'source_location_id': '',
        'member_ids': ['phone_ojxio', 'phone_qcgnr'],
        'recipient_id': 'sim_person_lznze',
        'ordering': 'none',
        'report_policy': 'none',
    }


def test_should_route_grounded_person_correction_to_preserved_execution() -> None:
    result = _make_result(
        intent='',
        user_intent={},
        route='dialogue',
    )

    assert should_route_intents_through_planner(
        [],
        turn_result=result,
        user_text='I meant person lznze!',
        pending_execution_context={
            'goal_text': 'Bring both phones to the person iznze',
            'intent': 'bring_object',
            'requested_person': 'iznze',
        },
        grounded_context={
            'entities': [
                {
                    'id': 'sim_person_lznze',
                    'label': 'sim_person_lznze',
                    'kind': 'person',
                },
            ],
        },
    )


def test_build_planner_request_payload_keeps_explicit_new_goal_after_clarification() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'request_kind': 'new_goal',
            'goal_text': 'Navigate to the table instead.',
        },
        updated_history=[
            (
                'system:{"planner_dialogue":{"act":"ask_clarification",'
                '"await_user_response":true,"goal_id":"goal_pending",'
                '"context":{"goal_text":"Bring the book.",'
                '"requested_intents":["bring_object"]}}}'
            ),
        ],
    )

    payload = build_planner_request_payload(
        turn_id='turn_new_goal',
        user_text='Navigate to the table instead.',
        turn_result=result,
        knowledge_context='',
        active_goal_id='goal_pending',
    )

    assert payload['request_kind'] == 'new_goal'
    assert payload['goal_id'] != 'goal_pending'
    assert payload['supersedes_goal_id'] == 'goal_pending'
    assert payload['goal_text'] == 'Navigate to the table instead.'


def test_build_planner_request_payload_bounds_ordered_visit_to_grounded_objects() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'goal_text': 'visit every selected object',
            'scene_targets': ['cup_1', 'kitchen', 'person_1', 'book_1'],
            'report_policy': 'per_target',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_visit',
        user_text='Walk to every object and report each arrival.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'book_1', 'label': 'book', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [{'id': 'kitchen', 'label': 'kitchen', 'kind': 'location'}],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'visit',
        'source_location_id': '',
        'member_ids': ['cup_1', 'book_1'],
        'recipient_id': '',
        'ordering': 'sequential',
        'report_policy': 'per_target',
    }


def test_build_planner_request_payload_recovers_quantified_visit_after_route_repair() -> None:
    result = _make_result(
        intent='navigate_to',
        user_intent={
            'type': 'navigate_to',
            'goal': 'Walk to every object and let me know when you get to each one.',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_repaired_visit',
        user_text='Walk to every object and let me know when you get to each one.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'apple_1', 'label': 'ATLAS', 'kind': 'object'},
                {'id': 'book_1', 'label': 'MIDAS', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [
                {
                    'id': 'table_1',
                    'label': 'table',
                    'role': 'support_group',
                    'contains': [
                        {'id': 'apple_1', 'kind': 'object'},
                        {'id': 'book_1', 'kind': 'object'},
                    ],
                }
            ],
        },
    )

    assert payload['scene_targets'] == []
    assert payload['target_selection'] == {
        'selection_kind': 'explicit_members',
        'operation': 'visit',
        'source_location_id': '',
        'member_ids': ['apple_1', 'book_1'],
        'recipient_id': '',
        'ordering': 'sequential',
        'report_policy': 'per_target',
    }


def test_build_planner_request_payload_derives_single_delivery_selection() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'object': 'cup',
            'recipient': 'ALEX',
            'goal_text': 'bring the cup to ALEX',
            'report_policy': 'final',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_delivery',
        user_text='Bring the cup to ALEX and report back.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ]
        },
    )

    assert payload['target_selection']['member_ids'] == ['cup_1']
    assert payload['target_selection']['recipient_id'] == 'person_1'
    assert payload['target_selection']['report_policy'] == 'final'


def test_build_planner_request_payload_treats_summary_request_as_final_report() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'object': 'cup',
            'recipient': 'ALEX',
            'goal_text': 'bring the cup to ALEX and summarize the result',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_delivery_summary',
        user_text='Bring the cup to ALEX and summarize the result.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ]
        },
    )

    assert payload['target_selection']['report_policy'] == 'final'


def test_build_planner_request_payload_uses_structured_scene_target_ids_for_delivery() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'Bring the gold apple to ALEX',
            'scene_targets': [
                {'id': 'gold_apple_1', 'class': 'Apple'},
                {'id': 'person_1', 'class': 'Human'},
            ],
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_structured_delivery',
        user_text='Can you bring that gold apple to the person named ALEX?',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {
                    'id': 'gold_apple_1',
                    'label': 'KAREN',
                    'kind': 'object',
                    'class': 'Apple',
                    'relations': [{'predicate': 'dbp:color', 'object': 'gold'}],
                },
                {
                    'id': 'person_1',
                    'label': 'ALEX',
                    'kind': 'person',
                    'class': 'Human',
                },
            ]
        },
    )

    assert payload['scene_targets'] == ['gold_apple_1', 'person_1']
    assert payload['target_selection']['member_ids'] == ['gold_apple_1']
    assert payload['target_selection']['recipient_id'] == 'person_1'


def test_build_planner_request_payload_matches_unique_object_attributes_from_goal() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'Bring the gold apple to ALEX',
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_attribute_delivery',
        user_text='Can you bring that gold apple to the person named ALEX?',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {
                    'id': 'gold_apple_1',
                    'label': 'KAREN',
                    'kind': 'object',
                    'class': 'Apple',
                    'relations': [
                        {'predicate': 'dbp:name', 'object': 'KAREN'},
                        {'predicate': 'dbp:color', 'object': 'gold'},
                    ],
                },
                {
                    'id': 'person_1',
                    'label': 'ALEX',
                    'kind': 'person',
                    'class': 'Human',
                },
            ]
        },
    )

    assert payload['target_selection']['member_ids'] == ['gold_apple_1']
    assert payload['target_selection']['recipient_id'] == 'person_1'


def test_build_planner_request_payload_expands_scene_target_location_members() -> None:
    result = _make_result(
        intent='bring_object',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'Bring every object from the kitchen to ALEX and report back',
            'scene_targets': ['codex_iiia_kitchen'],
        },
    )
    payload = build_planner_request_payload(
        turn_id='turn_location_target_delivery',
        user_text='Bring every object from the kitchen to ALEX and report back.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'kitchen_table_1', 'label': 'kitchen table', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [
                {
                    'id': 'codex_iiia_kitchen',
                    'label': 'codex_iiia_kitchen',
                    'role': 'navigation_target',
                    'contains': [{'id': 'cup_1', 'kind': 'object'}],
                }
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'codex_iiia_kitchen',
        'member_ids': ['cup_1'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'final',
    }


def test_build_planner_request_payload_salvages_route_repair_location_delivery() -> None:
    result = _make_result(
        intent='bring_object',
        intent_source='llm_response_route_repair',
        user_intent={
            'type': 'bring_object',
            'goal_text': 'Bring every object from the kitchen to ALEX and report back',
        },
    )

    payload = build_planner_request_payload(
        turn_id='turn_route_repair_location_delivery',
        user_text='Bring every object from the kitchen to ALEX and report back.',
        turn_result=result,
        knowledge_context='',
        grounded_context={
            'entities': [
                {'id': 'cup_1', 'label': 'cup', 'kind': 'object'},
                {'id': 'person_1', 'label': 'ALEX', 'kind': 'person'},
            ],
            'locations': [
                {
                    'id': 'kitchen_1',
                    'label': 'kitchen',
                    'contains': [{'id': 'cup_1', 'kind': 'object'}],
                }
            ],
        },
    )

    assert payload['target_selection'] == {
        'selection_kind': 'location_members',
        'operation': 'deliver',
        'source_location_id': 'kitchen_1',
        'member_ids': ['cup_1'],
        'recipient_id': 'person_1',
        'ordering': 'none',
        'report_policy': 'final',
    }


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


def test_ordered_intent_sequence_is_not_polluted_by_summary_intent() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_sequence',
        user_text='walk to the cup, pick it up, sit, and report',
        turn_result=_make_result(
            intent='posture_stand',
            user_intent={
                'type': 'posture_stand',
                'intent_sequence': [
                    'navigate_to',
                    'pick_object',
                    'posture_sit',
                    'report_result',
                ],
            },
        ),
        knowledge_context='',
    )

    assert payload['normalized_intents'] == [
        'navigate_to',
        'pick_object',
        'posture_sit',
        'report_result',
    ]


def test_summary_intent_is_used_when_no_intent_sequence_exists() -> None:
    payload = build_planner_request_payload(
        turn_id='turn_single',
        user_text='stand up',
        turn_result=_make_result(
            intent='posture_stand',
            user_intent={'type': 'posture_stand'},
        ),
        knowledge_context='',
    )

    assert payload['normalized_intents'] == ['posture_stand']


def test_dialogue_turn_id_is_scoped_by_dialogue_uuid() -> None:
    first = dialogue_turn_id('__default__', (1, 2, 3, 4), 1)
    second = dialogue_turn_id('__default__', (9, 8, 7, 6), 1)

    assert first == '__default__:01020304:1'
    assert second == '__default__:09080706:1'
    assert first != second
