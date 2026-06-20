from chatbot_llm.irr_contract import guard_irr_decision


def _turn_state() -> dict:
    return {
        'world_state': {
            'grounding_id': 'gc:abc',
            'entities': [
                {'id': 'cup_1', 'label': 'cup'},
                {'id': 'person_1', 'label': 'Alex'},
            ],
        },
        'available_skills': [
            {
                'name': 'bring_object',
                'params': ['object', 'recipient'],
                'required_params': ['object', 'recipient'],
            }
        ],
    }


def test_guard_accepts_grounded_execution_and_role_bindings() -> None:
    guarded = guard_irr_decision(
        {
            'route': 'execution',
            'route_reason': 'immediate_supported_request',
            'confidence': 0.94,
            'intent': {
                'type': 'bring_object',
                'goal_text': 'bring the cup to Alex',
                'request_kind': 'new_goal',
                'arguments': {'object': 'cup_1', 'recipient': 'person_1'},
            },
            'response': {'text': 'I will bring the cup to Alex.', 'style': 'acknowledgement'},
            'planner_handoff': {'requested': True},
            'evidence_used': {
                'grounding_id': 'gc:abc',
                'entity_ids': ['cup_1', 'person_1'],
            },
            'safety_flags': [],
        },
        turn_state=_turn_state(),
        fallback_response='fallback',
    )

    assert guarded.route == 'execution'
    assert guarded.planner_handoff_requested is True
    assert guarded.violations == ()
    assert guarded.user_intent()['object'] == 'cup_1'
    assert guarded.user_intent()['recipient'] == 'person_1'


def test_guard_downgrades_missing_target_to_one_clarification() -> None:
    guarded = guard_irr_decision(
        {
            'route': 'execution',
            'route_reason': 'bring_request',
            'confidence': 0.7,
            'intent': {
                'type': 'bring_object',
                'goal_text': 'bring the cup',
                'arguments': {'object': 'cup_1'},
            },
            'response': {'text': 'I will bring it.', 'style': 'acknowledgement'},
            'planner_handoff': {'requested': True},
            'evidence_used': {'grounding_id': 'gc:abc', 'entity_ids': ['cup_1']},
            'safety_flags': [],
        },
        turn_state=_turn_state(),
        fallback_response='fallback',
    )

    assert guarded.route == 'dialogue'
    assert guarded.planner_handoff_requested is False
    assert guarded.response_style == 'clarification'
    assert guarded.response_text == 'Which recipient should I use?'
    assert 'missing_required_arguments' in guarded.violations


def test_guard_removes_non_execution_handoff_and_action_commitment() -> None:
    guarded = guard_irr_decision(
        {
            'route': 'dialogue',
            'route_reason': 'future_discussion',
            'confidence': 0.8,
            'intent': {'type': 'help', 'request_kind': 'none'},
            'response': {'text': 'I will navigate there now.', 'style': 'answer'},
            'planner_handoff': {'requested': True},
            'evidence_used': {},
            'safety_flags': [],
        },
        turn_state=_turn_state(),
        fallback_response='fallback',
    )

    assert guarded.route == 'dialogue'
    assert guarded.planner_handoff_requested is False
    assert guarded.response_text == 'Could you clarify what you would like me to do?'
    assert set(guarded.violations) == {
        'dialogue_action_commitment',
        'handoff_requested_for_non_execution',
    }
