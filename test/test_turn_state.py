from chatbot_llm.turn_state import build_turn_state
from chatbot_llm.turn_state import resolve_mentioned_subject_ids


def test_turn_state_binds_grounding_and_registry_contract() -> None:
    state = build_turn_state(
        turn_id='default:4',
        utterance='Bring the kitchen cup to Alex',
        history=['user:hello', 'assistant:hello'],
        grounded_context={
            'entities': [
                {
                    'id': 'cup_1',
                    'label': 'kitchen cup',
                    'relations': [{'predicate': 'dbp:name', 'object': 'TITAS'}],
                }
            ]
        },
        active_goal_id='goal_1',
        skill_manifest=[
            {
                'name': 'bring_object',
                'params': ['object', 'recipient'],
                'required_params': ['object', 'recipient'],
            }
        ],
    )

    assert state['turn_state_version'] == 'ts.v1'
    assert state['active_goal']['goal_id'] == 'goal_1'
    assert state['world_state']['grounding_id'].startswith('gc:')
    assert state['available_skills'][0]['required_params'] == ['object', 'recipient']


def test_subject_resolution_uses_only_current_canonical_entities() -> None:
    context = {
        'entities': [
            {
                'id': 'codex_arch_marker',
                'label': 'cube',
                'relations': [{'predicate': 'dbp:name', 'object': 'NOVA'}],
            },
            {'id': 'cup_1', 'label': 'cup', 'relations': []},
        ]
    }

    assert resolve_mentioned_subject_ids('What do you remember about NOVA?', context) == [
        'codex_arch_marker'
    ]
    assert resolve_mentioned_subject_ids('Tell me about a banana', context) == []


def test_subject_resolution_does_not_expand_an_ambiguous_label() -> None:
    context = {
        'entities': [
            {'id': 'cup_1', 'label': 'cup', 'relations': []},
            {'id': 'cup_2', 'label': 'cup', 'relations': []},
        ]
    }

    assert resolve_mentioned_subject_ids('What do you know about the cup?', context) == []
    assert resolve_mentioned_subject_ids('What do you know about cup_2?', context) == ['cup_2']
