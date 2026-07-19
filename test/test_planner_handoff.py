import sys
import types
from types import SimpleNamespace


def _install_ros_message_stubs() -> None:
    if 'hri_actions_msgs.msg' not in sys.modules:
        hri_actions_msgs_mod = types.ModuleType('hri_actions_msgs')
        hri_actions_msgs_msg_mod = types.ModuleType('hri_actions_msgs.msg')

        class _Intent:
            BRING_OBJECT = 'bring_object'
            GRAB_OBJECT = 'grab_object'
            GREET = 'greet'
            GUIDE = 'guide'
            MOVE_TO = 'move_to'
            PERFORM_MOTION = 'perform_motion'
            PLACE_OBJECT = 'place_object'
            PRESENT_CONTENT = 'present_content'
            RAW_USER_INPUT = 'raw_user_input'
            SAY = 'say'
            START_ACTIVITY = 'start_activity'
            STOP_ACTIVITY = 'stop_activity'
            SUSPEND = 'suspend'
            WAKEUP = 'wakeup'
            MODALITY_SPEECH = 'speech'
            UNKNOWN_AGENT = 'unknown_agent'

            def __init__(self) -> None:
                self.intent = ''
                self.source = ''
                self.modality = ''
                self.confidence = 0.0
                self.priority = 0
                self.data = ''

        hri_actions_msgs_msg_mod.Intent = _Intent
        hri_actions_msgs_mod.msg = hri_actions_msgs_msg_mod
        sys.modules['hri_actions_msgs'] = hri_actions_msgs_mod
        sys.modules['hri_actions_msgs.msg'] = hri_actions_msgs_msg_mod

    if 'std_msgs.msg' not in sys.modules:
        std_msgs_mod = types.ModuleType('std_msgs')
        std_msgs_msg_mod = types.ModuleType('std_msgs.msg')

        class _String:
            pass

        std_msgs_msg_mod.String = _String
        std_msgs_mod.msg = std_msgs_msg_mod
        sys.modules['std_msgs'] = std_msgs_mod
        sys.modules['std_msgs.msg'] = std_msgs_msg_mod


_install_ros_message_stubs()

from chatbot_llm.planner_handoff import _kb_references_from_scene
from chatbot_llm.planner_handoff import _grounded_context_or_refresh
from chatbot_llm.planner_handoff import _planner_handoff_observability
from chatbot_llm.planner_handoff import _scene_summary_payload
from chatbot_llm.planner_handoff import _state_t0_payload


def test_planner_handoff_observability_marks_repaired_execution_without_intent() -> None:
    evidence = _planner_handoff_observability(
        SimpleNamespace(
            route='execution',
            intent='',
            intent_source='llm_response_route_repair',
        ),
        {
            'goal_id': 'goal_head_1',
            'dialogue_turn_id': 'turn_head_1',
            'normalized_intents': [],
        },
    )

    assert evidence == {
        'route': 'execution',
        'intent': '',
        'intent_source': 'llm_response_route_repair',
        'route_repaired': True,
        'normalized_intents': [],
        'intent_gap': True,
        'goal_id': 'goal_head_1',
        'dialogue_turn_id': 'turn_head_1',
    }


def test_planner_handoff_observability_accepts_structured_execution_intent() -> None:
    evidence = _planner_handoff_observability(
        SimpleNamespace(
            route='execution',
            intent='head_look_left',
            intent_source='llm_response_route+llm_intent',
        ),
        {
            'goal_id': 'goal_head_2',
            'dialogue_turn_id': 'turn_head_2',
            'normalized_intents': ['head_look_left', 'report_result'],
        },
    )

    assert evidence['route_repaired'] is False
    assert evidence['intent_gap'] is False
    assert evidence['normalized_intents'] == [
        'head_look_left',
        'report_result',
    ]


def test_scene_summary_payload_keeps_people_entries() -> None:
    payload = _scene_summary_payload(
        """
        {
          "observer": "myself",
          "backend": "emorobcare_cv",
          "captured_at_sec": 1777040000.0,
          "objects": [],
          "people": [
            {"id": "anonymous_person_1", "type": "Human", "source": "person_manager"}
          ]
        }
        """
    )

    assert payload['schema_version'] == 'scene_summary_v2'
    assert payload['captured_at_sec'] == 0.0
    assert payload['people'] == [
        {
            'id': 'anonymous_person_1',
            'label': 'anonymous_person_1',
            'type': 'Human',
            'source': 'person_manager',
            'score': 0.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'last_seen_sec': 0.0,
        }
    ]


def test_state_t0_and_references_include_people_ids() -> None:
    scene_summary = {
        'observer': 'myself',
        'backend': 'emorobcare_cv',
        'objects': [{'entity_id': 'book_1', 'label': 'book_1', 'kb_class': 'Book'}],
        'people': [{'id': 'anonymous_person_1', 'label': 'anonymous_person_1', 'type': 'Human'}],
    }

    state_t0 = _state_t0_payload(scene_summary)
    refs = _kb_references_from_scene(scene_summary)

    assert state_t0['schema_version'] == 'state_t0_v2'
    assert 'entity_counts' not in state_t0
    assert any(
        item.get('id') == 'anonymous_person_1' and item.get('kind') == 'person'
        for item in state_t0['entities']
    )
    assert {'normalized_name': 'anonymous_person_1', 'id': 'anonymous_person_1', 'type': 'Human'} in refs


def test_grounded_context_projection_prefers_structured_rows_over_text_fallback() -> None:
    raw_scene = _scene_summary_payload(
        """
        {
          "observer": "myself",
          "backend": "emorobcare_cv",
          "objects": [
            {
              "entity_id": "apple_armzq",
              "label": "apple_armzq",
              "kb_class": "Apple",
              "source": "emorobcare_cv",
              "center_x": 20.0,
              "center_y": 30.0,
              "last_seen_sec": 1777040000.0
            }
          ]
        }
        """
    )

    from planner_common import project_llm_grounded_context

    projected = project_llm_grounded_context(
        {
            'knowledge_snapshot': {},
            'scene_summary': raw_scene,
            'state_t0': _state_t0_payload(raw_scene),
        },
        knowledge_rows=[
            {'entity': 'apple_armzq', 'predicate': 'rdf:type', 'object': 'dbr:Apple'}
        ],
    )

    assert projected['entities'] == [
        {
            'id': 'apple_armzq',
            'label': 'apple',
            'kind': 'object',
            'class': 'Apple',
            'visible': True,
        }
    ]
    assert 'state_t0' not in projected
def test_empty_turn_grounding_refreshes_before_planner_handoff() -> None:
    refreshed = {'entities': [{'id': 'person_alex', 'kind': 'person'}]}

    assert _grounded_context_or_refresh(
        {'entities': [], 'counts': {'entities': 0}},
        lambda: refreshed,
    ) == refreshed


def test_nonempty_turn_grounding_remains_authoritative() -> None:
    current = {'entities': [{'id': 'person_current', 'kind': 'person'}]}

    assert _grounded_context_or_refresh(
        current,
        lambda: {'entities': [{'id': 'person_stale', 'kind': 'person'}]},
    ) is current
