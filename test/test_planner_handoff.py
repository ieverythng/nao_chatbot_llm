import sys
import types


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
from chatbot_llm.planner_handoff import _merge_tracked_people
from chatbot_llm.planner_handoff import _scene_summary_payload
from chatbot_llm.planner_handoff import _state_t0_payload


def test_scene_summary_payload_keeps_people_entries() -> None:
    payload = _scene_summary_payload(
        """
        {
          "observer": "myself",
          "backend": "emorobcare_cv",
          "objects": [],
          "people": [
            {"id": "anonymous_person_1", "type": "Human", "source": "person_manager"}
          ]
        }
        """
    )

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

    assert any(
        item.get('id') == 'anonymous_person_1' and item.get('kind') == 'person'
        for item in state_t0['entities']
    )
    assert {'normalized_name': 'anonymous_person_1', 'id': 'anonymous_person_1', 'type': 'Human'} in refs


def test_merge_tracked_people_adds_missing_ids() -> None:
    merged = _merge_tracked_people(
        {'people': [{'id': 'anonymous_person_1', 'label': 'anonymous_person_1'}]},
        tracked_people_ids=('anonymous_person_1', 'anonymous_person_2'),
        tracked_people_ts=0.0,
    )

    merged_ids = {item['id'] for item in merged['people']}
    assert merged_ids == {'anonymous_person_1', 'anonymous_person_2'}
