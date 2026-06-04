import json

from chatbot_llm.planner_handoff import _knowledge_snapshot_payload
from chatbot_llm.planner_handoff import _scene_summary_payload
from chatbot_llm.planner_handoff import _state_t0_payload
from planner_common import project_llm_grounded_context


def _scene_summary_fixture() -> dict:
    return {
        'observer': 'myself',
        'backend': 'emorobcare_cv',
        'objects': [
            {
                'entity_id': 'cup_1',
                'label': 'cup',
                'kb_class': 'Cup',
                'score': 0.92,
                'tracker_id': '',
                'source': 'emorobcare_cv',
                'center_x': 321.0,
                'center_y': 238.0,
                'last_seen_sec': 1777040000.1,
            },
            {
                'entity_id': 'person_1',
                'label': 'person',
                'kb_class': 'Person',
                'score': 0.81,
                'tracker_id': '',
                'source': 'emorobcare_cv',
                'center_x': 186.0,
                'center_y': 202.0,
                'last_seen_sec': 1777040000.2,
            },
        ],
    }


def test_scene_summary_payload_keeps_scene_objects_and_stays_concise() -> None:
    payload = _scene_summary_payload(json.dumps(_scene_summary_fixture()))

    assert payload['schema_version'] == 'scene_summary_v2'
    assert payload['observer'] == 'myself'
    assert payload['backend'] == 'emorobcare_cv'
    assert payload['captured_at_sec'] == 1777040000.2
    assert [item['entity_id'] for item in payload['objects']] == ['cup_1']
    assert [item['id'] for item in payload['people']] == ['person_1']
    assert 'look_at_candidates' not in payload


def test_scene_summary_payload_uses_explicit_people_entries() -> None:
    raw_payload = _scene_summary_fixture()
    raw_payload['people'] = [
        {
            'id': 'face_1',
            'label': 'face_1',
            'type': 'HumanFace',
            'source': 'hri_face_tracker',
            'center_x': 144.0,
            'center_y': 120.0,
            'last_seen_sec': 1777040000.3,
        }
    ]

    payload = _scene_summary_payload(json.dumps(raw_payload))

    people_by_id = {item['id']: item for item in payload['people']}
    assert set(people_by_id) == {'person_1', 'face_1'}
    assert people_by_id['face_1']['type'] == 'HumanFace'
    assert 'look_at_candidates' not in payload


def test_knowledge_snapshot_payload_exposes_structured_entities() -> None:
    scene_summary = _scene_summary_payload(json.dumps(_scene_summary_fixture()))

    payload = _knowledge_snapshot_payload(
        knowledge_context='',
        scene_summary=scene_summary,
    )

    assert payload['schema_version'] == 'knowledge_snapshot_v2'
    assert payload['captured_at_sec'] == 1777040000.2
    assert payload['references'] == [
        {'normalized_name': 'cup', 'id': 'cup_1', 'type': 'Cup'},
        {'normalized_name': 'person', 'id': 'person_1', 'type': 'Person'},
    ]
    assert payload['counts'] == {
        'entities': 2,
        'people': 1,
        'objects': 1,
    }


def test_state_t0_payload_tracks_entity_kind_without_duplicate_candidate_lists() -> None:
    scene_summary = _scene_summary_payload(json.dumps(_scene_summary_fixture()))

    state_t0 = _state_t0_payload(scene_summary)

    assert state_t0['schema_version'] == 'state_t0_v2'
    assert state_t0['captured_at_sec'] == 1777040000.2
    assert state_t0['entity_counts'] == {'entities': 2, 'people': 1, 'objects': 1}
    assert {item['id'] for item in state_t0['entities']} == {'cup_1', 'person_1'}
    entity_kinds = {item['id']: item['kind'] for item in state_t0['entities']}
    assert entity_kinds['person_1'] == 'person'
    assert entity_kinds['cup_1'] == 'object'


def test_handoff_source_payload_projects_to_compact_llm_grounded_context() -> None:
    scene_summary = _scene_summary_payload(json.dumps(_scene_summary_fixture()))
    source_context = {
        'knowledge_snapshot': _knowledge_snapshot_payload(
            knowledge_context='',
            scene_summary=scene_summary,
        ),
        'scene_summary': scene_summary,
        'state_t0': _state_t0_payload(scene_summary),
    }

    compact = project_llm_grounded_context(source_context)

    assert set(compact) == {'entities'}
    assert compact['entities'] == [
        {
            'id': 'cup_1',
            'label': 'cup',
            'kind': 'object',
            'class': 'Cup',
            'visible': True,
        },
        {
            'id': 'person_1',
            'label': 'person',
            'kind': 'person',
            'class': 'Person',
            'visible': True,
        },
    ]
