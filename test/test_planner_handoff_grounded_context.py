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


def test_scene_summary_payload_drops_stale_tracked_people() -> None:
    raw_payload = _scene_summary_fixture()
    raw_payload['people'] = [
        {'id': 'person_old', 'type': 'Human', 'last_seen_sec': 1777039990.0},
        {'id': 'person_current', 'type': 'Human', 'last_seen_sec': 1777040000.3},
        {'id': 'person_without_timestamp', 'type': 'Human'},
    ]

    payload = _scene_summary_payload(json.dumps(raw_payload))

    assert [person['id'] for person in payload['people']] == [
        'person_current',
        'person_without_timestamp',
        'person_1',
    ]


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
    assert 'counts' not in payload


def test_state_t0_payload_tracks_entity_kind_without_duplicate_candidate_lists() -> None:
    scene_summary = _scene_summary_payload(json.dumps(_scene_summary_fixture()))

    state_t0 = _state_t0_payload(scene_summary)

    assert state_t0['schema_version'] == 'state_t0_v2'
    assert state_t0['captured_at_sec'] == 1777040000.2
    assert 'entity_counts' not in state_t0
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

    assert set(compact) == {'entities', 'counts'}
    assert compact['counts'] == {
        'entities': 2,
        'people': 1,
        'objects': 1,
        'locations': 0,
    }
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
            'label': 'person_1',
            'kind': 'person',
            'class': 'Person',
            'visible': True,
        },
    ]


def test_handoff_grounded_context_forwards_knowledge_rows() -> None:
    from types import SimpleNamespace

    class Node:
        def create_publisher(self, *args, **kwargs):
            return object()

        def create_subscription(self, *args, **kwargs):
            return object()

    from chatbot_llm.planner_handoff import PlannerHandoff

    config = SimpleNamespace(
        planner_request_topic='/planner/request',
        planner_scene_summary_topic='/scene/summary',
    )
    handoff = PlannerHandoff(Node(), config, trace=lambda *args, **kwargs: None)

    compact = handoff.grounded_context(
        '',
        knowledge_rows=[
            {'entity': 'cup_1', 'predicate': 'rdf:type', 'object': 'dbr:Cup'},
            {'entity': 'cup_1', 'predicate': 'oro:isOn', 'object': 'table_1'},
        ],
    )

    assert compact['entities'] == [
        {
            'id': 'cup_1',
            'label': 'cup_1',
            'kind': 'object',
            'class': 'Cup',
            'visible': True,
            'relations': [{'predicate': 'oro:isOn', 'object': 'table_1'}],
        }
    ]
    assert compact['locations'] == [
        {
            'id': 'table_1',
            'label': 'table_1',
            'role': 'support_group',
            'member_count': 1,
            'object_count': 1,
            'person_count': 0,
            'contains': [
                {
                    'id': 'cup_1',
                    'label': 'cup_1',
                    'kind': 'object',
                    'class': 'Cup',
                    'relation': 'oro:isOn',
                }
            ],
        }
    ]


def test_handoff_filters_inactive_anonymous_people_after_tracker_update() -> None:
    from types import SimpleNamespace

    from chatbot_llm.planner_handoff import PlannerHandoff

    class Node:
        def create_publisher(self, *args, **kwargs):
            return object()

        def create_subscription(self, *args, **kwargs):
            return object()

    config = SimpleNamespace(
        planner_request_topic='/planner/request',
        planner_scene_summary_topic='/scene/summary',
    )
    handoff = PlannerHandoff(Node(), config, trace=lambda *args, **kwargs: None)
    handoff._on_tracked_persons(
        SimpleNamespace(ids=['anonymous_person_current', 'sim_person_current'])
    )

    compact = handoff.grounded_context(
        '',
        knowledge_rows=[
            {'entity': 'anonymous_person_old', 'predicate': 'rdf:type', 'object': 'Human'},
            {'entity': 'sim_person_old', 'predicate': 'rdf:type', 'object': 'Human'},
            {
                'entity': 'anonymous_person_current',
                'predicate': 'rdf:type',
                'object': 'Human',
            },
            {'entity': 'sim_person_current', 'predicate': 'rdf:type', 'object': 'Human'},
            {'entity': 'fixture_person', 'predicate': 'rdf:type', 'object': 'Human'},
        ],
    )

    assert {item['id'] for item in compact['entities']} == {
        'anonymous_person_current',
        'fixture_person',
        'sim_person_current',
    }
