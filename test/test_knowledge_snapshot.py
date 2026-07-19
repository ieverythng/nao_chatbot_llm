from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.knowledge_snapshot import build_scene_context
from chatbot_llm.knowledge_snapshot import build_scene_digest
from chatbot_llm.knowledge_snapshot import build_grounded_context_block
from chatbot_llm.knowledge_snapshot import extract_scene_memory_entry
from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot
from chatbot_llm.knowledge_snapshot import resolve_knowledge_snapshot_settings
from chatbot_llm.knowledge_snapshot_client import _annotate_constant_subject_rows
from chatbot_llm.knowledge_snapshot_client import KnowledgeSnapshotClient
from chatbot_llm.knowledge_snapshot_client import _subject_query_groups


def make_config() -> ChatbotConfig:
    return ChatbotConfig(
        server_url='http://localhost:11434/api/chat',
        model='llama3.2:1b',
        api_key='',
        system_prompt='You are {robot_name}.',
        enabled=True,
        intent_model='llama3.2:1b',
        request_timeout_sec=20.0,
        first_request_timeout_sec=60.0,
        intent_request_timeout_sec=10.0,
        context_window_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        think=False,
        response_max_tokens=64,
        intent_max_tokens=64,
        preflight_enabled=True,
        preflight_required=False,
        preflight_timeout_sec=45.0,
        preflight_attempts=1,
        preflight_realistic_enabled=False,
        preflight_keepalive_interval_sec=0.0,
        fallback_response='fallback',
        max_history_messages=20,
        scene_memory_turns=4,
        robot_name='NAO',
        persona_prompt_path='',
        response_prompt_addendum='Respond briefly.',
        intent_prompt_addendum='Infer intent.',
        environment_description='No specific objects described.',
        response_schema={'type': 'object'},
        intent_schema={'type': 'object'},
        planner_multi_step_heuristics={
            'coordination_markers': [' and then ', ' then '],
            'action_hint_tokens': ['stand', 'sit', 'look', 'move', 'head'],
        },
        identity_reminder_every_n_turns=6,
        intent_detection_mode='llm',
        turn_pipeline_mode='response_first',
        prompt_pack_path='',
        use_skill_catalog=False,
        skill_catalog_packages=[],
        skill_catalog_max_entries=0,
        skill_catalog_max_chars=0,
        planner_mode_enabled=False,
        planner_request_topic='/planner/request',
        planner_request_intent='planner_request',
        planner_scene_summary_topic='/scene/summary',
        grounded_context_digest_enabled=True,
        grounded_context_include_state_t0=False,
        turn_trace_enabled=True,
        turn_trace_topic='/chatbot_llm/turn_trace',
        knowledge_enabled=False,
        knowledge_query_service_name='/kb/query',
        knowledge_query_timeout_sec=0.5,
        knowledge_default_query_groups=[
            'myself sees ?entity && ?entity rdf:type ?type',
        ],
        knowledge_default_patterns=['myself sees ?entity', '?entity rdf:type ?type'],
        knowledge_default_vars=['?entity', '?type'],
        knowledge_default_models=[],
        knowledge_max_results=40,
        knowledge_max_chars=3000,
    )


def test_resolve_knowledge_snapshot_settings_uses_role_overrides():
    settings = resolve_knowledge_snapshot_settings(
        '{"knowledge_snapshot":{"enabled":true,"patterns":["?person likes ?thing"],'
        '"vars":["?person","?thing"],"models":["all"],"max_results":3,"max_chars":120}}',
        make_config(),
    )

    assert settings == KnowledgeSnapshotSettings(
        enabled=True,
        query_groups=[
            ['myself sees ?entity', '?entity rdf:type ?type'],
        ],
        patterns=['?person likes ?thing'],
        query_vars=['?person', '?thing'],
        models=['all'],
        max_results=3,
        max_chars=120,
    )


def test_resolve_knowledge_snapshot_settings_falls_back_on_invalid_json():
    settings = resolve_knowledge_snapshot_settings('{not json}', make_config())

    assert settings.enabled is False
    assert settings.query_groups == [
        ['myself sees ?entity', '?entity rdf:type ?type'],
    ]
    assert settings.patterns == ['myself sees ?entity', '?entity rdf:type ?type']
    assert settings.query_vars == ['?entity', '?type']


def test_subject_query_groups_add_direct_kb_subject_queries():
    groups = _subject_query_groups('What do you remember about codex_arch_marker?')

    assert groups == [['codex_arch_marker ?predicate ?object']]


def test_subject_query_groups_can_resolve_explicit_dbp_name_tokens():
    groups = _subject_query_groups('What do you know about NOVA?')

    assert groups == [['?entity dbp:name NOVA', '?entity ?predicate ?object']]


def test_subject_query_groups_do_not_expand_broad_scene_questions():
    assert _subject_query_groups('What can you see now?') == []


def test_constant_subject_rows_are_annotated_for_grounded_projection():
    rows = _annotate_constant_subject_rows(
        [{'predicate': 'dbp:name', 'object': 'NOVA'}],
        'codex_arch_marker',
    )

    assert rows == [
        {'entity': 'codex_arch_marker', 'predicate': 'dbp:name', 'object': 'NOVA'}
    ]


def test_snapshot_queries_unseen_spatial_targets_for_location_metadata():
    class FakeQueryClient:
        service_name = '/kb/query'

        def __init__(self):
            self.calls = []

        def query_rows(self, *, patterns, query_vars, **_kwargs):
            self.calls.append((patterns, query_vars))
            if patterns == ['myself sees ?entity', '?entity oro:isAt ?object']:
                return [{'entity': 'cup_1', 'object': 'codex_iiia_kitchen'}]
            if patterns == ['codex_iiia_kitchen ?predicate ?object']:
                return [
                    {'predicate': 'dbp:name', 'object': 'kitchen'},
                    {'predicate': 'rdf:type', 'object': 'Room'},
                ]
            return []

    client = KnowledgeSnapshotClient.__new__(KnowledgeSnapshotClient)
    client._query_client = FakeQueryClient()
    client.last_rows = ()
    settings = KnowledgeSnapshotSettings(
        enabled=True,
        query_groups=[['myself sees ?entity', '?entity oro:isAt ?object']],
        patterns=[],
        query_vars=['?entity', '?predicate', '?object'],
        models=['default'],
        max_results=20,
        max_chars=2000,
    )

    client.fetch_snapshot(settings, user_text='Bring every object from the kitchen.')

    assert client._query_client.calls == [
        (
            ['myself sees ?entity', '?entity oro:isAt ?object'],
            ['?entity', '?object'],
        ),
        (
            ['codex_iiia_kitchen ?predicate ?object'],
            ['?predicate', '?object'],
        ),
    ]
    assert {'entity': 'codex_iiia_kitchen', 'predicate': 'dbp:name', 'object': 'kitchen'} in client.last_rows


def test_format_knowledge_snapshot_formats_triples_and_truncates():
    settings = KnowledgeSnapshotSettings(
        enabled=True,
        query_groups=[],
        patterns=['?s ?p ?o'],
        query_vars=['?s', '?p', '?o'],
        models=[],
        max_results=2,
        max_chars=29,
    )

    snapshot = format_knowledge_snapshot(
        '[{"s":"mug","p":"isOn","o":"table"},{"s":"book","p":"isOn","o":"shelf"}]',
        settings,
    )

    assert snapshot == 'mug is on table\n...'


def test_format_knowledge_snapshot_adds_person_face_summary_and_humanizes_triples():
    settings = KnowledgeSnapshotSettings(
        enabled=True,
        query_groups=[],
        patterns=['?s ?p ?o'],
        query_vars=['?s', '?p', '?o'],
        models=[],
        max_results=4,
        max_chars=400,
    )

    snapshot = format_knowledge_snapshot(
        (
            '[{"s":"person_1","p":"rdf:type","o":"Person"},'
            '{"s":"face_1","p":"isVisible","o":"true"},'
            '{"s":"mug","p":"isOn","o":"table"}]'
        ),
        settings,
    )

    assert 'Detected person/face-related entities right now: person 1, face 1' in snapshot
    assert 'Scene facts:' in snapshot
    assert '- person 1 is a Person' in snapshot
    assert '- face 1 is visible true' in snapshot
    assert '- mug is on table' in snapshot


def test_format_knowledge_snapshot_summarizes_entities_seen_by_robot():
    settings = KnowledgeSnapshotSettings(
        enabled=True,
        query_groups=[
            ['myself sees ?entity', '?entity rdf:type ?type'],
        ],
        patterns=['myself sees ?entity', '?entity rdf:type ?type'],
        query_vars=['?entity', '?type'],
        models=[],
        max_results=20,
        max_chars=500,
    )

    snapshot = format_knowledge_snapshot(
        (
            '[{"entity":"book_bkjwb","type":"dbr:Book"},'
            '{"entity":"book_bkjwb","type":"Artifact"},'
            '{"entity":"anonymous_person_dhgef","type":"Human"},'
            '{"entity":"anonymous_person_dhgef","type":"foaf:Person"}]'
        ),
        settings,
    )

    assert 'Entities currently seen by the robot: book bkjwb (Book), anonymous person dhgef (Human, Person)' in snapshot
    assert '- book bkjwb is currently classified as Book' in snapshot
    assert '- anonymous person dhgef is currently classified as Human, Person' in snapshot


def test_extract_scene_memory_entry_prefers_summary_line():
    snapshot = (
        'Entities currently seen by the robot: book bkjwb (Book)\n'
        'Scene facts:\n'
        '- book bkjwb is currently classified as Book'
    )

    assert (
        extract_scene_memory_entry(snapshot)
        == 'Entities currently seen by the robot: book bkjwb (Book)'
    )


def test_build_scene_context_includes_current_scene_and_recent_memory():
    context = build_scene_context(
        'Entities currently seen by the robot: anonymous person dhgef (Human)',
        recent_scene_memory=[
            'Entities currently seen by the robot: book bkjwb (Book)',
        ],
    )

    assert 'Current grounded scene:' in context
    assert 'Recent scene memory from previous turns:' in context
    assert '- Entities currently seen by the robot: book bkjwb (Book)' in context


def test_build_grounded_context_block_summarizes_entities_and_targets():
    block = build_grounded_context_block(
        {
            'entities': [
                {
                    'id': 'book_qibia',
                    'label': 'book',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                },
                {
                    'id': 'anonymous_person_gjjbd',
                    'label': None,
                    'kind': 'person',
                    'class': 'Human',
                    'visible': True,
                },
            ],
        }
    )

    assert block.startswith('Grounded context JSON:')
    assert '"id": "book_qibia"' in block
    assert '"id": "anonymous_person_gjjbd"' in block
    assert '"label": null' in block


def test_build_grounded_context_block_renders_authoritative_relations():
    block = build_grounded_context_block(
        {
            'entities': [
                {
                    'id': 'book_znpbs',
                    'label': 'book',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                    'relations': [
                        {'predicate': 'dbp:name', 'object': 'TITAS'},
                        {'predicate': 'dbp:color', 'object': 'blue'},
                    ],
                },
            ],
        }
    )

    assert '"relations": [' in block
    assert '"predicate": "dbp:name"' in block
    assert '"object": "TITAS"' in block
    assert '"predicate": "dbp:color"' in block


def test_grounded_context_projection_collapses_movable_object_spatial_clutter_to_place():
    context = {
        'entities': [
            {
                'id': 'book_onigw',
                'label': 'book',
                'kind': 'object',
                'class': 'Book',
                'visible': True,
                'relations': [
                    {'predicate': 'oro:isAt', 'object': 'Park'},
                    {'predicate': 'oro:isAt', 'object': 'cup_cqvmy'},
                    {'predicate': 'oro:contains', 'object': 'cup_cqvmy'},
                ],
            },
            {
                'id': 'cup_cqvmy',
                'label': 'cup',
                'kind': 'object',
                'class': 'Tableware',
                'visible': True,
                'relations': [
                    {'predicate': 'dbp:color', 'object': 'Black'},
                    {'predicate': 'oro:isAt', 'object': 'book_onigw'},
                    {'predicate': 'oro:isOn', 'object': 'book_onigw'},
                    {'predicate': 'oro:contains', 'object': 'book_onigw'},
                ],
            },
        ],
    }

    block = build_grounded_context_block(context)
    digest = build_scene_digest(context)

    assert '"id": "cup_cqvmy"' in block
    assert '"object": "Park"' in block
    assert '"object": "book_onigw"' not in block
    assert '"object": "cup_cqvmy"' not in block
    assert 'cup x1 [black, at Park]' in digest
    assert 'book x1 [at Park]' in digest
    assert 'book onigw' not in digest


def _apple(entity_id: str, color: str) -> dict:
    return {
        'id': entity_id,
        'label': 'apple',
        'kind': 'object',
        'class': 'Apple',
        'visible': True,
        'relations': [{'predicate': 'dbp:color', 'object': color}],
    }


def test_build_scene_digest_counts_people_and_attribute_qualifies_duplicates():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'sim_person_isdki',
                    'label': None,
                    'kind': 'person',
                    'class': 'Human',
                    'visible': True,
                },
                _apple('apple_1', 'yellow'),
                _apple('apple_2', 'green'),
                _apple('apple_3', 'purple'),
                _apple('apple_4', 'red'),
                {
                    'id': 'book_1',
                    'label': 'book',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                    'relations': [{'predicate': 'oro:isOn', 'object': 'shelf_1'}],
                },
            ],
        }
    )

    assert digest.startswith('Scene digest (authoritative for this turn;')
    assert 'People (1): sim_person_isdki' in digest
    assert 'Objects (5):' in digest
    assert 'apple x4 [' in digest
    for color in ('yellow', 'green', 'purple', 'red'):
        assert color in digest
    assert 'book x1 [on shelf 1]' in digest


def test_build_scene_digest_includes_stable_name_before_color_and_support():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_probe_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                    'relations': [
                        {'predicate': 'dbp:name', 'object': 'TITAS'},
                        {'predicate': 'dbp:color', 'object': 'gold'},
                        {'predicate': 'oro:isOn', 'object': 'codex_probe_table'},
                    ],
                }
            ],
        }
    )

    assert 'cup x1 [named TITAS, gold, on codex probe table]' in digest


def test_build_scene_digest_renders_location_groups_from_compact_context():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_kitchen',
                    'label': 'kitchen',
                    'kind': 'object',
                    'class': 'Room',
                    'visible': True,
                },
                {
                    'id': 'codex_kitchen_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                    'relations': [
                        {'predicate': 'dbp:name', 'object': 'TITAS'},
                        {'predicate': 'oro:isIn', 'object': 'codex_kitchen'},
                    ],
                },
            ],
            'locations': [
                {
                    'id': 'codex_kitchen',
                    'label': 'kitchen',
                    'contains': [
                        {
                            'id': 'codex_kitchen_cup',
                            'label': 'cup',
                            'kind': 'object',
                            'class': 'Cup',
                            'relation': 'oro:isIn',
                        }
                    ],
                }
            ],
        }
    )

    assert 'Objects (1):' in digest
    assert 'cup x1 [named TITAS, in codex kitchen]' in digest
    assert 'Locations: kitchen contains cup' in digest
    assert 'room' not in digest.lower()


def test_build_scene_digest_filters_meta_support_and_people_from_location_members():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_work_table',
                    'label': 'work table',
                    'kind': 'object',
                    'class': 'Table',
                    'visible': True,
                },
                {
                    'id': 'localized_marker',
                    'label': 'localized marker',
                    'kind': 'object',
                    'class': 'cyc:SpatialThing-Localized',
                    'visible': True,
                    'relations': [
                        {'predicate': 'rdf:type', 'object': 'owl:Thing'},
                        {'predicate': 'oro:isOn', 'object': 'codex_work_table'},
                    ],
                },
                {
                    'id': 'codex_manual',
                    'label': 'manual',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                    'relations': [
                        {'predicate': 'oro:isOn', 'object': 'codex_work_table'},
                    ],
                },
                {
                    'id': 'codex_recipient_person',
                    'label': 'ALEX',
                    'kind': 'person',
                    'class': 'Human',
                    'visible': True,
                    'relations': [
                        {'predicate': 'oro:isAt', 'object': 'codex_work_table'},
                    ],
                },
                {
                    'id': 'codex_lab_bench',
                    'label': 'bench',
                    'kind': 'object',
                    'class': 'Bench',
                    'visible': True,
                },
            ],
        }
    )

    assert 'Objects (3): table x1; book x1' in digest
    assert 'bench x1' in digest
    assert 'codex work table' in digest
    assert 'Locations: work table contains manual' in digest
    assert 'spatial thing localized' not in digest.lower()
    assert 'localized marker' not in digest
    assert 'work table contains' in digest.lower()
    assert 'contains ALEX' not in digest


def test_build_scene_digest_does_not_promote_people_or_objects_to_locations():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'anonymous_person_dcdbc',
                    'label': 'anonymous_person_dcdbc',
                    'kind': 'person',
                    'class': 'Human',
                    'visible': True,
                },
                {
                    'id': 'book_nwred',
                    'label': 'book',
                    'kind': 'object',
                    'class': 'Book',
                    'visible': True,
                    'relations': [
                        {'predicate': 'oro:isAt', 'object': 'anonymous_person_dcdbc'},
                        {'predicate': 'oro:contains', 'object': 'anonymous_person_dcdbc'},
                    ],
                },
                {
                    'id': 'cup_ohjps',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Tableware',
                    'visible': True,
                    'relations': [
                        {'predicate': 'oro:contains', 'object': 'anonymous_person_dcdbc'},
                    ],
                },
            ],
            'locations': [
                {
                    'id': 'book_nwred',
                    'label': 'book',
                    'class': 'Book',
                    'contains': [
                        {
                            'id': 'anonymous_person_dcdbc',
                            'label': 'anonymous_person_dcdbc',
                            'kind': 'object',
                            'relation': 'oro:contains',
                        }
                    ],
                },
                {
                    'id': 'anonymous_person_dcdbc',
                    'label': 'anonymous_person_dcdbc',
                    'class': 'Human',
                    'contains': [
                        {
                            'id': 'book_nwred',
                            'label': 'book',
                            'kind': 'object',
                            'class': 'Book',
                            'relation': 'oro:isAt',
                        }
                    ],
                },
            ],
        }
    )

    assert 'Locations:' not in digest
    assert 'book contains' not in digest
    assert 'anonymous person dcdbc contains' not in digest


def test_build_scene_digest_keeps_objects_with_domain_and_spatial_rdf_types():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_probe_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                    'relations': [
                        {'predicate': 'rdf:type', 'object': 'cyc:SpatialThing-Localized'},
                        {'predicate': 'rdf:type', 'object': 'Cup'},
                        {'predicate': 'dbp:name', 'object': 'TITAS'},
                    ],
                },
                {
                    'id': 'codex_probe_book',
                    'label': 'book',
                    'kind': 'object',
                    'class': '',
                    'visible': True,
                    'relations': [
                        {'predicate': 'rdf:type', 'object': 'cyc:SpatialThing-Localized'},
                        {'predicate': 'rdf:type', 'object': 'Book'},
                    ],
                },
            ],
        }
    )

    assert 'Objects (2):' in digest
    assert 'cup x1 [named TITAS]' in digest
    assert 'book x1' in digest


def test_build_scene_digest_filters_compact_location_members():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_kitchen_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                }
            ],
            'locations': [
                {
                    'id': 'codex_kitchen',
                    'label': 'kitchen',
                    'contains': [
                        {
                            'id': 'codex_kitchen',
                            'label': 'kitchen',
                            'kind': 'object',
                            'class': 'Room',
                        },
                        {
                            'id': 'codex_kitchen_table',
                            'label': 'kitchen table',
                            'kind': 'object',
                            'class': 'Table',
                        },
                        {
                            'id': 'codex_recipient_person',
                            'label': 'ALEX',
                            'kind': 'person',
                            'class': 'Human',
                        },
                        {
                            'id': 'codex_kitchen_cup',
                            'label': 'cup',
                            'kind': 'object',
                            'class': 'Cup',
                        },
                    ],
                }
            ],
        }
    )

    assert 'Objects (1): cup x1' in digest
    assert 'Locations: kitchen contains cup' in digest
    assert 'kitchen table' not in digest
    assert 'ALEX' not in digest


def test_build_scene_digest_deduplicates_entity_ids_and_merges_relations():
    digest = build_scene_digest(
        {
            'entities': [
                {
                    'id': 'codex_probe_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                    'relations': [{'predicate': 'dbp:name', 'object': 'TITAS'}],
                },
                {
                    'id': 'codex_probe_cup',
                    'label': '',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': True,
                    'relations': [
                        {'predicate': 'dbp:color', 'object': 'gold'},
                        {'predicate': 'oro:isOn', 'object': 'codex_probe_table'},
                    ],
                },
                {
                    'id': 'stale_cup',
                    'label': 'cup',
                    'kind': 'object',
                    'class': 'Cup',
                    'visible': False,
                    'relations': [{'predicate': 'dbp:color', 'object': 'red'}],
                },
            ],
        }
    )

    assert 'Objects (1): cup x1 [named TITAS, gold, on codex probe table]' in digest
    assert 'stale' not in digest
    assert 'red' not in digest


def test_build_scene_digest_empty_for_no_entities():
    assert build_scene_digest({}) == ''
    assert build_scene_digest({'entities': []}) == ''


def test_build_grounded_context_block_caps_entities_visible_first():
    entities = [
        {
            'id': 'hidden_%d' % index,
            'label': None,
            'kind': 'object',
            'class': 'Cup',
            'visible': False,
        }
        for index in range(3)
    ] + [
        {
            'id': 'cup_visible',
            'label': 'cup',
            'kind': 'object',
            'class': 'Cup',
            'visible': True,
        }
    ]

    block = build_grounded_context_block({'entities': entities}, max_entities=2)

    assert '"id": "cup_visible"' in block
    assert '"id": "hidden_2"' not in block
    assert 'Showing 2 of 4 entities' in block
