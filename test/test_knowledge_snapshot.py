from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.knowledge_snapshot import build_scene_context
from chatbot_llm.knowledge_snapshot import extract_scene_memory_entry
from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot
from chatbot_llm.knowledge_snapshot import resolve_knowledge_snapshot_settings


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
        identity_reminder_every_n_turns=6,
        intent_detection_mode='llm',
        prompt_pack_path='',
        use_skill_catalog=False,
        skill_catalog_packages=[],
        skill_catalog_max_entries=0,
        skill_catalog_max_chars=0,
        planner_mode_enabled=False,
        planner_request_topic='/planner/request',
        planner_request_intent='planner_request',
        planner_scene_summary_topic='/scene/summary',
        planner_world_model_snapshot_topic='/world_model/enriched_snapshot',
        planner_world_model_text_topic='/world_model/enriched_text',
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
