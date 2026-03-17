from chatbot_llm.backend_config import ChatbotConfig
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
        fallback_response='fallback',
        max_history_messages=12,
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
        knowledge_enabled=False,
        knowledge_query_service_name='/kb/query',
        knowledge_query_timeout_sec=0.5,
        knowledge_default_patterns=['?s ?p ?o'],
        knowledge_default_vars=['?s', '?p', '?o'],
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
        patterns=['?person likes ?thing'],
        query_vars=['?person', '?thing'],
        models=['all'],
        max_results=3,
        max_chars=120,
    )


def test_resolve_knowledge_snapshot_settings_falls_back_on_invalid_json():
    settings = resolve_knowledge_snapshot_settings('{not json}', make_config())

    assert settings.enabled is False
    assert settings.patterns == ['?s ?p ?o']
    assert settings.query_vars == ['?s', '?p', '?o']


def test_format_knowledge_snapshot_formats_triples_and_truncates():
    settings = KnowledgeSnapshotSettings(
        enabled=True,
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

    assert snapshot == 'mug isOn table\n...'
