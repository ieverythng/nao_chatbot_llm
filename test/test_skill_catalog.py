from chatbot_llm.skill_catalog import build_skill_catalog_text
from chatbot_llm.skill_catalog import build_turn_state_skill_manifest


def test_build_skill_catalog_text_includes_json_exported_skills() -> None:
    rendered, descriptors = build_skill_catalog_text(
        ['interaction_skills', 'nao_skills'],
        max_entries=0,
        max_chars=0,
    )

    descriptor_ids = {descriptor.skill_id for descriptor in descriptors}

    assert 'look_at' in descriptor_ids
    assert 'do_head_motion' in descriptor_ids
    assert 'look_at' in rendered


def test_build_skill_catalog_text_includes_kb_skills_metadata() -> None:
    rendered, descriptors = build_skill_catalog_text(
        package_names=['kb_skills'],
        max_entries=8,
        max_chars=1000,
        logger=None,
    )

    assert descriptors
    assert descriptors[0].package == 'kb_skills'
    assert descriptors[0].skill_id == 'kb_query'
    assert descriptors[0].interface_path == '/kb/query'
    assert descriptors[0].datatype == 'kb_msgs/srv/Query'
    assert 'Available skills:' in rendered
    assert '[kb_skills] kb_query -> /kb/query (kb_msgs/srv/Query)' in rendered


def test_turn_state_manifest_keeps_registry_required_params(tmp_path) -> None:
    registry_path = tmp_path / 'skills.json'
    registry_path.write_text(
        '{"skills":['
        '{"name":"bring_object","params":["object","recipient"],'
        '"required_params":["object","recipient"]},'
        '{"name":"pick_object","params":["object"],"required_params":["object"]},'
        '{"name":"place_object","params":["object","target"],'
        '"required_params":["object","target"]}'
        ']}',
        encoding='utf-8',
    )
    manifest = {
        item['name']: item
        for item in build_turn_state_skill_manifest(str(registry_path))
    }

    assert manifest['bring_object']['required_params'] == ['object', 'recipient']
    assert manifest['pick_object']['required_params'] == ['object']
    assert manifest['place_object']['required_params'] == ['object', 'target']
