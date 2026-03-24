from chatbot_llm.skill_catalog import build_skill_catalog_text


def test_build_skill_catalog_text_includes_kb_skills_metadata():
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
