from chatbot_llm.person_references import canonicalize_person_references
from chatbot_llm.person_references import resolve_grounded_person_id
from chatbot_llm.person_references import resolve_grounded_person_in_text


def _context(*people: dict) -> dict:
    return {'entities': list(people)}


def test_resolves_unique_hri_detector_suffix_to_canonical_id() -> None:
    context = _context(
        {
            'id': 'anonymous_person_jeebd',
            'label': 'anonymous_person',
            'kind': 'person',
            'class': 'Human',
        }
    )

    assert resolve_grounded_person_id(context, 'jeebd') == 'anonymous_person_jeebd'
    assert resolve_grounded_person_id(context, 'person jeebd') == 'anonymous_person_jeebd'


def test_does_not_resolve_ambiguous_hri_detector_suffix() -> None:
    context = _context(
        {'id': 'anonymous_person_jeebd', 'kind': 'person'},
        {'id': 'ANONYMOUS_PERSON_JEEBD', 'kind': 'person'},
    )

    assert resolve_grounded_person_id(context, 'jeebd') == ''


def test_preserves_named_person_resolution() -> None:
    context = _context(
        {
            'id': 'codex_lab_alex',
            'label': 'anonymous_person',
            'kind': 'person',
            'relations': [{'predicate': 'dbp:name', 'object': 'ALEX'}],
        }
    )

    assert resolve_grounded_person_id(context, 'ALEX') == 'codex_lab_alex'


def test_canonicalizes_recipient_and_scene_target_for_planner_handoff() -> None:
    context = _context({'id': 'anonymous_person_jeebd', 'kind': 'person'})

    normalized = canonicalize_person_references(
        {'recipient': 'jeebd', 'scene_targets': ['cup_1', 'person jeebd']},
        context,
    )

    assert normalized['recipient'] == 'anonymous_person_jeebd'
    assert normalized['scene_targets'] == ['cup_1', 'anonymous_person_jeebd']


def test_resolves_tracker_suffix_from_goal_text() -> None:
    context = _context({'id': 'anonymous_person_jdefb', 'kind': 'person'})

    assert (
        resolve_grounded_person_in_text(
            context,
            'Walk to the person jdefb and tell me when you arrive.',
        )
        == 'anonymous_person_jdefb'
    )


def test_resolves_sim_person_tracker_suffix_from_goal_text() -> None:
    context = _context({'id': 'sim_person_lznze', 'kind': 'person'})

    assert resolve_grounded_person_in_text(context, 'I meant person lznze!') == (
        'sim_person_lznze'
    )
