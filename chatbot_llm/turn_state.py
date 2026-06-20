"""Deterministic state packet supplied to atomic chatbot decisions."""

from __future__ import annotations

import hashlib
import json
import re


TURN_STATE_VERSION = 'ts.v1'


def build_turn_state(
    *,
    turn_id: str,
    utterance: str,
    history: list[str],
    grounded_context: dict,
    active_goal_id: str = '',
    skill_manifest: list[dict] | None = None,
    latest_execution: dict | None = None,
) -> dict:
    """Build one bounded, read-only state packet from current runtime evidence."""
    world_state = _world_state(grounded_context)
    return {
        'turn_state_version': TURN_STATE_VERSION,
        'turn_id': str(turn_id or '').strip(),
        'utterance': str(utterance or '').strip(),
        'dialogue_context': {
            'recent_turns': [str(item) for item in list(history or [])[-6:]],
        },
        'active_goal': {
            'goal_id': str(active_goal_id or '').strip(),
            'status': 'active' if str(active_goal_id or '').strip() else 'none',
        },
        'world_state': world_state,
        'latest_execution': dict(latest_execution or {}),
        'available_skills': _bounded_skill_manifest(skill_manifest or []),
        'route_policy': {
            'allowed_routes': ['dialogue', 'knowledge_query', 'execution'],
            'execution_requires_admission': True,
            'kb_mutation_requires_skill': True,
            'required_skill_params_must_be_grounded': True,
        },
    }


def resolve_mentioned_subject_ids(utterance: str, grounded_context: dict) -> list[str]:
    """Return canonical IDs whose ID, label, or stable name is mentioned uniquely."""
    text = _normalized_text(utterance)
    if not text:
        return []

    direct_matches: list[str] = []
    alias_matches: dict[str, list[str]] = {}
    for entity in _entities(grounded_context):
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id:
            continue
        if _contains_phrase(text, entity_id):
            direct_matches.append(entity_id)
        aliases = {str(entity.get('label', '')).strip()}
        for relation in entity.get('relations', []):
            if not isinstance(relation, dict):
                continue
            if str(relation.get('predicate', '')).strip() == 'dbp:name':
                aliases.add(str(relation.get('object', '')).strip())
        for alias in aliases:
            clean_alias = _normalized_text(alias)
            if clean_alias and _contains_phrase(text, alias):
                alias_matches.setdefault(clean_alias, []).append(entity_id)
    unique_alias_matches = [
        ids[0]
        for ids in alias_matches.values()
        if len(set(ids)) == 1
    ]
    return _unique([*direct_matches, *unique_alias_matches])


def grounding_id(grounded_context: dict) -> str:
    """Create a stable identifier without adding a new public snapshot contract."""
    canonical = json.dumps(
        {'entities': _entities(grounded_context)},
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    return 'gc:%s' % hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


def _world_state(grounded_context: dict) -> dict:
    entities = _entities(grounded_context)
    return {
        'grounding_id': grounding_id({'entities': entities}),
        'entities': entities,
    }


def _entities(grounded_context: dict) -> list[dict]:
    payload = dict(grounded_context or {})
    entities = payload.get('entities', [])
    if not isinstance(entities, list):
        return []
    return [dict(item) for item in entities if isinstance(item, dict)]


def _bounded_skill_manifest(items: list[dict]) -> list[dict]:
    skills: list[dict] = []
    for item in items[:32]:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', item.get('skill_id', ''))).strip()
        if not name:
            continue
        skills.append(
            {
                'name': name,
                'aliases': _str_list(item.get('aliases', [])),
                'params': _str_list(item.get('params', item.get('input_names', []))),
                'required_params': _str_list(item.get('required_params', [])),
            }
        )
    return skills


def _str_list(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalized_text(value: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9_]+', ' ', str(value or '').lower()).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    clean_phrase = _normalized_text(phrase)
    if not clean_phrase:
        return False
    return re.search(r'(?<![a-z0-9_])%s(?![a-z0-9_])' % re.escape(clean_phrase), text) is not None


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
