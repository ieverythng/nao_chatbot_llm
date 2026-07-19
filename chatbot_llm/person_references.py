"""Resolve user-facing person references against grounded HRI identities."""

from __future__ import annotations

import re


_TRACKED_PERSON_ID = re.compile(
    r'^(?:anonymous_person|sim_person|person)_(?P<suffix>[a-z0-9_]+)$',
    re.IGNORECASE,
)


def resolve_grounded_person_id(grounded_context: dict, reference: str) -> str:
    """Return one canonical grounded person ID, or empty when absent or ambiguous."""
    folded_reference = _fold(reference)
    if not folded_reference:
        return ''

    exact_matches: list[str] = []
    shorthand_matches: list[str] = []
    for entity in _person_entities(grounded_context):
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id:
            continue
        if folded_reference in {_fold(name) for name in person_reference_names(entity)}:
            exact_matches.append(entity_id)
            continue
        if folded_reference in _anonymous_person_aliases(entity_id):
            shorthand_matches.append(entity_id)

    matches = exact_matches or shorthand_matches
    unique_matches = list(dict.fromkeys(matches))
    return unique_matches[0] if len(unique_matches) == 1 else ''


def resolve_grounded_person_in_text(grounded_context: dict, text: str) -> str:
    """Return one person whose grounded name or tracker suffix occurs in text."""
    normalized_text = _words(text)
    if not normalized_text:
        return ''
    matches: list[str] = []
    for entity in _person_entities(grounded_context):
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id:
            continue
        references = {_words(name) for name in person_reference_names(entity)}
        suffix_match = _TRACKED_PERSON_ID.fullmatch(entity_id)
        if suffix_match:
            suffix = _words(suffix_match.group('suffix'))
            references.update({suffix, 'person %s' % suffix})
        if any(
            reference and re.search(r'\b%s\b' % re.escape(reference), normalized_text)
            for reference in references
        ):
            matches.append(entity_id)
    unique_matches = list(dict.fromkeys(matches))
    return unique_matches[0] if len(unique_matches) == 1 else ''


def canonicalize_person_references(user_intent: dict, grounded_context: dict) -> dict:
    """Replace uniquely grounded person references with their canonical IDs."""
    normalized = dict(user_intent or {})
    resolved_ids: list[str] = []

    for key in ('recipient', 'target_person'):
        resolved = resolve_grounded_person_id(grounded_context, normalized.get(key, ''))
        if resolved:
            normalized[key] = resolved
            resolved_ids.append(resolved)

    has_scene_targets = 'scene_targets' in normalized
    raw_targets = normalized.get('scene_targets', [])
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if isinstance(raw_targets, (list, tuple)):
        targets = []
        for target in raw_targets:
            if isinstance(target, dict):
                updated = dict(target)
                reference = updated.get('id') or updated.get('label') or updated.get('name')
                resolved = resolve_grounded_person_id(grounded_context, reference)
                if resolved:
                    updated['id'] = resolved
                    resolved_ids.append(resolved)
                targets.append(updated)
                continue
            resolved = resolve_grounded_person_id(grounded_context, target)
            targets.append(resolved or target)
            if resolved:
                resolved_ids.append(resolved)
        if has_scene_targets:
            normalized['scene_targets'] = targets

    if resolved_ids:
        targets = normalized.setdefault('scene_targets', [])
        if isinstance(targets, list):
            existing_ids = {
                str(target.get('id', '') if isinstance(target, dict) else target).strip()
                for target in targets
            }
            for entity_id in dict.fromkeys(resolved_ids):
                if entity_id not in existing_ids:
                    targets.append(entity_id)
    return normalized


def _person_entities(value):
    if isinstance(value, dict):
        if looks_like_person_entity(value):
            yield value
        for nested in value.values():
            yield from _person_entities(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _person_entities(item)


def looks_like_person_entity(entity: dict) -> bool:
    descriptor = ' '.join(
        str(entity.get(key, ''))
        for key in ('kind', 'class', 'type', 'entity_class', 'label', 'id')
    ).lower()
    if any(term in descriptor for term in ('location', 'place', 'room')):
        return False
    return any(term in descriptor for term in ('person', 'human', 'recipient'))


def person_reference_names(entity: dict) -> list[str]:
    names = [
        str(entity.get(key, '')).strip()
        for key in ('id', 'label', 'name', 'display_name')
        if str(entity.get(key, '')).strip()
    ]
    for relation in entity.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', '')).strip().lower()
        if predicate not in {'dbp:name', 'rdfs:label'} and 'name' not in predicate:
            continue
        for key in ('object', 'value', 'target', 'label'):
            value = str(relation.get(key, '')).strip()
            if value:
                names.append(value)
    return names


def _anonymous_person_aliases(entity_id: str) -> set[str]:
    match = _TRACKED_PERSON_ID.fullmatch(str(entity_id or '').strip())
    if not match:
        return set()
    suffix = _fold(match.group('suffix'))
    return (
        {suffix, f'person{suffix}', f'theperson{suffix}', f'anonymousperson{suffix}'}
        if suffix
        else set()
    )


def _fold(value) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


def _words(value) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', str(value or '').lower()))
