"""KnowledgeCore role configuration parsing and snapshot formatting helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from chatbot_llm.backend_config import ChatbotConfig


_DIGEST_SUPPORT_CLASSES = frozenset(
    ('bench', 'counter', 'desk', 'shelf', 'surface', 'table', 'workbench')
)
_DIGEST_PLACE_CLASSES = frozenset(
    ('corridor', 'kitchen', 'lab', 'location', 'park', 'place', 'robot station', 'room', 'station')
)
_DIGEST_NON_USER_OBJECT_CLASSES = frozenset(
    (
        'cyc spatial thing',
        'cyc spatial thing localized',
        'location',
        'owl thing',
        'place',
        'room',
        'spatial thing',
        'spatial thing localized',
        'support surface',
    )
)
_DIGEST_GENERIC_OBJECT_CLASSES = frozenset(('tableware',))
_DIGEST_SPATIAL_PREDICATES = frozenset(
    (
        'oro:contains',
        'oro:isat',
        'oro:isin',
        'oro:ison',
        'oro:isunder',
        'oro:placeof',
    )
)


# ---------------------------------------------------------------------------
# Role-scoped snapshot settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeSnapshotSettings:
    """Resolved KnowledgeCore query settings for one dialogue role."""

    enabled: bool
    query_groups: list[list[str]]
    patterns: list[str]
    query_vars: list[str]
    models: list[str]
    max_results: int
    max_chars: int


# ---------------------------------------------------------------------------
# Role configuration parsing
# ---------------------------------------------------------------------------

def resolve_knowledge_snapshot_settings(
    role_configuration: str,
    config: ChatbotConfig,
    logger=None,
) -> KnowledgeSnapshotSettings:
    """Merge node defaults with optional role-level knowledge snapshot overrides."""
    resolved = KnowledgeSnapshotSettings(
        enabled=config.knowledge_enabled,
        query_groups=_parse_query_groups(config.knowledge_default_query_groups),
        patterns=list(config.knowledge_default_patterns),
        query_vars=list(config.knowledge_default_vars),
        models=list(config.knowledge_default_models),
        max_results=config.knowledge_max_results,
        max_chars=config.knowledge_max_chars,
    )

    raw_role_config = str(role_configuration or '{}').strip() or '{}'
    if raw_role_config in {'', '{}'}:
        return resolved

    try:
        parsed = json.loads(raw_role_config)
    except json.JSONDecodeError:
        _warn(logger, 'Ignoring invalid dialogue role configuration JSON for knowledge snapshot')
        return resolved

    if not isinstance(parsed, dict):
        return resolved

    block = parsed.get('knowledge_snapshot')
    if block is None:
        return resolved
    if not isinstance(block, dict):
        _warn(logger, 'Ignoring non-object knowledge_snapshot role configuration')
        return resolved

    enabled = _coerce_bool(block.get('enabled'), resolved.enabled)
    query_groups = _coerce_query_groups(
        block.get('query_groups'),
        fallback=resolved.query_groups,
    )
    patterns = _coerce_str_list(block.get('patterns'), fallback=resolved.patterns)
    query_vars = _coerce_str_list(block.get('vars'), fallback=resolved.query_vars)
    models = _coerce_str_list(block.get('models'), fallback=resolved.models)
    max_results = _coerce_positive_int(block.get('max_results'), resolved.max_results)
    max_chars = _coerce_positive_int(block.get('max_chars'), resolved.max_chars)

    return KnowledgeSnapshotSettings(
        enabled=enabled,
        query_groups=query_groups,
        patterns=patterns or list(resolved.patterns),
        query_vars=query_vars or list(resolved.query_vars),
        models=models,
        max_results=max_results,
        max_chars=max_chars,
    )


# ---------------------------------------------------------------------------
# Snapshot formatting
# ---------------------------------------------------------------------------

def format_knowledge_snapshot(json_payload: str, settings: KnowledgeSnapshotSettings) -> str:
    """Format `/kb/query` JSON bindings into a bounded deterministic text block."""
    rows = _parse_query_rows(json_payload)
    if not rows:
        return ''

    ordered_vars = [item.lstrip('?') for item in settings.query_vars if str(item).strip()]
    formatted_lines: list[str] = []
    remaining_chars = max(settings.max_chars, 0)
    total_rows = min(len(rows), max(settings.max_results, 0))
    visible_entity_lines = _build_visible_entity_summary(rows[:total_rows])
    summary_lines = visible_entity_lines or _build_snapshot_summary(rows[:total_rows])

    if summary_lines:
        remaining_chars = _append_bounded_lines(
            formatted_lines,
            summary_lines,
            remaining_chars,
        )
        if formatted_lines:
            remaining_chars = _append_bounded_lines(
                formatted_lines,
                ['Scene facts:'],
                remaining_chars,
            )

    fact_lines = []
    typed_entity_fact_lines = _build_typed_entity_fact_lines(rows[:total_rows])
    if typed_entity_fact_lines:
        fact_lines.extend(
            f'- {line}' if summary_lines else line for line in typed_entity_fact_lines
        )

    for row in rows[:total_rows]:
        if {'entity', 'type'}.issubset(row.keys()) and typed_entity_fact_lines:
            continue
        line = _format_query_row(row, ordered_vars)
        if line:
            fact_lines.append(f'- {line}' if summary_lines else line)

    _append_bounded_lines(formatted_lines, fact_lines, remaining_chars)

    return '\n'.join(formatted_lines).strip()


def build_scene_context(
    snapshot: str,
    recent_scene_memory: list[str] | None = None,
) -> str:
    """Combine the live snapshot with a short, explicit scene-memory section."""
    clean_snapshot = str(snapshot or '').strip()
    memory_entries = _unique_preserving_order(recent_scene_memory or [])
    sections: list[str] = []

    if clean_snapshot:
        sections.append('Current grounded scene:\n%s' % clean_snapshot)
    elif memory_entries:
        sections.append(
            'Current grounded scene:\n'
            'No entities are confirmed in the live KnowledgeCore snapshot for this turn.'
        )

    if memory_entries:
        sections.append(
            'Recent scene memory from previous turns:\n%s'
            % '\n'.join('- %s' % entry for entry in memory_entries)
        )

    return '\n\n'.join(sections).strip()


def build_grounded_context_block(
    grounded_context: dict,
    *,
    max_entities: int | None = None,
) -> str:
    """Serialize the compact grounded-context projection used by the chatbot.

    The live context object is never mutated. The prompt projection removes contradictory
    movable-object spatial aliases before ``max_entities`` caps visible entities first.
    Authoritative counts always live in the scene digest above the block.
    """
    payload = _sanitize_grounded_context_projection(grounded_context)
    if not payload:
        return ''

    truncation_note = ''
    entities = payload.get('entities')
    if (
        max_entities is not None
        and max_entities > 0
        and isinstance(entities, list)
        and len(entities) > max_entities
    ):
        visible_first = sorted(
            entities,
            key=lambda item: 0
            if isinstance(item, dict) and coerce_visible(item.get('visible', True))
            else 1,
        )
        payload['entities'] = visible_first[:max_entities]
        truncation_note = (
            '\n(Showing %d of %d entities; see the scene digest above for full counts.)'
            % (max_entities, len(entities))
        )

    return 'Grounded context JSON:\n```json\n%s\n```%s' % (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        truncation_note,
    )


# ---------------------------------------------------------------------------
# Deterministic scene digest (derived view over the grounded-context contract)
# ---------------------------------------------------------------------------

_SCENE_DIGEST_MAX_PEOPLE = 8
_SCENE_DIGEST_MAX_OBJECT_CLASSES = 12
_SCENE_DIGEST_MAX_ATTRS_PER_CLASS = 8


def build_scene_digest(grounded_context: dict) -> str:
    """Build an authoritative, deterministic scene summary from the live grounded context.

    This is a derived view layered on top of the grounded-context contract (which is left
    untouched). It gives the model a pre-counted, attribute-qualified inventory so it
    cannot under- or over-count, or invent scene facts, under long-context pressure.

    The digest is recomputed every turn from the current entities, so KB mutations
    (``kb/revise`` updates, ``kb/remove`` retractions) are reflected on the next turn with
    no caching: removed entities disappear, revised colours/relations re-qualify the label.
    """
    projected_context = _sanitize_grounded_context_projection(grounded_context)
    entities = _digest_entities(projected_context)
    locations = _digest_locations(projected_context, entities)
    if not entities and not locations:
        return ''

    people = [item for item in entities if _digest_kind(item) == 'person']
    objects = [
        item for item in entities
        if _digest_kind(item) != 'person' and _is_user_facing_digest_object(item)
    ]

    body = [
        line
        for line in (
            _digest_people_line(people),
            _digest_objects_line(objects),
            _digest_locations_line(locations),
        )
        if line
    ]
    if not body:
        return ''

    header = (
        'Scene digest (authoritative for this turn; derived live from the grounded '
        'context. Trust these counts and this inventory over manual counting):'
    )
    return '\n'.join([header, *('- %s' % line for line in body)])


def _sanitize_grounded_context_projection(grounded_context: dict) -> dict:
    """Return an LLM-facing copy with movable-object spatial clutter collapsed."""
    if not isinstance(grounded_context, dict) or not grounded_context:
        return {}
    payload = dict(grounded_context)
    raw_entities = payload.get('entities')
    if not isinstance(raw_entities, list):
        return payload

    entity_index = {
        str(entity.get('id', '')).strip(): entity
        for entity in raw_entities
        if isinstance(entity, dict) and str(entity.get('id', '')).strip()
    }
    payload['entities'] = [
        _sanitize_entity_spatial_projection(entity, entity_index)
        if isinstance(entity, dict)
        else entity
        for entity in raw_entities
    ]
    return payload


def _sanitize_entity_spatial_projection(entity: dict, entity_index: dict[str, dict]) -> dict:
    relations = entity.get('relations', [])
    if not isinstance(relations, list) or not _is_user_facing_digest_object(entity):
        return dict(entity)

    canonical_place = _spatial_place_for_entity(entity, entity_index, set())
    projected = dict(entity)
    kept_relations: list[dict] = []
    dropped_movable_spatial_relation = False
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', relation.get('p', ''))).strip()
        obj = str(relation.get('object', relation.get('o', ''))).strip()
        if not predicate or not obj:
            continue
        predicate_key = _spatial_predicate_key(predicate)
        target_entity = entity_index.get(obj)
        target_is_movable = (
            isinstance(target_entity, dict)
            and target_entity
            and _is_user_facing_digest_object(target_entity)
        )
        if predicate_key in _DIGEST_SPATIAL_PREDICATES and target_is_movable:
            dropped_movable_spatial_relation = True
            continue
        kept_relations.append(dict(relation))

    has_canonical_place = any(
        isinstance(relation, dict)
        and _spatial_predicate_key(str(relation.get('predicate', relation.get('p', ''))))
        in _DIGEST_SPATIAL_PREDICATES
        and str(relation.get('object', relation.get('o', ''))).strip() == canonical_place
        for relation in kept_relations
    )
    if canonical_place and dropped_movable_spatial_relation and not has_canonical_place:
        kept_relations.append({'predicate': 'oro:isAt', 'object': canonical_place})
    projected['relations'] = _unique_relations(kept_relations)
    return projected


def _spatial_place_for_entity(
    entity: dict,
    entity_index: dict[str, dict],
    seen: set[str],
) -> str:
    entity_id = str(entity.get('id', '')).strip()
    if entity_id:
        if entity_id in seen:
            return ''
        seen.add(entity_id)
    relations = entity.get('relations', [])
    if not isinstance(relations, list):
        return ''
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', relation.get('p', ''))).strip()
        if _spatial_predicate_key(predicate) not in {'oro:isat', 'oro:isin', 'oro:ison'}:
            continue
        obj = str(relation.get('object', relation.get('o', ''))).strip()
        if not obj:
            continue
        target_entity = entity_index.get(obj)
        if _is_digest_location_group_candidate(target_entity or {'id': obj}, obj):
            return obj
        if isinstance(target_entity, dict) and _is_user_facing_digest_object(target_entity):
            resolved = _spatial_place_for_entity(target_entity, entity_index, seen)
            if resolved:
                return resolved
    return ''


def _spatial_predicate_key(predicate: str) -> str:
    return str(predicate or '').strip().lower().replace('_', '')


def _digest_entities(grounded_context: dict) -> list[dict]:
    if not isinstance(grounded_context, dict):
        return []
    raw_entities = grounded_context.get('entities', [])
    if not isinstance(raw_entities, list):
        return []
    by_id: dict[str, dict] = {}
    anonymous: list[dict] = []
    for item in raw_entities:
        if not isinstance(item, dict) or not coerce_visible(item.get('visible', True)):
            continue
        entity_id = str(item.get('id', '')).strip()
        if not entity_id:
            anonymous.append(item)
            continue
        if entity_id not in by_id:
            by_id[entity_id] = dict(item)
            continue
        by_id[entity_id] = _merge_digest_entity(by_id[entity_id], item)
    return [*by_id.values(), *anonymous]


def _merge_digest_entity(base: dict, candidate: dict) -> dict:
    merged = dict(base)
    for key in ('label', 'kind', 'class'):
        if (
            not str(merged.get(key, '') or '').strip()
            and str(candidate.get(key, '') or '').strip()
        ):
            merged[key] = candidate.get(key)
    merged['visible'] = True
    merged['relations'] = _unique_relations(
        list(base.get('relations', []) if isinstance(base.get('relations'), list) else [])
        + list(
            candidate.get('relations', [])
            if isinstance(candidate.get('relations'), list)
            else []
        )
    )
    return merged


def _unique_relations(relations: list) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', relation.get('p', ''))).strip()
        obj = str(relation.get('object', relation.get('o', ''))).strip()
        key = (predicate.lower(), obj)
        if not predicate or not obj or key in seen:
            continue
        seen.add(key)
        unique.append(dict(relation))
    return unique


def _digest_kind(entity: dict) -> str:
    return str(entity.get('kind', '')).strip().lower()


def _digest_people_line(people: list[dict]) -> str:
    if not people:
        return 'People (0): none currently grounded'
    names = _unique_preserving_order(_digest_entity_name(item) for item in people)
    shown = names[:_SCENE_DIGEST_MAX_PEOPLE]
    suffix = ', ...' if len(names) > _SCENE_DIGEST_MAX_PEOPLE else ''
    return 'People (%d): %s%s' % (len(people), ', '.join(shown), suffix)


def _digest_objects_line(objects: list[dict]) -> str:
    if not objects:
        return 'Objects (0): none currently grounded'

    counts: dict[str, int] = {}
    attributes: dict[str, list[str]] = {}
    order: list[str] = []
    for entity in objects:
        class_label = _digest_class_label(entity) or 'object'
        if class_label not in counts:
            counts[class_label] = 0
            attributes[class_label] = []
            order.append(class_label)
        counts[class_label] += 1
        attribute = _digest_object_attribute(entity)
        if attribute:
            attributes[class_label].append(attribute)

    parts: list[str] = []
    for class_label in order[:_SCENE_DIGEST_MAX_OBJECT_CLASSES]:
        attrs = _unique_preserving_order(attributes[class_label])
        attr_text = ''
        if attrs:
            shown_attrs = attrs[:_SCENE_DIGEST_MAX_ATTRS_PER_CLASS]
            more = ', ...' if len(attrs) > _SCENE_DIGEST_MAX_ATTRS_PER_CLASS else ''
            attr_text = ' [%s%s]' % (', '.join(shown_attrs), more)
        parts.append('%s x%d%s' % (class_label, counts[class_label], attr_text))

    more_classes = '; ...' if len(order) > _SCENE_DIGEST_MAX_OBJECT_CLASSES else ''
    return 'Objects (%d): %s%s' % (len(objects), '; '.join(parts), more_classes)


def _digest_object_attribute(entity: dict) -> str:
    attributes: list[str] = []
    name = _digest_relation_object(entity, ('dbp:name', 'name'))
    if name:
        attributes.append('named %s' % _humanize_value(name))
    color = _digest_relation_object(entity, ('dbp:color', 'color'))
    if color:
        attributes.append(_normalized_identifier(color))
    support = _digest_support_phrase(entity)
    if support:
        attributes.append(support)
    return ', '.join(_unique_preserving_order(attributes))


def _digest_locations(grounded_context: dict, entities: list[dict]) -> list[dict]:
    if not isinstance(grounded_context, dict):
        return []
    raw_locations = grounded_context.get('locations', grounded_context.get('location_groups', []))
    if isinstance(raw_locations, list) and raw_locations:
        return [
            _filter_digest_location(item)
            for item in raw_locations
            if isinstance(item, dict)
        ]
    return _derive_digest_locations(entities)


def _derive_digest_locations(entities: list[dict]) -> list[dict]:
    entity_index = {
        str(entity.get('id', '')).strip(): entity
        for entity in entities
        if isinstance(entity, dict) and str(entity.get('id', '')).strip()
    }
    groups: dict[str, dict] = {}
    for entity in entities:
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id:
            continue
        for relation in entity.get('relations', []):
            if not isinstance(relation, dict):
                continue
            predicate = str(relation.get('predicate', relation.get('p', ''))).strip()
            predicate_key = predicate.lower().replace('_', '')
            obj = str(relation.get('object', relation.get('o', ''))).strip()
            if not obj:
                continue
            if predicate_key in {'oro:isat', 'oro:ison', 'oro:isin'}:
                if not _is_user_facing_digest_object(entity):
                    continue
                group_entity = entity_index.get(obj, {'id': obj})
                if not _is_digest_location_group_candidate(group_entity, obj):
                    continue
                group = groups.setdefault(
                    obj,
                    {
                        'id': obj,
                        'label': _digest_entity_name(group_entity),
                        'contains': [],
                    },
                )
                _add_digest_location_member(group, entity, predicate)
            elif predicate_key == 'oro:contains':
                if not _is_digest_location_group_candidate(entity, entity_id):
                    continue
                group = groups.setdefault(
                    entity_id,
                    {
                        'id': entity_id,
                        'label': _digest_entity_name(entity),
                        'contains': [],
                    },
                )
                _add_digest_location_member(
                    group,
                    entity_index.get(obj, {'id': obj}),
                    predicate,
                )
    return sorted(groups.values(), key=lambda item: str(item.get('label') or item.get('id', '')))


def _add_digest_location_member(group: dict, entity: dict, relation: str) -> None:
    member_id = str(entity.get('id', '')).strip()
    if not member_id or member_id == str(group.get('id', '')).strip():
        return
    if not _is_digest_location_member(entity):
        return
    member = {
        'id': member_id,
        'label': _digest_entity_name(entity),
        'kind': _digest_kind(entity) or 'object',
        'class': str(entity.get('class', '')).strip(),
        'relation': str(relation or '').strip(),
    }
    contains = group.setdefault('contains', [])
    if not any(item.get('id') == member_id for item in contains if isinstance(item, dict)):
        contains.append(member)


def _filter_digest_location(location: dict) -> dict:
    if not _is_digest_location_group_candidate(
        location,
        str(location.get('id', '')).strip(),
    ):
        return {'contains': []}
    filtered = dict(location)
    members = location.get('contains', location.get('members', []))
    if not isinstance(members, list):
        filtered['contains'] = []
        return filtered
    filtered['contains'] = [
        dict(member) for member in members
        if isinstance(member, dict) and _is_digest_location_member(member)
    ]
    return filtered


def _is_digest_location_group_candidate(entity: dict, entity_id: str) -> bool:
    if isinstance(entity, dict) and entity:
        kind = str(entity.get('kind', '')).strip().lower()
        if kind and kind not in {'object', 'location', 'place'}:
            return False
        class_token = _digest_class_token(entity.get('class', ''))
        if class_token in _DIGEST_SUPPORT_CLASSES or class_token in _DIGEST_PLACE_CLASSES:
            return True
        if class_token and _is_digest_user_object_type(class_token):
            return False
    clean_id = _humanize_value(entity_id).strip().lower()
    return any(
        marker in clean_id
        for marker in (
            'counter',
            'corridor',
            'desk',
            'kitchen',
            'lab',
            'park',
            'room',
            'shelf',
            'station',
            'surface',
            'table',
        )
    )


def _is_user_facing_digest_object(entity: dict) -> bool:
    """Return true for objects that should appear in user-facing scene digests."""
    if not isinstance(entity, dict) or not entity:
        return False
    kind = _digest_kind(entity)
    if kind and kind != 'object':
        return False
    class_token = _digest_class_token(entity.get('class', ''))
    if class_token in _DIGEST_NON_USER_OBJECT_CLASSES:
        return False
    if class_token in _DIGEST_PLACE_CLASSES:
        return False
    if _is_digest_user_object_type(class_token):
        return True
    rdf_type_tokens = []
    for relation in entity.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', relation.get('p', ''))).strip().lower()
        if predicate != 'rdf:type':
            continue
        rdf_type_tokens.append(_digest_class_token(relation.get('object', relation.get('o', ''))))
    if any(_is_digest_user_object_type(token) for token in rdf_type_tokens):
        return True
    for relation_class in rdf_type_tokens:
        if relation_class in _DIGEST_NON_USER_OBJECT_CLASSES:
            return False
        if relation_class in _DIGEST_PLACE_CLASSES:
            return False
    return True


def _is_digest_location_member(entity: dict) -> bool:
    """Exclude physical support anchors from their derived member list."""
    if not _is_user_facing_digest_object(entity):
        return False
    class_token = _digest_class_token(entity.get('class', ''))
    if class_token in _DIGEST_SUPPORT_CLASSES:
        return False
    for relation in entity.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(
            relation.get('predicate', relation.get('p', ''))
        ).strip().lower()
        if predicate != 'rdf:type':
            continue
        if _digest_class_token(relation.get('object', relation.get('o', ''))) in _DIGEST_SUPPORT_CLASSES:
            return False
    return True


def _is_digest_user_object_type(class_token: str) -> bool:
    token = str(class_token or '').strip()
    if not token:
        return False
    if token in _DIGEST_NON_USER_OBJECT_CLASSES:
        return False
    if token in _DIGEST_PLACE_CLASSES:
        return False
    return True


def _digest_class_token(value) -> str:
    return _humanize_value(value).strip().lower()


def _digest_locations_line(locations: list[dict]) -> str:
    if not locations:
        return ''
    parts: list[str] = []
    for location in locations[:_SCENE_DIGEST_MAX_OBJECT_CLASSES]:
        if not isinstance(location, dict):
            continue
        members = location.get('contains', location.get('members', []))
        if not isinstance(members, list) or not members:
            continue
        member_names = []
        for member in members[:_SCENE_DIGEST_MAX_ATTRS_PER_CLASS]:
            if isinstance(member, dict):
                member_names.append(
                    _humanize_value(
                        member.get('label', member.get('id', 'unknown object'))
                    )
                )
            else:
                member_names.append(_humanize_value(member))
        if not member_names:
            continue
        more = ', ...' if len(members) > _SCENE_DIGEST_MAX_ATTRS_PER_CLASS else ''
        location_name = _humanize_value(
            location.get('label', location.get('id', 'unknown location'))
        )
        parts.append('%s contains %s%s' % (location_name, ', '.join(member_names), more))
    if not parts:
        return ''
    more_locations = '; ...' if len(locations) > _SCENE_DIGEST_MAX_OBJECT_CLASSES else ''
    return 'Locations: %s%s' % ('; '.join(parts), more_locations)


def _digest_support_phrase(entity: dict) -> str:
    for predicate_key, prefix in (
        ('oro:ison', 'on'),
        ('oro:isat', 'at'),
        ('oro:isin', 'in'),
        ('oro:contains', 'contains'),
    ):
        obj = _digest_relation_object(entity, (predicate_key,))
        if obj:
            return '%s %s' % (prefix, _humanize_value(obj))
    return ''


def _digest_relation_object(entity: dict, predicate_keys) -> str:
    relations = entity.get('relations', [])
    if not isinstance(relations, list):
        return ''
    wanted = {str(key).strip().lower() for key in predicate_keys}
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', relation.get('p', ''))).strip().lower()
        if predicate in wanted:
            obj = str(relation.get('object', relation.get('o', ''))).strip()
            if obj:
                return obj
    return ''


def _digest_entity_name(entity: dict) -> str:
    name = _digest_relation_object(entity, ('dbp:name', 'name'))
    if name:
        return _humanize_value(name)
    label = str(entity.get('label', '') or '').strip()
    if label:
        return _humanize_value(label)
    identifier = str(entity.get('id', '')).strip()
    return identifier or 'unknown'


def _digest_class_label(entity: dict) -> str:
    class_value = str(entity.get('class', '')).strip()
    label = str(entity.get('label', '') or '').strip()
    class_token = _digest_class_token(class_value)
    if label and class_token in _DIGEST_GENERIC_OBJECT_CLASSES:
        return _humanize_value(label).lower()
    if class_value:
        return _display_type(class_value).lower()
    if label:
        return _humanize_value(label).lower()
    return ''


def coerce_visible(value) -> bool:
    """Coerce a grounded-context ``visible`` flag, defaulting to visible."""
    return _coerce_bool(value, True)


def extract_scene_memory_entry(snapshot: str) -> str:
    """Extract one compact scene summary line to retain across turns."""
    for raw_line in str(snapshot or '').splitlines():
        clean_line = str(raw_line).strip()
        if not clean_line or clean_line == 'Scene facts:':
            continue
        if clean_line.startswith('- '):
            clean_line = clean_line[2:].strip()
        return clean_line
    return ''


# ---------------------------------------------------------------------------
# Query row formatting helpers
# ---------------------------------------------------------------------------

def _parse_query_rows(json_payload: str) -> list[dict]:
    payload = str(json_payload or '').strip()
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    rows = [row for row in parsed if isinstance(row, dict)]
    return rows


def _format_query_row(row: dict, ordered_vars: list[str]) -> str:
    if {'s', 'p', 'o'}.issubset(row.keys()):
        subject = _humanize_value(row.get('s', ''))
        predicate = _humanize_predicate(row.get('p', ''))
        obj = _humanize_value(row.get('o', ''))
        return ' '.join(part for part in (subject, predicate, obj) if part).strip()

    if {'entity', 'type'}.issubset(row.keys()):
        entity = _humanize_value(row.get('entity', ''))
        entity_type = _display_type(row.get('type', ''))
        if entity and entity_type:
            return f'{entity} is a {entity_type}'
        return entity or entity_type

    if ordered_vars:
        values = []
        for key in ordered_vars:
            value = _humanize_value(row.get(key, ''))
            if value:
                values.append(f'{key}={value}')
        return ', '.join(values).strip()

    return ', '.join(
        f'{key}={_humanize_value(value)}'
        for key, value in sorted(row.items())
        if _humanize_value(value)
    ).strip()


# ---------------------------------------------------------------------------
# Scene summarization helpers
# ---------------------------------------------------------------------------

def _append_bounded_lines(
    target_lines: list[str],
    candidate_lines: list[str],
    remaining_chars: int,
) -> int:
    for line in candidate_lines:
        clean_line = str(line).strip()
        if not clean_line:
            continue
        separator_len = 1 if target_lines else 0
        if remaining_chars and len(clean_line) + separator_len > remaining_chars:
            if target_lines:
                target_lines.append('...')
            return 0
        target_lines.append(clean_line)
        remaining_chars -= len(clean_line) + separator_len
    return remaining_chars


def _build_snapshot_summary(rows: list[dict]) -> list[str]:
    detected_people = _unique_preserving_order(
        _humanize_value(row.get('s', ''))
        for row in rows
        if _is_person_like_entity(row.get('s', ''))
    )
    if not detected_people:
        return []

    preview = ', '.join(detected_people[:4])
    if len(detected_people) > 4:
        preview += ', ...'
    return [f'Detected person/face-related entities right now: {preview}']


def _build_visible_entity_summary(rows: list[dict]) -> list[str]:
    typed_entities = _collect_typed_entities(rows)
    if not typed_entities:
        return []

    entity_descriptions = []
    for entity, raw_types in typed_entities.items():
        informative_types = [
            _display_type(raw_type)
            for raw_type in raw_types
            if _is_informative_type(raw_type)
        ]
        informative_types = _unique_preserving_order(informative_types)
        if informative_types:
            entity_descriptions.append(
                f'{_humanize_value(entity)} ({", ".join(informative_types[:3])})'
            )
        else:
            entity_descriptions.append(_humanize_value(entity))

    if not entity_descriptions:
        return []

    return [
        'Entities currently seen by the robot: %s'
        % ', '.join(entity_descriptions[:6])
    ]


def _build_typed_entity_fact_lines(rows: list[dict]) -> list[str]:
    typed_entities = _collect_typed_entities(rows)
    if not typed_entities:
        return []

    lines = []
    for entity, raw_types in typed_entities.items():
        informative_types = [
            _display_type(raw_type)
            for raw_type in raw_types
            if _is_informative_type(raw_type)
        ]
        informative_types = _unique_preserving_order(informative_types)
        entity_name = _humanize_value(entity)
        if entity_name and informative_types:
            lines.append(
                '%s is currently classified as %s'
                % (entity_name, ', '.join(informative_types[:3]))
            )
        elif entity_name:
            lines.append('%s is currently present' % entity_name)
    return lines


# ---------------------------------------------------------------------------
# Humanization and coercion helpers
# ---------------------------------------------------------------------------

def _collect_typed_entities(rows: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        entity = str(row.get('entity', '')).strip()
        raw_type = str(row.get('type', '')).strip()
        if not entity:
            continue
        grouped.setdefault(entity, [])
        if raw_type:
            grouped[entity].append(raw_type)
    return grouped


def _is_person_like_entity(value) -> bool:
    text = _normalized_identifier(value)
    if not text:
        return False
    return any(
        keyword in text
        for keyword in ('person', 'people', 'human', 'face', 'speaker', 'visitor', 'user')
    )


def _humanize_predicate(value) -> str:
    token = _normalized_identifier(value)
    if not token:
        return ''
    if token in {'type', 'rdf type', 'isa', 'is a'}:
        return 'is a'
    return token


def _display_type(value) -> str:
    text = _humanize_value(value)
    if not text:
        return ''
    if ':' in text:
        text = text.rsplit(':', 1)[-1]
    return text


def _humanize_value(value) -> str:
    raw = str(value).strip()
    if not raw:
        return ''
    return _normalized_identifier(raw, preserve_case=True) or raw


def _normalized_identifier(value, preserve_case: bool = False) -> str:
    text = str(value).strip()
    if not text:
        return ''

    for separator in ('#', '/'):
        if separator in text:
            text = text.rsplit(separator, 1)[-1]
    if ':' in text and not text.startswith(('http://', 'https://')):
        text = text.rsplit(':', 1)[-1]

    text = text.replace('_', ' ').replace('-', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not preserve_case:
        text = text.lower()
    return text


def _unique_preserving_order(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _is_informative_type(value) -> bool:
    token = _normalized_identifier(value)
    if not token:
        return False
    return token not in {
        'thing',
        'location',
        'spatial thing',
        'spatial thing localized',
        'solid tangible thing',
        'partially tangible',
        'enduring thing localized',
        'agent',
        'artifact',
    }


def _coerce_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return bool(fallback)


def _coerce_positive_int(value, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _coerce_str_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
        return list(fallback)
    if isinstance(value, str):
        clean_value = value.strip()
        if clean_value:
            return [clean_value]
    return list(fallback)


def _coerce_query_groups(value, fallback: list[list[str]]) -> list[list[str]]:
    if isinstance(value, (list, tuple)):
        parsed = _parse_query_groups(value)
        if parsed:
            return parsed
    elif isinstance(value, str):
        parsed = _parse_query_groups([value])
        if parsed:
            return parsed
    return [list(group) for group in fallback]


def _parse_query_groups(raw_groups) -> list[list[str]]:
    groups: list[list[str]] = []
    for raw_group in raw_groups or []:
        clean_group = str(raw_group).strip()
        if not clean_group:
            continue
        patterns = [part.strip() for part in clean_group.split('&&') if part.strip()]
        if patterns:
            groups.append(patterns)
    return groups


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
