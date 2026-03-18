"""KnowledgeCore role configuration parsing and snapshot formatting helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from chatbot_llm.backend_config import ChatbotConfig


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
    for row in rows[:total_rows]:
        line = _format_query_row(row, ordered_vars)
        if line:
            fact_lines.append(f'- {line}' if summary_lines else line)

    _append_bounded_lines(formatted_lines, fact_lines, remaining_chars)

    return '\n'.join(formatted_lines).strip()


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
