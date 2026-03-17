"""KnowledgeCore role configuration parsing and snapshot formatting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from chatbot_llm.backend_config import ChatbotConfig


@dataclass(frozen=True)
class KnowledgeSnapshotSettings:
    """Resolved KnowledgeCore query settings for one dialogue role."""

    enabled: bool
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
    patterns = _coerce_str_list(block.get('patterns'), fallback=resolved.patterns)
    query_vars = _coerce_str_list(block.get('vars'), fallback=resolved.query_vars)
    models = _coerce_str_list(block.get('models'), fallback=resolved.models)
    max_results = _coerce_positive_int(block.get('max_results'), resolved.max_results)
    max_chars = _coerce_positive_int(block.get('max_chars'), resolved.max_chars)

    return KnowledgeSnapshotSettings(
        enabled=enabled,
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

    for row in rows[:total_rows]:
        line = _format_query_row(row, ordered_vars)
        if not line:
            continue
        separator_len = 1 if formatted_lines else 0
        if remaining_chars and len(line) + separator_len > remaining_chars:
            if formatted_lines:
                formatted_lines.append('...')
            break
        formatted_lines.append(line)
        remaining_chars -= len(line) + separator_len

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
        return ' '.join(str(row[key]).strip() for key in ('s', 'p', 'o')).strip()

    if ordered_vars:
        values = []
        for key in ordered_vars:
            value = str(row.get(key, '')).strip()
            if value:
                values.append(f'{key}={value}')
        return ', '.join(values).strip()

    return ', '.join(
        f'{key}={str(value).strip()}'
        for key, value in sorted(row.items())
        if str(value).strip()
    ).strip()


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


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
