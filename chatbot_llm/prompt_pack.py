"""Prompt-pack loading and defaults for the migrated chatbot backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency
    yaml = None


# ---------------------------------------------------------------------------
# Built-in prompt-pack defaults
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    'You are a friendly robot called {robot_name}. '
    'You are helpful, concise, and clear in spoken interactions.'
)

DEFAULT_RESPONSE_PROMPT_ADDENDUM = (
    'Reply with short natural spoken text suitable for TTS. '
    'Avoid markdown and avoid long lists.'
)

DEFAULT_INTENT_PROMPT_ADDENDUM = (
    'Map user requests to one canonical intent label when possible.'
)

DEFAULT_ENVIRONMENT_DESCRIPTION = 'No specific objects described.'

DEFAULT_RESPONSE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'verbal_ack': {'type': 'string'},
        'confidence': {'type': 'number'},
    },
    'required': ['verbal_ack'],
}

DEFAULT_INTENT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'user_intent': {
            'type': 'object',
            'properties': {
                'type': {'type': 'string'},
                'object': {'type': 'string'},
                'recipient': {'type': 'string'},
                'input': {'type': 'string'},
                'goal': {'type': 'string'},
            },
            'required': ['type'],
        },
        'intent_confidence': {'type': 'number'},
        'confidence': {'type': 'number'},
    },
}


@dataclass(frozen=True)
class PromptPack:
    """Prompt and schema assets for two-stage LLM execution."""

    system_prompt: str
    response_prompt_addendum: str
    intent_prompt_addendum: str
    environment_description: str
    response_schema: dict[str, Any]
    intent_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# Public prompt-pack loading API
# ---------------------------------------------------------------------------

def default_prompt_pack() -> PromptPack:
    """Return built-in defaults used when no external prompt pack is available."""
    return PromptPack(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_prompt_addendum=DEFAULT_RESPONSE_PROMPT_ADDENDUM,
        intent_prompt_addendum=DEFAULT_INTENT_PROMPT_ADDENDUM,
        environment_description=DEFAULT_ENVIRONMENT_DESCRIPTION,
        response_schema=dict(DEFAULT_RESPONSE_SCHEMA),
        intent_schema=dict(DEFAULT_INTENT_SCHEMA),
    )


def load_prompt_pack(path: str, logger=None) -> PromptPack:
    """Load prompt pack from YAML file; return defaults on errors."""
    defaults = default_prompt_pack()
    pack_path = str(path or '').strip()
    if not pack_path:
        return defaults

    source = Path(pack_path)
    if not source.exists():
        _warn(logger, f'Prompt pack path does not exist: "{pack_path}"')
        return defaults

    if yaml is None:
        _warn(logger, 'PyYAML unavailable; prompt pack ignored')
        return defaults

    try:
        raw = source.read_text(encoding='utf-8')
    except Exception as err:  # pragma: no cover - filesystem dependent
        _warn(logger, f'Could not read prompt pack: {err}')
        return defaults

    try:
        parsed = yaml.safe_load(raw)
    except Exception as err:
        _warn(logger, f'Prompt pack parse failed: {err}')
        return defaults

    if not isinstance(parsed, dict):
        _warn(logger, 'Prompt pack root must be a mapping')
        return defaults

    response_schema = parsed.get('response_schema', defaults.response_schema)
    if not isinstance(response_schema, dict):
        _warn(logger, 'response_schema must be a mapping; using defaults')
        response_schema = defaults.response_schema

    intent_schema = parsed.get('intent_schema', defaults.intent_schema)
    if not isinstance(intent_schema, dict):
        _warn(logger, 'intent_schema must be a mapping; using defaults')
        intent_schema = defaults.intent_schema

    return PromptPack(
        system_prompt=_as_text(parsed.get('system_prompt', defaults.system_prompt)),
        response_prompt_addendum=_as_text(
            parsed.get('response_prompt_addendum', defaults.response_prompt_addendum)
        ),
        intent_prompt_addendum=_as_text(
            parsed.get('intent_prompt_addendum', defaults.intent_prompt_addendum)
        ),
        environment_description=_as_text(
            parsed.get('environment_description', defaults.environment_description)
        ),
        response_schema=response_schema,
        intent_schema=intent_schema,
    )


# ---------------------------------------------------------------------------
# Prompt-pack parsing helpers
# ---------------------------------------------------------------------------

def _as_text(value) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
