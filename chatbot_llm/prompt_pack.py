"""Prompt-pack loading and defaults for the migrated chatbot backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - runtime dependency
    from ament_index_python.packages import PackageNotFoundError
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - import-light unit tests
    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError('ament_index_python is unavailable')

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
    'Avoid markdown and avoid long lists. '
    'For greeting-only turns, keep route as dialogue unless the user explicitly '
    'asks for a physical action. '
    'If unsure between dialogue and execution without an explicit action verb, '
    'prefer dialogue. '
    'For execution turns, keep verbal_ack as intent-to-act acknowledgement and '
    'do not claim completion in that same utterance. Do not narrate the action '
    'in parentheses or report observations/results in verbal_ack. '
    'For execution turns, return only verbal_ack, route, confidence, and '
    'user_intent metadata. Do not include a top-level plan field or '
    'user_intent.plan.'
)

DEFAULT_INTENT_PROMPT_ADDENDUM = (
    'Map user requests to one canonical intent label when possible. '
    'When the user requests an action, you may return ack_text, ack_mode, '
    'scene_targets, and goal. For greeting/social turns, prefer greet unless '
    'an explicit physical action request is present. '
    'Do not include a top-level plan field or user_intent.plan.'
)

DEFAULT_ENVIRONMENT_DESCRIPTION = 'No specific objects described.'

DEFAULT_PLANNER_MULTI_STEP_HEURISTICS: dict[str, list[str]] = {
    'coordination_markers': [
        ' and then ',
        ' then ',
        ' while ',
        ' while also ',
        ' also ',
        ' after that ',
        ' after you ',
        ' before that ',
        ' before you ',
        ' once you ',
        ' once that ',
        ' at the same time ',
        ' simultaneously ',
    ],
    'action_hint_tokens': [
        'stand',
        'sit',
        'kneel',
        'crouch',
        'look',
        'move',
        'turn',
        'head',
        'bring',
        'grab',
        'pick',
        'place',
        'go',
        'guide',
        'walk',
    ],
}

DEFAULT_RESPONSE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'verbal_ack': {'type': 'string'},
        'route': {
            'type': 'string',
            'enum': ['dialogue', 'knowledge_query', 'execution'],
        },
        'user_intent': {
            'type': 'object',
            'properties': {
                'type': {'type': 'string'},
                'object': {'type': 'string'},
                'recipient': {'type': 'string'},
                'input': {'type': 'string'},
                'goal': {'type': 'string'},
                'ack_text': {'type': 'string'},
                'ack_mode': {'type': 'string'},
                'intent_sequence': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'scene_targets': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
            },
        },
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
                'ack_text': {'type': 'string'},
                'ack_mode': {'type': 'string'},
                'intent_sequence': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'scene_targets': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
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
    planner_multi_step_heuristics: dict[str, list[str]]


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
        planner_multi_step_heuristics=_heuristics_copy(
            DEFAULT_PLANNER_MULTI_STEP_HEURISTICS
        ),
    )


def load_prompt_pack(path: str, logger=None) -> PromptPack:
    """Load prompt pack from YAML file; return defaults on errors."""
    defaults = default_prompt_pack()
    pack_path = str(path or '').strip()
    source = Path(pack_path) if pack_path else _default_prompt_pack_path()
    if source is None:
        return defaults
    if not source.exists():
        _warn(logger, f'Prompt pack path does not exist: "{source}"')
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

    planner_multi_step_heuristics = _coerce_heuristics(
        parsed.get(
            'planner_multi_step_heuristics',
            defaults.planner_multi_step_heuristics,
        ),
        defaults=defaults.planner_multi_step_heuristics,
        logger=logger,
    )

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
        planner_multi_step_heuristics=planner_multi_step_heuristics,
    )


# ---------------------------------------------------------------------------
# Prompt-pack parsing helpers
# ---------------------------------------------------------------------------

def _as_text(value) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _default_prompt_pack_path() -> Path | None:
    try:
        return Path(get_package_share_directory('chatbot_llm')) / 'config' / 'chat_prompt_pack.yaml'
    except PackageNotFoundError:
        source_candidate = Path(__file__).resolve().parents[1] / 'config' / 'chat_prompt_pack.yaml'
        return source_candidate if source_candidate.exists() else None


def _coerce_heuristics(value, *, defaults: dict[str, list[str]], logger=None) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        _warn(logger, 'planner_multi_step_heuristics must be a mapping; using defaults')
        return _heuristics_copy(defaults)
    return {
        'coordination_markers': _coerce_str_list(
            value.get('coordination_markers', defaults.get('coordination_markers', []))
        ),
        'action_hint_tokens': _coerce_str_list(
            value.get('action_hint_tokens', defaults.get('action_hint_tokens', []))
        ),
    }


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _heuristics_copy(value: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        'coordination_markers': list(value.get('coordination_markers', [])),
        'action_hint_tokens': list(value.get('action_hint_tokens', [])),
    }


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
