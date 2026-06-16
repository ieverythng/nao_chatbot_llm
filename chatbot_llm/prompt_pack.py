"""Prompt-pack loading for the migrated chatbot backend."""

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


# Structural defaults are safe because they are not prompt wording.
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
    'required': ['verbal_ack', 'route', 'confidence'],
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
    """Load the canonical packaged prompt pack."""
    return load_prompt_pack('')


def load_prompt_pack(path: str, logger=None) -> PromptPack:
    """Load the canonical prompt pack from YAML; fail loudly on prompt defects."""
    pack_path = str(path or '').strip()
    source = Path(pack_path) if pack_path else _default_prompt_pack_path()
    if source is None:
        raise FileNotFoundError('Could not resolve chatbot prompt pack path')
    if not source.exists():
        raise FileNotFoundError(f'Chatbot prompt pack path does not exist: "{source}"')

    if yaml is None:
        raise RuntimeError('PyYAML is required to load chatbot prompt packs')

    try:
        raw = source.read_text(encoding='utf-8')
    except Exception as err:  # pragma: no cover - filesystem dependent
        raise RuntimeError(f'Could not read chatbot prompt pack "{source}": {err}') from err

    try:
        parsed = yaml.safe_load(raw)
    except Exception as err:
        raise ValueError(f'Chatbot prompt pack parse failed for "{source}": {err}') from err

    if not isinstance(parsed, dict):
        raise ValueError(f'Chatbot prompt pack root must be a mapping: "{source}"')

    _require_text(parsed, 'system_prompt', source)
    _require_text(parsed, 'response_prompt_addendum', source)
    _require_text(parsed, 'intent_prompt_addendum', source)

    response_schema = parsed.get('response_schema', DEFAULT_RESPONSE_SCHEMA)
    if not isinstance(response_schema, dict):
        _warn(logger, 'response_schema must be a mapping; using structural defaults')
        response_schema = DEFAULT_RESPONSE_SCHEMA

    intent_schema = parsed.get('intent_schema', DEFAULT_INTENT_SCHEMA)
    if not isinstance(intent_schema, dict):
        _warn(logger, 'intent_schema must be a mapping; using structural defaults')
        intent_schema = DEFAULT_INTENT_SCHEMA

    planner_multi_step_heuristics = _coerce_heuristics(
        parsed.get(
            'planner_multi_step_heuristics',
            DEFAULT_PLANNER_MULTI_STEP_HEURISTICS,
        ),
        defaults=DEFAULT_PLANNER_MULTI_STEP_HEURISTICS,
        logger=logger,
    )

    return PromptPack(
        system_prompt=_as_text(parsed.get('system_prompt')),
        response_prompt_addendum=_as_text(parsed.get('response_prompt_addendum')),
        intent_prompt_addendum=_as_text(parsed.get('intent_prompt_addendum')),
        environment_description=_as_text(parsed.get('environment_description')),
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


def _require_text(parsed: dict, key: str, source: Path) -> None:
    if not _as_text(parsed.get(key)):
        raise ValueError(f'Chatbot prompt pack "{source}" must define non-empty {key}')


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
