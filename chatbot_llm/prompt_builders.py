"""Prompt builders for response and intent generation."""

from __future__ import annotations

from pathlib import Path
from string import Template


# ---------------------------------------------------------------------------
# Shared prompt templates
# ---------------------------------------------------------------------------

RESPONSE_STAGE_TEMPLATE = Template(
    """
You are a friendly robot called $robot_name.
Provide a concise spoken response to the user.

The user_id of the person you are talking to is $user_id.
Environment description:
$environment

Recent conversation history is included in the messages above.
Use it to maintain continuity across the last several turns.

Output requirements:
- Return only valid JSON (no markdown or extra text).
- Include field verbal_ack as a short answer suitable for TTS.
""".strip()
)

PLANNER_MODE_RESPONSE_TEMPLATE = """
Planner-mode response fields:
- Also include route with one of: dialogue, knowledge_query, execution.
- Include confidence as a number.
- Include user_intent only as routing metadata when useful.
- Do not include a top-level plan field or user_intent.plan. planner_llm owns
  executable plans after this response.
- Route policy, response style, and examples are defined by the configured
  chatbot prompt pack below; treat that pack as authoritative.
""".strip()

INTENT_STAGE_TEMPLATE = Template(
    """
You are an intent extraction component for robot $robot_name.
Infer the user's intent from the latest user text, recent dialogue, grounded
context, and the canonical labels described in the configured prompt pack.

The user_id of the person you are talking to is $user_id.
Environment description:
$environment

Recent conversation history is included in the messages above.
Use it to maintain continuity across the last several turns.

Output requirements:
- Return only valid JSON (no markdown or extra text).
- Provide user_intent with key "type".
- Include intent_confidence as a number.
- You may include goal_text, scene_targets, request_kind, and intent_sequence
  when the configured prompt pack calls for them.
- Do not include a top-level plan field or user_intent.plan. planner_llm owns
  executable plans.
- Intent labels, route policy, and examples are defined by the configured
  chatbot prompt pack below; treat that pack as authoritative.
""".strip()
)


# ---------------------------------------------------------------------------
# Public prompt assembly helpers
# ---------------------------------------------------------------------------

def load_persona_prompt(path: str, logger=None) -> str:
    """Load optional persona prompt text from file."""
    prompt_path = str(path or '').strip()
    if not prompt_path:
        return ''
    source = Path(prompt_path)
    if not source.exists():
        _warn(logger, f'persona_prompt_path does not exist: "{prompt_path}"')
        return ''
    try:
        return source.read_text(encoding='utf-8').strip()
    except Exception as err:  # pragma: no cover - filesystem dependent
        _warn(logger, f'Could not read persona prompt: {err}')
        return ''


def build_response_prompt(
    robot_name: str,
    user_id: str,
    system_prompt: str,
    environment_description: str,
    knowledge_snapshot: str,
    response_prompt_addendum: str,
    skill_catalog_text: str,
    persona_prompt: str,
    planner_mode_enabled: bool,
) -> str:
    """Build system prompt used for verbal response generation."""
    return _join_prompt_parts(
        persona_prompt,
        _safe_format(system_prompt, robot_name=robot_name, user_id=user_id),
        RESPONSE_STAGE_TEMPLATE.safe_substitute(
            robot_name=robot_name,
            user_id=user_id or 'user1',
            environment=environment_description or 'No specific objects described.',
        ),
        PLANNER_MODE_RESPONSE_TEMPLATE if planner_mode_enabled else '',
        _knowledge_snapshot_block(knowledge_snapshot),
        skill_catalog_text,
        response_prompt_addendum,
    )


def build_intent_prompt(
    robot_name: str,
    user_id: str,
    system_prompt: str,
    environment_description: str,
    knowledge_snapshot: str,
    intent_prompt_addendum: str,
    skill_catalog_text: str,
    persona_prompt: str,
) -> str:
    """Build system prompt used for structured intent extraction."""
    return _join_prompt_parts(
        persona_prompt,
        _safe_format(system_prompt, robot_name=robot_name, user_id=user_id),
        INTENT_STAGE_TEMPLATE.safe_substitute(
            robot_name=robot_name,
            user_id=user_id or 'user1',
            environment=environment_description or 'No specific objects described.',
        ),
        _knowledge_snapshot_block(knowledge_snapshot),
        skill_catalog_text,
        intent_prompt_addendum,
    )


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _join_prompt_parts(*parts: str) -> str:
    cleaned = [str(part).strip() for part in parts if str(part).strip()]
    return '\n\n'.join(cleaned).strip()


def _safe_format(template: str, **kwargs) -> str:
    raw = str(template or '').strip()
    if not raw:
        return ''
    try:
        return raw.format(**kwargs).strip()
    except Exception:
        return raw


def _knowledge_snapshot_block(snapshot: str) -> str:
    clean_snapshot = str(snapshot or '').strip()
    if not clean_snapshot:
        return ''
    return (
        'Grounded context for this turn:\n'
        'Grounded context:\n%s' % clean_snapshot
    )


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
