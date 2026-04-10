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
Planner-mode routing requirements:
- Also include route with one of: dialogue, knowledge_query, execution.
- Use route="execution" for physical actions, skill requests, or multi-step requests.
- Use route="knowledge_query" for grounded scene/perception questions answered from the
  live knowledge snapshot.
- Use route="dialogue" for greetings, identity, wellbeing, help, or general conversation.
- When possible include user_intent with key "type".
- For execution turns, you may also include ack_text, ack_mode, scene_targets, and a
  plan list shaped as {type,name,args}.
- When the user combines multiple requested actions, or mixes action with perception
  or dialogue, keep every requested step in `plan` in execution order.
""".strip()

INTENT_STAGE_TEMPLATE = Template(
    """
You are an intent extraction component for robot $robot_name.
Infer the user's intent from the latest user text and assistant reply.

Canonical intent labels:
- posture_stand
- posture_sit
- posture_kneel
- head_center
- head_look_left
- head_look_right
- head_look_up
- head_look_down
- greet
- identity
- wellbeing
- help
- kb_query_visible_people
- kb_query_visible_objects
- kb_query_scene_change
- fallback

The user_id of the person you are talking to is $user_id.
Environment description:
$environment

Recent conversation history is included in the messages above.
Use it to maintain continuity across the last several turns.

Output requirements:
- Return only valid JSON (no markdown or extra text).
- Provide user_intent with key "type" when possible.
- Prefer the `kb_query_*` labels when the user is asking who is visible now,
  what objects are visible now, or whether the scene changed compared with
  earlier turns.
- When the user requests an action, you may also include ack_text, ack_mode,
  scene_targets, and a plan list shaped as {type,name,args}.
- When the user combines multiple requested actions, or an action plus a
  follow-up perception or dialogue task, keep every requested step in `plan`
  in execution order instead of dropping the later steps.
- If no single canonical label covers the whole request, keep `user_intent.type`
  on the closest executable label or use `fallback`, but still return the full
  ordered `plan`.
- For posture or head-motion requests, prefer one plan step with
  type="skill", name="perform_motion", and args.object set to the canonical motion.
- If uncertain, use user_intent.type = "fallback".
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
        'Live symbolic scene state from KnowledgeCore for this turn:\n'
        "- Treat it as the robot's best grounded view of the current scene.\n"
        '- Use the "Current grounded scene" section for what is visible right now.\n'
        '- Use any "Recent scene memory" section only as bounded cross-turn context.\n'
        '- Distinguish carefully between what is visible now and what was only seen '
        'earlier.\n'
        '- Use it when answering who is present, whether a face/person is detected, '
        'and what objects or relations are currently known.\n'
        '- Combine it with the recent dialogue history when the user asks whether '
        'the current scene matches what was seen earlier.\n'
        '- If it mentions a person or face entity without a stable name, say you can '
        'currently detect someone without inventing an identity.\n'
        '- If the current entity ID changed since earlier turns, do not claim it is '
        'definitely the same person unless the evidence supports that.\n'
        '- If an entity was only present in recent scene memory, say it was seen '
        'earlier but cannot be confirmed as currently visible.\n'
        '- If the snapshot does not support a perception claim, say you cannot confirm it.\n'
        'Knowledge snapshot:\n%s' % clean_snapshot
    )


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
