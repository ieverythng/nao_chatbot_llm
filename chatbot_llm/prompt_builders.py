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
- For visibility checks ("who do you see", "do you see anyone", "what objects are visible"),
  prefer route="knowledge_query" unless the user explicitly asks you to perform a new
  scan/action first.
- For knowledge_query visibility answers, ground visible entities only in the
  current turn's grounded snapshot. Do not infer object/person totals from
  omitted count fields, and do not reuse old visibility claims from prior
  dialogue turns.
- If the current grounded snapshot shows no visible people, do not say you see
  a person.
- Use route="dialogue" for greetings, identity, wellbeing, help, or general conversation.
- For greeting-only turns (hi/hello/hey + social opener), keep route="dialogue"
  unless the user explicitly asks for a physical action (for example "wave at me").
- When possible include user_intent with key "type".
- For execution turns, include only routing metadata in user_intent: type,
  intent_sequence, goal, object, ack_text, ack_mode, and scene_targets.
- Use user_intent.intent_sequence for compound semantics such as motion followed by
  perception/reporting. It is a list of intent labels, not executable plan steps.
- For execution turns, verbal_ack is only a brief future-tense acknowledgement.
  Do not narrate the action in parentheses and do not report observations/results there.
- Do not include a top-level plan field or user_intent.plan. planner_llm owns all
  executable steps after this response.
- For multi-step requests, summarize the whole requested task in user_intent.goal
  and keep route="execution".
- Ask at most one clarification question when a missing detail blocks safe execution.
  Do not chain multiple follow-up questions in the same turn.
- Examples:
  Stand: {"verbal_ack":"Sure, I will stand up now.","route":"execution",
    "confidence":0.84,"user_intent":{"type":"posture_stand","ack_mode":"say"}}
  Sequence: {"verbal_ack":"Sure, I will move my head right and then sit down.",
    "route":"execution","confidence":0.82,
    "user_intent":{"type":"head_look_right","goal":"move your head right, then sit down"}}
  Head motion with report: {"verbal_ack":"Sure, I will move my head right and report what I see.",
    "route":"execution","confidence":0.82,
    "user_intent":{"type":"head_look_right",
    "intent_sequence":["head_look_right","inspect_scene","report_result"],
    "goal":"move your head right and report what is visible"}}
  Scan: {"verbal_ack":"Sure, I will look around and report what I can see.",
    "route":"execution","confidence":0.78,
    "user_intent":{"type":"inspect_scene",
    "intent_sequence":["inspect_scene","report_result"],
    "goal":"look around and report what is visible"}}
  KB visibility: {"verbal_ack":"I can currently see two objects: a cup and a phone.",
    "route":"knowledge_query","confidence":0.82,
    "user_intent":{"type":"kb_query_visible_objects"}}
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
  intent_sequence, scene_targets, and goal.
- When the user combines multiple requested actions, or an action plus a
  follow-up perception or dialogue task, summarize the whole task in goal.
- Use intent_sequence to list the semantic labels in order for compound requests,
  for example ["head_look_right","inspect_scene","report_result"]. Do not put
  executable step dictionaries in user_intent.
- If no single canonical label covers the whole request, keep `user_intent.type`
  on the closest executable label or use `fallback`.
- Do not include a top-level plan field or user_intent.plan. planner_llm owns all
  executable steps.
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
        '- If a Grounded context JSON block is present, treat its entities array as '
        'the current visible world and its relations arrays as known KB facts.\n'
        '- If an entity was only present in recent scene memory, say it was seen '
        'earlier but cannot be confirmed as currently visible.\n'
        '- If the snapshot does not support a perception claim, say you cannot confirm it.\n'
        'Knowledge snapshot:\n%s' % clean_snapshot
    )


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
