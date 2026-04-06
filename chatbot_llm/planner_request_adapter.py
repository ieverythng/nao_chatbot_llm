"""Helpers for optional planner-mode handoff from ``chatbot_llm``."""

from __future__ import annotations

import json

from kb_skills.intent_labels import KB_QUERY_INTENTS

try:  # pragma: no cover - optional dependency in local unit tests
    from hri_actions_msgs.msg import Intent
except ImportError:  # pragma: no cover - import-light fallback
    class Intent:  # type: ignore[no-redef]
        SAY = 'say'
        GREET = 'greet'
        RAW_USER_INPUT = 'raw_user_input'
        PERFORM_MOTION = 'perform_motion'
        MODALITY_SPEECH = 'speech'
        UNKNOWN_AGENT = 'unknown_agent'

        def __init__(self) -> None:
            self.intent = ''
            self.source = ''
            self.modality = ''
            self.confidence = 0.0
            self.priority = 0
            self.data = ''


_NON_PLANNER_INTENT_NAMES = {
    Intent.SAY,
    Intent.GREET,
    Intent.RAW_USER_INPUT,
    *KB_QUERY_INTENTS,
}
_NON_SCENE_TARGET_OBJECTS = {
    'stand',
    'sit',
    'kneel',
    'head_center',
    'head_look_left',
    'head_look_right',
    'head_look_up',
    'head_look_down',
    'look_at_reset',
}


def should_route_intents_through_planner(intents: list[Intent]) -> bool:
    """Return true when the turn contains execution-oriented intents."""
    if not isinstance(intents, list) or not intents:
        return False
    return any(str(intent.intent).strip() not in _NON_PLANNER_INTENT_NAMES for intent in intents)


def build_planner_request_payload(
    *,
    turn_id: str,
    user_text: str,
    turn_result,
    knowledge_context: str,
    planner_mode: str = 'default',
    max_history_entries: int = 6,
) -> dict:
    """Build the planner ingress payload from the current turn result."""
    user_intent = turn_result.user_intent if isinstance(turn_result.user_intent, dict) else {}
    ack_text = str(user_intent.get('ack_text', '')).strip() or str(turn_result.verbal_ack).strip()
    ack_mode = str(user_intent.get('ack_mode', '')).strip() or 'say'
    dialogue_context = _bounded_dialogue_context(
        turn_result.updated_history,
        max_history_entries=max_history_entries,
    )
    grounded_context = {}
    clean_knowledge_context = str(knowledge_context or '').strip()
    if clean_knowledge_context:
        grounded_context['knowledge_snapshot'] = clean_knowledge_context

    return {
        'request_id': str(turn_id).strip(),
        'user_text': str(user_text or '').strip(),
        'normalized_intents': _normalized_intents(turn_result.intent),
        'ack_text': ack_text,
        'ack_mode': ack_mode,
        'scene_targets': _scene_targets_from_user_intent(user_intent),
        'dialogue_context': dialogue_context,
        'grounded_context': grounded_context,
        'planner_mode': str(planner_mode or 'default').strip() or 'default',
    }


def build_planner_request_intent(
    *,
    turn_id: str,
    user_text: str,
    source_user_id: str,
    turn_result,
    knowledge_context: str,
    planner_request_intent: str = 'planner_request',
    planner_mode: str = 'default',
    max_history_entries: int = 6,
) -> Intent:
    """Create the ``Intent`` message published on ``/planner/request``."""
    payload = build_planner_request_payload(
        turn_id=turn_id,
        user_text=user_text,
        turn_result=turn_result,
        knowledge_context=knowledge_context,
        planner_mode=planner_mode,
        max_history_entries=max_history_entries,
    )
    intent = Intent()
    intent.intent = str(planner_request_intent or 'planner_request').strip() or 'planner_request'
    intent.source = str(source_user_id).strip() or getattr(Intent, 'UNKNOWN_AGENT', 'unknown_agent')
    intent.modality = getattr(Intent, 'MODALITY_SPEECH', 'speech')
    intent.confidence = float(getattr(turn_result, 'intent_confidence', 0.0))
    intent.priority = 0
    intent.data = json.dumps(payload, separators=(',', ':'))
    return intent


def _bounded_dialogue_context(history: list[str], *, max_history_entries: int) -> list[str]:
    if not isinstance(history, list) or max_history_entries <= 0:
        return []
    return [str(item).strip() for item in history[-max_history_entries:] if str(item).strip()]


def _normalized_intents(intent_name: str) -> list[str]:
    clean_intent = str(intent_name or '').strip()
    return [clean_intent] if clean_intent else []


def _scene_targets_from_user_intent(user_intent: dict) -> list[str]:
    scene_targets = _coerce_str_list(user_intent.get('scene_targets'))
    if scene_targets:
        return scene_targets
    scene_object = str(user_intent.get('object', '')).strip()
    if scene_object and scene_object not in _NON_SCENE_TARGET_OBJECTS:
        return [scene_object]
    return []


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        clean_value = value.strip()
        return [clean_value] if clean_value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
