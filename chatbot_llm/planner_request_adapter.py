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
    str(intent_name).strip().lower()
    for intent_name in (
        Intent.SAY,
        Intent.GREET,
        Intent.RAW_USER_INPUT,
        *KB_QUERY_INTENTS,
    )
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
_PLANNER_REQUEST_KINDS = {
    'new_goal',
    'goal_update',
    'clarification_answer',
    'cancel_request',
}
_CANCEL_INTENT_TYPES = {
    'cancel',
    'cancel_request',
    'stop',
    'stop_activity',
    'suspend',
}


def should_route_intents_through_planner(intents: list[Intent]) -> bool:
    """Return true when the turn contains execution-oriented intents."""
    if not isinstance(intents, list) or not intents:
        return False
    return any(
        _normalize_token(getattr(intent, 'intent', '')) not in _NON_PLANNER_INTENT_NAMES
        for intent in intents
    )


def build_planner_request_payload(
    *,
    turn_id: str,
    user_text: str,
    turn_result,
    knowledge_context: str,
    planner_mode: str = 'default',
    max_history_entries: int = 6,
    active_goal_id: str = '',
) -> dict:
    """Build the planner ingress payload from the current turn result."""
    user_intent = _turn_user_intent(turn_result)
    resolved_intent = getattr(turn_result, 'intent', '')
    ack_text = _resolved_ack_text(user_intent, getattr(turn_result, 'verbal_ack', ''))
    ack_mode = _resolved_ack_mode(user_intent)
    dialogue_context = _bounded_dialogue_context(
        getattr(turn_result, 'updated_history', []),
        max_history_entries=max_history_entries,
    )
    request_kind = _resolved_request_kind(user_intent, resolved_intent)
    goal_id = _resolved_goal_id(
        user_intent=user_intent,
        turn_id=turn_id,
        active_goal_id=active_goal_id,
        request_kind=request_kind,
    )

    return {
        'request_id': str(turn_id).strip(),
        'goal_id': goal_id,
        'parent_goal_id': str(user_intent.get('parent_goal_id', '')).strip(),
        'supersedes_goal_id': str(user_intent.get('supersedes_goal_id', '')).strip(),
        'request_kind': request_kind,
        'user_text': str(user_text or '').strip(),
        'normalized_intents': _normalized_intents(resolved_intent),
        'ack_text': ack_text,
        'ack_mode': ack_mode,
        'scene_targets': _scene_targets_from_user_intent(user_intent),
        'dialogue_context': dialogue_context,
        'grounded_context': _grounded_context_payload(knowledge_context),
        'planner_mode': str(planner_mode or 'default').strip() or 'default',
        'interaction_mode': str(user_intent.get('interaction_mode', 'speech')).strip()
        or 'speech',
        'dialogue_turn_id': str(user_intent.get('dialogue_turn_id', turn_id)).strip()
        or str(turn_id).strip(),
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
    active_goal_id: str = '',
) -> Intent:
    """Create the ``Intent`` message published on ``/planner/request``."""
    payload = build_planner_request_payload(
        turn_id=turn_id,
        user_text=user_text,
        turn_result=turn_result,
        knowledge_context=knowledge_context,
        planner_mode=planner_mode,
        max_history_entries=max_history_entries,
        active_goal_id=active_goal_id,
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
    clean_intent = _normalize_token(intent_name)
    return [clean_intent] if clean_intent else []


def _scene_targets_from_user_intent(user_intent: dict) -> list[str]:
    scene_targets = _coerce_str_list(user_intent.get('scene_targets'))
    if scene_targets:
        return scene_targets
    scene_object = str(user_intent.get('object', '')).strip()
    if scene_object and _normalize_token(scene_object) not in _NON_SCENE_TARGET_OBJECTS:
        return [scene_object]
    return []


def _turn_user_intent(turn_result) -> dict:
    user_intent = getattr(turn_result, 'user_intent', {})
    if isinstance(user_intent, dict):
        return user_intent
    return {}


def _resolved_ack_text(user_intent: dict, verbal_ack: str) -> str:
    return str(user_intent.get('ack_text', '')).strip() or str(verbal_ack).strip()


def _resolved_ack_mode(user_intent: dict) -> str:
    return str(user_intent.get('ack_mode', '')).strip() or 'say'


def _grounded_context_payload(knowledge_context: str) -> dict:
    clean_knowledge_context = str(knowledge_context or '').strip()
    knowledge_snapshot = {}
    if clean_knowledge_context:
        knowledge_snapshot = {'summary_text': clean_knowledge_context}
    return {
        'knowledge_snapshot': knowledge_snapshot,
        'scene_summary': {},
        'world_model_snapshot': {},
        'world_model_text': '',
    }


def _resolved_request_kind(user_intent: dict, resolved_intent: str) -> str:
    explicit_kind = _normalize_token(user_intent.get('request_kind', ''))
    if explicit_kind in _PLANNER_REQUEST_KINDS:
        return explicit_kind

    normalized_intent = _normalize_token(resolved_intent)
    if normalized_intent in _CANCEL_INTENT_TYPES:
        return 'cancel_request'
    return 'new_goal'


def _resolved_goal_id(
    *,
    user_intent: dict,
    turn_id: str,
    active_goal_id: str,
    request_kind: str,
) -> str:
    explicit_goal_id = str(user_intent.get('goal_id', '')).strip()
    if explicit_goal_id:
        return explicit_goal_id
    if request_kind in {'goal_update', 'clarification_answer', 'cancel_request'}:
        clean_active_goal_id = str(active_goal_id or '').strip()
        if clean_active_goal_id:
            return clean_active_goal_id
    clean_turn_id = str(turn_id or '').strip()
    if clean_turn_id:
        normalized_turn_id = ''.join(
            char if char.isalnum() or char in ('_', '-') else '_'
            for char in clean_turn_id
        ).strip('_')
        if normalized_turn_id:
            return 'goal_%s' % normalized_turn_id
    return 'goal_unknown'


def _normalize_token(value) -> str:
    return str(value or '').strip().lower()


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
