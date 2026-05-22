"""Helpers for optional planner-mode handoff from ``chatbot_llm``."""

from __future__ import annotations

from dataclasses import asdict
import json

from kb_skills.intent_labels import KB_QUERY_INTENTS
from chatbot_llm.prompt_pack import default_prompt_pack
from planner_common import PLANNER_REQUEST_KINDS
from planner_common import PlannerRequest
from planner_common import extract_json_object
from planner_common import IntentLabels
from planner_common import is_perform_motion_object_label
from planner_common import normalize_grounded_context

try:  # pragma: no cover - ROS runtime dependency
    from hri_actions_msgs.msg import Intent
except ImportError:  # pragma: no cover - import-light unit tests
    class Intent:  # type: ignore[no-redef]
        BRING_OBJECT = IntentLabels.BRING_OBJECT
        GRAB_OBJECT = IntentLabels.GRAB_OBJECT
        GREET = IntentLabels.GREET
        GUIDE = IntentLabels.GUIDE
        MOVE_TO = IntentLabels.MOVE_TO
        PERFORM_MOTION = IntentLabels.PERFORM_MOTION
        PLACE_OBJECT = IntentLabels.PLACE_OBJECT
        PRESENT_CONTENT = IntentLabels.PRESENT_CONTENT
        RAW_USER_INPUT = IntentLabels.RAW_USER_INPUT
        SAY = IntentLabels.SAY
        START_ACTIVITY = IntentLabels.START_ACTIVITY
        STOP_ACTIVITY = IntentLabels.STOP_ACTIVITY
        SUSPEND = IntentLabels.SUSPEND
        WAKEUP = IntentLabels.WAKEUP
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
        'say',
        'greet',
        'raw_user_input',
        'identity',
        'wellbeing',
        'help',
    )
}
_CANCEL_INTENT_TYPES = {
    'cancel',
    'cancel_request',
    'stop',
    'stop_activity',
    'suspend',
}
_DEFAULT_PLANNER_PRIORITY = 128
_MIN_PLANNER_CONFIDENCE = 0.5
_MAX_ACK_PARSE_DEPTH = 3


def should_route_intents_through_planner(
    intents: list[Intent],
    *,
    turn_result=None,
    user_text: str = '',
    multi_step_heuristics: dict | None = None,
) -> bool:
    """Return true when the turn contains execution-oriented intents."""
    if _normalize_token(getattr(turn_result, 'route', '')) == 'execution':
        return True

    if _contains_execution_intent(intents):
        return True

    user_intent = _turn_user_intent(turn_result)
    return _is_multi_step_turn(
        user_intent=user_intent,
        resolved_intent=getattr(turn_result, 'intent', ''),
        user_text=user_text,
        heuristics=multi_step_heuristics,
    )


def build_planner_request_payload(
    *,
    turn_id: str,
    user_text: str,
    turn_result,
    knowledge_context: str,
    grounded_context: dict | None = None,
    planner_mode: str = 'auto',
    multi_step_heuristics: dict | None = None,
    max_history_entries: int = 6,
    active_goal_id: str = '',
    active_goal_token: str = '',
) -> dict:
    """Build the planner ingress payload from the current turn result."""
    user_intent = _turn_user_intent(turn_result)
    resolved_intent = getattr(turn_result, 'intent', '')
    resolved_planner_mode = _resolved_planner_mode(
        planner_mode=planner_mode,
        turn_result=turn_result,
        user_text=user_text,
        multi_step_heuristics=multi_step_heuristics,
    )
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
    goal_token = _resolved_goal_token(
        user_intent=user_intent,
        turn_id=turn_id,
        goal_id=goal_id,
        request_kind=request_kind,
        active_goal_token=active_goal_token,
    )

    payload = {
        'request_id': str(turn_id).strip(),
        'goal_id': goal_id,
        'goal_token': goal_token,
        'parent_goal_id': str(user_intent.get('parent_goal_id', '')).strip(),
        'supersedes_goal_id': _resolved_supersedes_goal_id(
            user_intent=user_intent,
            request_kind=request_kind,
            active_goal_id=active_goal_id,
            goal_id=goal_id,
        ),
        'request_kind': request_kind,
        'goal_text': _goal_text_from_user_intent(user_intent, user_text=user_text),
        'normalized_intents': _normalized_intents_for_turn(turn_result),
        'ack_text': ack_text,
        'ack_mode': ack_mode,
        'scene_targets': _scene_targets_from_user_intent(user_intent),
        'dialogue_context': dialogue_context,
        'requested_plan': [],
        'grounded_context': _grounded_context_payload(
            knowledge_context,
            grounded_context=grounded_context,
        ),
        'planner_mode': resolved_planner_mode,
        'interaction_mode': str(user_intent.get('interaction_mode', 'speech')).strip()
        or 'speech',
        'dialogue_turn_id': str(user_intent.get('dialogue_turn_id', turn_id)).strip()
        or str(turn_id).strip(),
    }
    return _planner_request_payload(payload)


def build_planner_request_intent(
    *,
    turn_id: str,
    user_text: str,
    source_user_id: str,
    turn_result,
    knowledge_context: str,
    grounded_context: dict | None = None,
    planner_request_intent: str = 'planner_request',
    planner_mode: str = 'auto',
    multi_step_heuristics: dict | None = None,
    max_history_entries: int = 6,
    active_goal_id: str = '',
    active_goal_token: str = '',
) -> Intent:
    """Create the ``Intent`` message published on ``/planner/request``."""
    payload = build_planner_request_payload(
        turn_id=turn_id,
        user_text=user_text,
        turn_result=turn_result,
        knowledge_context=knowledge_context,
        grounded_context=grounded_context,
        planner_mode=planner_mode,
        multi_step_heuristics=multi_step_heuristics,
        max_history_entries=max_history_entries,
        active_goal_id=active_goal_id,
        active_goal_token=active_goal_token,
    )
    return build_planner_request_intent_from_payload(
        payload=payload,
        source_user_id=source_user_id,
        planner_request_intent=planner_request_intent,
        confidence=_planner_confidence(turn_result),
    )


def build_planner_request_intent_from_payload(
    *,
    payload: dict,
    source_user_id: str,
    planner_request_intent: str = 'planner_request',
    confidence: float = _MIN_PLANNER_CONFIDENCE,
) -> Intent:
    """Create a planner ``Intent`` from an already-normalized request payload."""
    intent = Intent()
    intent.intent = str(planner_request_intent or 'planner_request').strip() or 'planner_request'
    intent.source = str(source_user_id).strip() or getattr(Intent, 'UNKNOWN_AGENT', 'unknown_agent')
    intent.modality = getattr(Intent, 'MODALITY_SPEECH', 'speech')
    intent.confidence = max(0.0, min(1.0, float(confidence)))
    intent.priority = _DEFAULT_PLANNER_PRIORITY
    intent.data = json.dumps(payload, separators=(',', ':'))
    return intent


def _planner_request_payload(payload: dict) -> dict:
    request = PlannerRequest.from_payload(payload)
    normalized = _jsonable(asdict(request))
    normalized.pop('user_text', None)
    return normalized


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _bounded_dialogue_context(history: list[str], *, max_history_entries: int) -> list[str]:
    if not isinstance(history, list) or max_history_entries <= 0:
        return []
    sanitized_history = []
    for item in history[-max_history_entries:]:
        clean_item = _sanitize_dialogue_history_entry(str(item).strip())
        if clean_item:
            sanitized_history.append(clean_item)
    return sanitized_history


def _sanitize_dialogue_history_entry(entry: str) -> str:
    if not entry:
        return ''
    role, separator, content = entry.partition(':')
    if separator != ':' or role.strip().lower() != 'assistant':
        return entry

    clean_content = _extract_assistant_ack_text(content.strip())
    if not clean_content:
        return entry
    return f'assistant:{clean_content}'


def _extract_assistant_ack_text(payload: str, *, _depth: int = 0) -> str:
    if not payload or _depth > _MAX_ACK_PARSE_DEPTH:
        return ''

    parsed = extract_json_object(payload)
    if parsed:
        for key in ('verbal_ack', 'ack_text'):
            text = str(parsed.get(key, '')).strip()
            if text:
                return text
        response_text = str(parsed.get('response', '')).strip()
        if response_text:
            nested_text = _extract_assistant_ack_text(response_text, _depth=_depth + 1)
            if nested_text:
                return nested_text
            return response_text
    return ''


def _normalized_intents(intent_name: str) -> list[str]:
    clean_intent = _normalize_token(intent_name)
    return [clean_intent] if clean_intent else []


def _normalized_intents_for_turn(turn_result) -> list[str]:
    user_intent = _turn_user_intent(turn_result)
    candidates = [
        user_intent.get('type', ''),
        getattr(turn_result, 'intent', ''),
    ]
    normalized = []
    for candidate in candidates:
        clean_candidate = _normalize_token(candidate)
        if clean_candidate and clean_candidate not in normalized:
            normalized.append(clean_candidate)
    return normalized


def _scene_targets_from_user_intent(user_intent: dict) -> list[str]:
    scene_targets = _coerce_str_list(user_intent.get('scene_targets'))
    if scene_targets:
        return scene_targets
    scene_object = str(user_intent.get('object', '')).strip()
    if scene_object and not is_perform_motion_object_label(scene_object):
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


def _goal_text_from_user_intent(user_intent: dict, *, user_text: str) -> str:
    for key in ('goal_text', 'goal', 'task'):
        clean_value = str(user_intent.get(key, '')).strip()
        if clean_value:
            return clean_value
    return str(user_text or '').strip()


def _planner_confidence(turn_result) -> float:
    try:
        confidence = float(getattr(turn_result, 'intent_confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 0.0 and _normalize_token(getattr(turn_result, 'route', '')) == 'execution':
        return _MIN_PLANNER_CONFIDENCE
    return max(0.0, min(1.0, confidence))


def _resolved_planner_mode(
    *,
    planner_mode: str,
    turn_result,
    user_text: str,
    multi_step_heuristics: dict | None,
) -> str:
    requested_mode = _normalize_token(planner_mode)
    if requested_mode not in ('', 'auto', 'default'):
        return requested_mode
    if _is_multi_step_turn(
        user_intent=_turn_user_intent(turn_result),
        resolved_intent=getattr(turn_result, 'intent', ''),
        user_text=user_text,
        heuristics=multi_step_heuristics,
    ):
        return 'multi_step'
    return 'default'


def _grounded_context_payload(
    knowledge_context: str,
    *,
    grounded_context: dict | None = None,
) -> dict:
    payload = normalize_grounded_context(grounded_context or {})
    clean_knowledge_context = str(knowledge_context or '').strip()
    if not clean_knowledge_context:
        return payload

    knowledge_snapshot = dict(payload.get('knowledge_snapshot', {}))
    summary_text = str(knowledge_snapshot.get('summary_text', '')).strip()
    if not summary_text:
        knowledge_snapshot['summary_text'] = clean_knowledge_context
    payload['knowledge_snapshot'] = knowledge_snapshot
    return payload


def _resolved_request_kind(user_intent: dict, resolved_intent: str) -> str:
    explicit_kind = _normalize_token(user_intent.get('request_kind', ''))
    if explicit_kind in PLANNER_REQUEST_KINDS:
        return explicit_kind

    user_intent_type = _normalize_token(user_intent.get('type', ''))
    if user_intent_type in _CANCEL_INTENT_TYPES or _normalize_token(resolved_intent) in _CANCEL_INTENT_TYPES:
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


def _resolved_goal_token(
    *,
    user_intent: dict,
    turn_id: str,
    goal_id: str,
    request_kind: str,
    active_goal_token: str,
) -> str:
    explicit_goal_token = str(user_intent.get('goal_token', '')).strip()
    if explicit_goal_token:
        return explicit_goal_token
    if request_kind in {'goal_update', 'clarification_answer', 'cancel_request'}:
        clean_active_goal_token = str(active_goal_token or '').strip()
        if clean_active_goal_token:
            return clean_active_goal_token
        clean_active_goal_id = str(goal_id or '').strip()
        if clean_active_goal_id:
            return f'{clean_active_goal_id}:active'
    clean_goal_id = str(goal_id or '').strip()
    clean_turn_id = str(turn_id or '').strip()
    if clean_goal_id and clean_turn_id:
        return f'{clean_goal_id}:{clean_turn_id}'
    return clean_goal_id


def _resolved_supersedes_goal_id(
    *,
    user_intent: dict,
    request_kind: str,
    active_goal_id: str,
    goal_id: str,
) -> str:
    explicit_supersedes_goal_id = str(user_intent.get('supersedes_goal_id', '')).strip()
    if explicit_supersedes_goal_id:
        return explicit_supersedes_goal_id
    if request_kind != 'new_goal':
        return ''
    clean_active_goal_id = str(active_goal_id or '').strip()
    clean_goal_id = str(goal_id or '').strip()
    if clean_active_goal_id and clean_active_goal_id != clean_goal_id:
        return clean_active_goal_id
    return ''


def _normalize_token(value) -> str:
    return str(value or '').strip().lower()


def _contains_execution_intent(intents: list[Intent]) -> bool:
    if not isinstance(intents, list) or not intents:
        return False
    return any(
        _normalize_token(getattr(intent, 'intent', '')) not in _NON_PLANNER_INTENT_NAMES
        for intent in intents
    )


def _is_multi_step_turn(
    *,
    user_intent: dict,
    resolved_intent: str,
    user_text: str,
    heuristics: dict | None = None,
) -> bool:
    clean_text = ' %s ' % str(user_text or '').strip().lower()
    if not clean_text.strip():
        return False
    coordination_markers = _heuristic_values(heuristics, 'coordination_markers')
    action_hint_tokens = _heuristic_values(heuristics, 'action_hint_tokens')
    if not any(marker in clean_text for marker in coordination_markers):
        return False
    if not any(token in clean_text for token in action_hint_tokens):
        return False

    clean_intent = _normalize_token(user_intent.get('type', '') or resolved_intent)
    if clean_intent and clean_intent not in _NON_PLANNER_INTENT_NAMES:
        return True
    return clean_intent in ('', 'fallback')


def _heuristic_values(heuristics: dict | None, key: str) -> list[str]:
    source = heuristics if isinstance(heuristics, dict) else None
    if source is None:
        source = default_prompt_pack().planner_multi_step_heuristics
    value = source.get(key, [])
    if isinstance(value, str):
        return [value.lower()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).lower() for item in value if str(item)]


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
