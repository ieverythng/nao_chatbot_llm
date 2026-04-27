"""Helpers for optional planner-mode handoff from ``chatbot_llm``."""

from __future__ import annotations

import json

from kb_skills.intent_labels import KB_QUERY_INTENTS
from planner_common import normalize_grounded_context

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
        'say',
        'greet',
        'raw_user_input',
        'identity',
        'wellbeing',
        'help',
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
_MOTION_OBJECT_INTENT_HINTS = {
    'stand': 'posture_stand',
    'standinit': 'posture_stand',
    'standfull': 'posture_stand',
    'standzero': 'posture_stand',
    'sit': 'posture_sit',
    'sitrelax': 'posture_sit',
    'kneel': 'posture_kneel',
    'crouch': 'posture_kneel',
    'head_center': 'head_center',
    'head_look_left': 'head_look_left',
    'head_look_right': 'head_look_right',
    'head_look_up': 'head_look_up',
    'head_look_down': 'head_look_down',
}
_MULTI_STEP_COORDINATION_MARKERS = (
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
)
_ACTION_HINT_TOKENS = (
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
)
_EXECUTABLE_PLAN_STEP_TYPES = {'say', 'skill', 'look_at', 'noop'}
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
_DEFAULT_PLANNER_PRIORITY = 128
_MIN_PLANNER_CONFIDENCE = 0.5


def should_route_intents_through_planner(
    intents: list[Intent],
    *,
    turn_result=None,
    user_text: str = '',
) -> bool:
    """Return true when the turn contains execution-oriented intents."""
    if _normalize_token(getattr(turn_result, 'route', '')) == 'execution':
        return True

    if _contains_execution_intent(intents):
        return True

    user_intent = _turn_user_intent(turn_result)
    if _has_executable_plan(user_intent):
        return True

    return _is_multi_step_turn(
        user_intent=user_intent,
        resolved_intent=getattr(turn_result, 'intent', ''),
        user_text=user_text,
    )


def build_planner_request_payload(
    *,
    turn_id: str,
    user_text: str,
    turn_result,
    knowledge_context: str,
    grounded_context: dict | None = None,
    planner_mode: str = 'auto',
    max_history_entries: int = 6,
    active_goal_id: str = '',
) -> dict:
    """Build the planner ingress payload from the current turn result."""
    user_intent = _turn_user_intent(turn_result)
    resolved_intent = getattr(turn_result, 'intent', '')
    resolved_planner_mode = _resolved_planner_mode(
        planner_mode=planner_mode,
        turn_result=turn_result,
        user_text=user_text,
    )
    ack_text = _resolved_ack_text(user_intent, getattr(turn_result, 'verbal_ack', ''))
    ack_mode = _resolved_ack_mode(user_intent)
    requested_plan = _requested_plan_from_user_intent(user_intent, ack_text=ack_text)
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
        'goal_text': _goal_text_from_user_intent(user_intent, user_text=user_text),
        'normalized_intents': _normalized_intents_for_turn(turn_result),
        'ack_text': ack_text,
        'ack_mode': ack_mode,
        'scene_targets': _scene_targets_from_user_intent(user_intent),
        'dialogue_context': dialogue_context,
        'requested_plan': requested_plan,
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
    max_history_entries: int = 6,
    active_goal_id: str = '',
) -> Intent:
    """Create the ``Intent`` message published on ``/planner/request``."""
    payload = build_planner_request_payload(
        turn_id=turn_id,
        user_text=user_text,
        turn_result=turn_result,
        knowledge_context=knowledge_context,
        grounded_context=grounded_context,
        planner_mode=planner_mode,
        max_history_entries=max_history_entries,
        active_goal_id=active_goal_id,
    )
    intent = Intent()
    intent.intent = str(planner_request_intent or 'planner_request').strip() or 'planner_request'
    intent.source = str(source_user_id).strip() or getattr(Intent, 'UNKNOWN_AGENT', 'unknown_agent')
    intent.modality = getattr(Intent, 'MODALITY_SPEECH', 'speech')
    intent.confidence = _planner_confidence(turn_result)
    intent.priority = _DEFAULT_PLANNER_PRIORITY
    intent.data = json.dumps(payload, separators=(',', ':'))
    return intent


def _bounded_dialogue_context(history: list[str], *, max_history_entries: int) -> list[str]:
    if not isinstance(history, list) or max_history_entries <= 0:
        return []
    return [str(item).strip() for item in history[-max_history_entries:] if str(item).strip()]


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
        if clean_candidate == 'fallback' and _coerce_plan(user_intent.get('plan')):
            continue
        if clean_candidate and clean_candidate not in normalized:
            normalized.append(clean_candidate)
    for hinted_intent in _plan_intent_hints(_coerce_plan(user_intent.get('plan'))):
        if hinted_intent not in normalized:
            normalized.append(hinted_intent)
    return normalized


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
) -> str:
    requested_mode = _normalize_token(planner_mode)
    if requested_mode not in ('', 'auto', 'default'):
        return requested_mode
    if _is_multi_step_turn(
        user_intent=_turn_user_intent(turn_result),
        resolved_intent=getattr(turn_result, 'intent', ''),
        user_text=user_text,
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
    if explicit_kind in _PLANNER_REQUEST_KINDS:
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


def _normalize_token(value) -> str:
    return str(value or '').strip().lower()


def _contains_execution_intent(intents: list[Intent]) -> bool:
    if not isinstance(intents, list) or not intents:
        return False
    return any(
        _normalize_token(getattr(intent, 'intent', '')) not in _NON_PLANNER_INTENT_NAMES
        for intent in intents
    )


def _has_executable_plan(user_intent: dict) -> bool:
    for step in _coerce_plan(user_intent.get('plan')):
        if _normalize_token(step.get('type', '')) in _EXECUTABLE_PLAN_STEP_TYPES:
            return True
    return False


def _is_multi_step_turn(*, user_intent: dict, resolved_intent: str, user_text: str) -> bool:
    if len(_coerce_plan(user_intent.get('plan'))) > 1:
        return True

    clean_text = ' %s ' % str(user_text or '').strip().lower()
    if not clean_text.strip():
        return False
    if not any(marker in clean_text for marker in _MULTI_STEP_COORDINATION_MARKERS):
        return False
    if not any(token in clean_text for token in _ACTION_HINT_TOKENS):
        return False

    clean_intent = _normalize_token(user_intent.get('type', '') or resolved_intent)
    if clean_intent and clean_intent not in _NON_PLANNER_INTENT_NAMES:
        return True
    return clean_intent in ('', 'fallback')


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_plan(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [step for step in value if isinstance(step, dict)]


def _requested_plan_from_user_intent(user_intent: dict, *, ack_text: str = '') -> list[dict]:
    requested_steps = []
    raw_plan = _coerce_plan(user_intent.get('plan'))
    if not raw_plan:
        return requested_steps

    clean_ack_text = str(ack_text or '').strip().lower()
    has_non_say_step = any(_normalize_token(step.get('type', '')) != 'say' for step in raw_plan)

    for step in raw_plan:
        step_type = _normalize_token(step.get('type', ''))
        step_name = str(step.get('name', '')).strip()
        step_args = step.get('args', {}) if isinstance(step.get('args'), dict) else {}

        if step_type == 'say' and has_non_say_step:
            step_text = str(step_args.get('text', step_args.get('object', ''))).strip().lower()
            if clean_ack_text and step_text == clean_ack_text:
                continue

        requested_steps.append(
            {
                'type': step_type,
                'name': step_name,
                'args': step_args,
            }
        )

    return requested_steps


def _plan_intent_hints(plan_steps: list[dict]) -> list[str]:
    hints: list[str] = []
    for step in plan_steps:
        step_type = _normalize_token(step.get('type', ''))
        step_name = _normalize_token(step.get('name', ''))
        step_args = step.get('args', {}) if isinstance(step.get('args'), dict) else {}

        if step_type == 'look_at':
            hints.append('look_at')
            continue

        if step_type != 'skill':
            continue

        if step_name == 'look_at':
            hints.append('look_at')
            continue

        if step_name not in ('', 'motion', 'perform_motion'):
            continue

        motion_object = _normalize_token(
            step_args.get('object', step_args.get('motion_name', ''))
        )
        hinted_intent = _MOTION_OBJECT_INTENT_HINTS.get(motion_object, '')
        if hinted_intent:
            hints.append(hinted_intent)
    return hints
