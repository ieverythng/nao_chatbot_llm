"""Translate local chatbot intent results to ``hri_actions_msgs/Intent``."""

from __future__ import annotations

import json

from hri_actions_msgs.msg import Intent


LOCAL_MOTION_MAP = {
    'posture_stand': 'stand',
    'posture_sit': 'sit',
    'posture_kneel': 'kneel',
    'head_center': 'head_center',
    'head_look_left': 'head_look_left',
    'head_look_right': 'head_look_right',
    'head_look_up': 'head_look_up',
    'head_look_down': 'head_look_down',
}

GENERIC_INTENT_MAP = {
    'greet': Intent.GREET,
    '__intent_greet__': Intent.GREET,
    'start_activity': Intent.START_ACTIVITY,
    '__intent_start_activity__': Intent.START_ACTIVITY,
    'stop_activity': Intent.STOP_ACTIVITY,
    '__intent_stop_activity__': Intent.STOP_ACTIVITY,
    'grab_object': Intent.GRAB_OBJECT,
    'pick_object': Intent.GRAB_OBJECT,
    '__intent_grab_object__': Intent.GRAB_OBJECT,
    'bring_object': Intent.BRING_OBJECT,
    '__intent_bring_object__': Intent.BRING_OBJECT,
    'place_object': Intent.PLACE_OBJECT,
    '__intent_place_object__': Intent.PLACE_OBJECT,
    'move_to': Intent.MOVE_TO,
    '__intent_move_to__': Intent.MOVE_TO,
    'guide': Intent.GUIDE,
    '__intent_guide__': Intent.GUIDE,
    'perform_motion': Intent.PERFORM_MOTION,
    '__intent_perform_motion__': Intent.PERFORM_MOTION,
    'wakeup': Intent.WAKEUP,
    '__intent_wakeup__': Intent.WAKEUP,
    'suspend': Intent.SUSPEND,
    '__intent_suspend__': Intent.SUSPEND,
}

RESPONSE_ONLY_INTENTS = {'identity', 'wellbeing', 'help'}
KB_QUERY_INTENTS = {
    'kb_query_visible_people',
    'kb_query_visible_objects',
    'kb_query_scene_change',
}
PLAN_STEP_TYPES = {'say', 'skill', 'look_at', 'noop'}


def build_response_intents(
    resolved_intent: str,
    user_intent: dict,
    source_user_id: str,
    verbal_ack: str,
    raw_input: str,
    confidence: float,
) -> list[Intent]:
    """Translate a backend turn result into HRI intents for dialogue_manager."""
    raw_type = str(user_intent.get('type', '')).strip().lower()
    normalized = raw_type or str(resolved_intent).strip().lower()
    source = str(source_user_id).strip() or Intent.UNKNOWN_AGENT

    if normalized in RESPONSE_ONLY_INTENTS:
        return []

    if normalized in KB_QUERY_INTENTS:
        payload = _clean_payload(user_intent)
        if verbal_ack:
            payload.setdefault('suggested_response', verbal_ack)
        _apply_ack_metadata(payload, verbal_ack)
        return [
            _make_intent(
                normalized,
                payload,
                source=source,
                confidence=confidence,
            )
        ]

    if normalized in LOCAL_MOTION_MAP:
        payload = {'object': LOCAL_MOTION_MAP[normalized]}
        _apply_ack_metadata(payload, verbal_ack)
        _apply_default_plan(Intent.PERFORM_MOTION, payload)
        return [
            _make_intent(
                Intent.PERFORM_MOTION,
                payload,
                source=source,
                confidence=confidence,
            )
        ]

    if normalized in GENERIC_INTENT_MAP:
        intent_name = GENERIC_INTENT_MAP[normalized]
        payload = _clean_payload(user_intent)
        _apply_ack_metadata(payload, verbal_ack)
        _apply_scene_targets(payload)

        if intent_name == Intent.PERFORM_MOTION and 'object' not in payload:
            motion_object = LOCAL_MOTION_MAP.get(str(resolved_intent).strip().lower(), '')
            if motion_object:
                payload['object'] = motion_object

        _apply_default_plan(intent_name, payload)

        if intent_name == Intent.GREET and verbal_ack:
            payload.setdefault('suggested_response', verbal_ack)

        if intent_name == Intent.SAY:
            if not payload.get('object') and verbal_ack:
                payload['object'] = verbal_ack
            if not payload.get('object'):
                return []

        return [
            _make_intent(
                intent_name,
                payload,
                source=source,
                confidence=confidence,
            )
        ]

    if normalized == 'fallback':
        fallback_payload = {'input': str(raw_input).strip()}
        if verbal_ack:
            fallback_payload['suggested_response'] = verbal_ack
        _apply_ack_metadata(fallback_payload, verbal_ack)
        return [
            _make_intent(
                Intent.RAW_USER_INPUT,
                fallback_payload,
                source=source,
                confidence=confidence,
            )
        ]

    if str(resolved_intent).strip().lower() in LOCAL_MOTION_MAP:
        payload = {'object': LOCAL_MOTION_MAP[str(resolved_intent).strip().lower()]}
        _apply_ack_metadata(payload, verbal_ack)
        _apply_default_plan(Intent.PERFORM_MOTION, payload)
        return [
            _make_intent(
                Intent.PERFORM_MOTION,
                payload,
                source=source,
                confidence=confidence,
            )
        ]

    return []


def _clean_payload(user_intent: dict) -> dict:
    payload = {}
    if not isinstance(user_intent, dict):
        return payload
    for key in (
        'object',
        'recipient',
        'input',
        'goal',
        'suggested_response',
        'ack_text',
        'ack_mode',
    ):
        value = str(user_intent.get(key, '')).strip()
        if value:
            payload[key] = value
    scene_targets = _coerce_str_list(user_intent.get('scene_targets'))
    if scene_targets:
        payload['scene_targets'] = scene_targets
    plan = _coerce_plan(user_intent.get('plan'))
    if plan:
        payload['plan'] = plan
    return payload


def _make_intent(intent_name: str, data: dict, source: str, confidence: float) -> Intent:
    intent = Intent()
    intent.intent = intent_name
    intent.source = source
    intent.modality = Intent.MODALITY_SPEECH
    intent.confidence = float(confidence)
    intent.priority = 0
    intent.data = json.dumps(data or {}, separators=(',', ':'))
    return intent


def _apply_ack_metadata(payload: dict, verbal_ack: str) -> None:
    clean_ack = str(payload.get('ack_text', '')).strip() or str(verbal_ack).strip()
    if clean_ack:
        payload['ack_text'] = clean_ack
        payload.setdefault('ack_mode', 'say')


def _apply_scene_targets(payload: dict) -> None:
    if payload.get('scene_targets'):
        return
    scene_target = str(payload.get('object', '')).strip()
    if scene_target and scene_target not in set(LOCAL_MOTION_MAP.values()) | {'look_at_reset'}:
        payload['scene_targets'] = [scene_target]


def _apply_default_plan(intent_name: str, payload: dict) -> None:
    if payload.get('plan'):
        return
    if intent_name == Intent.PERFORM_MOTION and payload.get('object'):
        payload['plan'] = [
            {
                'type': 'skill',
                'name': 'perform_motion',
                'args': {'object': payload['object']},
            }
        ]


def _coerce_plan(raw_plan) -> list[dict]:
    if not isinstance(raw_plan, list):
        return []
    cleaned_steps = []
    for step in raw_plan:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get('type', '')).strip().lower()
        if step_type not in PLAN_STEP_TYPES:
            continue
        step_name = str(step.get('name', '')).strip().lower()
        raw_args = step.get('args', {})
        args = {}
        if isinstance(raw_args, dict):
            for key, value in raw_args.items():
                if value not in (None, '', []):
                    args[str(key)] = value
        cleaned_steps.append(
            {
                'type': step_type,
                'name': step_name,
                'args': args,
            }
        )
    return cleaned_steps


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        cleaned = [item.strip() for item in value.split(',') if item.strip()]
        return cleaned
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
