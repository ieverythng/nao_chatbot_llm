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

    if normalized in LOCAL_MOTION_MAP:
        return [
            _make_intent(
                Intent.PERFORM_MOTION,
                {'object': LOCAL_MOTION_MAP[normalized]},
                source=source,
                confidence=confidence,
            )
        ]

    if normalized in GENERIC_INTENT_MAP:
        intent_name = GENERIC_INTENT_MAP[normalized]
        payload = _clean_payload(user_intent)

        if intent_name == Intent.PERFORM_MOTION and 'object' not in payload:
            motion_object = LOCAL_MOTION_MAP.get(str(resolved_intent).strip().lower(), '')
            if motion_object:
                payload['object'] = motion_object

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
        return [
            _make_intent(
                Intent.RAW_USER_INPUT,
                fallback_payload,
                source=source,
                confidence=confidence,
            )
        ]

    if str(resolved_intent).strip().lower() in LOCAL_MOTION_MAP:
        return [
            _make_intent(
                Intent.PERFORM_MOTION,
                {'object': LOCAL_MOTION_MAP[str(resolved_intent).strip().lower()]},
                source=source,
                confidence=confidence,
            )
        ]

    return []


def _clean_payload(user_intent: dict) -> dict:
    payload = {}
    if not isinstance(user_intent, dict):
        return payload
    for key in ('object', 'recipient', 'input', 'goal'):
        value = str(user_intent.get(key, '')).strip()
        if value:
            payload[key] = value
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
