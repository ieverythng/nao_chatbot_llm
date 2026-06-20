"""Internal Intent-Route-Response contract and deterministic admission guard."""

from __future__ import annotations

from dataclasses import dataclass
import re


IRR_VERSION = 'irr.v1'
SUPPORTED_ROUTES = {'dialogue', 'knowledge_query', 'execution'}
SUPPORTED_RESPONSE_STYLES = {'answer', 'acknowledgement', 'clarification', 'failure'}
SUPPORTED_REQUEST_KINDS = {
    'new_goal',
    'clarification_answer',
    'goal_update',
    'cancel_request',
    'none',
}
_ACTION_COMMITMENT_RE = re.compile(
    r"\b(?:i(?:'ll| will| am going to)|let me)\s+"
    r'(?:move|look|scan|navigate|walk|pick|grab|place|bring|wave|execute|do)\b',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardedIRR:
    """Normalized atomic decision accepted by the deterministic chatbot seam."""

    route: str
    route_reason: str
    confidence: float
    intent: dict
    response_text: str
    response_style: str
    planner_handoff_requested: bool
    evidence_used: dict
    safety_flags: tuple[str, ...]
    violations: tuple[str, ...]

    def user_intent(self) -> dict:
        payload = {
            'type': str(self.intent.get('type', '')).strip(),
            'goal_text': str(self.intent.get('goal_text', '')).strip(),
        }
        for key in ('request_kind', 'scene_targets', 'intent_sequence'):
            value = self.intent.get(key)
            if value not in (None, '', [], 'none'):
                payload[key] = value
        for key, value in dict(self.intent.get('arguments', {})).items():
            if str(key).strip() and str(value).strip():
                payload[str(key).strip()] = str(value).strip()
        return {key: value for key, value in payload.items() if value not in ('', [])}


def guard_irr_decision(
    payload: dict,
    *,
    turn_state: dict,
    fallback_response: str,
    enforce_canonical_targets: bool = True,
) -> GuardedIRR:
    """Normalize one model decision and prevent contradictory planner admission."""
    data = dict(payload or {})
    violations: list[str] = []
    safety_flags = _str_list(data.get('safety_flags', []))

    route = str(data.get('route', '')).strip().lower()
    if route not in SUPPORTED_ROUTES:
        violations.append('invalid_route')
        route = 'dialogue'

    intent = _normalize_intent(data.get('intent', {}))
    response = data.get('response', {})
    response = dict(response) if isinstance(response, dict) else {}
    response_text = str(response.get('text', '')).strip()
    response_style = str(response.get('style', '')).strip().lower()
    if response_style not in SUPPORTED_RESPONSE_STYLES:
        violations.append('invalid_response_style')
        response_style = 'answer' if route != 'execution' else 'acknowledgement'
    if not response_text:
        violations.append('missing_response_text')
        response_text = str(fallback_response or '').strip()

    handoff = data.get('planner_handoff', {})
    handoff = dict(handoff) if isinstance(handoff, dict) else {}
    requested = bool(handoff.get('requested', False))

    entity_ids = _turn_state_entity_ids(turn_state)
    raw_evidence = data.get('evidence_used', {})
    raw_evidence = dict(raw_evidence) if isinstance(raw_evidence, dict) else {}
    expected_grounding = str(
        turn_state.get('world_state', {}).get('grounding_id', '')
    ).strip()
    supplied_grounding = str(raw_evidence.get('grounding_id', '')).strip()
    if supplied_grounding and supplied_grounding != expected_grounding:
        violations.append('grounding_id_mismatch')
    evidence = _normalize_evidence(raw_evidence, turn_state)
    invalid_evidence = [item for item in evidence['entity_ids'] if item not in entity_ids]
    if invalid_evidence:
        violations.append('unknown_evidence_entity')
        evidence['entity_ids'] = [item for item in evidence['entity_ids'] if item in entity_ids]

    if route == 'dialogue' and _ACTION_COMMITMENT_RE.search(response_text):
        violations.append('dialogue_action_commitment')
        response_text = 'Could you clarify what you would like me to do?'
        response_style = 'clarification'

    if route == 'knowledge_query' and not _has_evidence(evidence):
        safety_flags.append('missing_evidence')

    if route == 'execution':
        missing_params, unknown_targets = ([], [])
        if enforce_canonical_targets:
            missing_params, unknown_targets = _validate_required_arguments(
                intent,
                turn_state=turn_state,
                entity_ids=entity_ids,
            )
        if missing_params or unknown_targets:
            if missing_params:
                violations.append('missing_required_arguments')
            if unknown_targets:
                violations.append('unknown_canonical_target')
            safety_flags.append('clarification_required')
            route = 'dialogue'
            requested = False
            response_style = 'clarification'
            response_text = _clarification_text(missing_params, unknown_targets)
        else:
            requested = True
            response_style = 'acknowledgement'
    elif requested:
        violations.append('handoff_requested_for_non_execution')
        requested = False

    return GuardedIRR(
        route=route,
        route_reason=str(data.get('route_reason', '')).strip() or 'unspecified',
        confidence=_confidence(data.get('confidence', 0.0)),
        intent=intent,
        response_text=response_text,
        response_style=response_style,
        planner_handoff_requested=requested,
        evidence_used=evidence,
        safety_flags=tuple(dict.fromkeys(safety_flags)),
        violations=tuple(dict.fromkeys(violations)),
    )


def _normalize_intent(value) -> dict:
    data = dict(value) if isinstance(value, dict) else {}
    request_kind = str(data.get('request_kind', 'none')).strip().lower() or 'none'
    if request_kind not in SUPPORTED_REQUEST_KINDS:
        request_kind = 'none'
    arguments = data.get('arguments', {})
    arguments = dict(arguments) if isinstance(arguments, dict) else {}
    return {
        'type': str(data.get('type', '')).strip(),
        'goal_text': str(data.get('goal_text', '')).strip(),
        'request_kind': request_kind,
        'scene_targets': _str_list(data.get('scene_targets', [])),
        'intent_sequence': _str_list(data.get('intent_sequence', [])),
        'arguments': {
            str(key).strip(): str(item).strip()
            for key, item in arguments.items()
            if str(key).strip() and str(item).strip()
        },
    }


def _normalize_evidence(value, turn_state: dict) -> dict:
    data = dict(value) if isinstance(value, dict) else {}
    grounding = str(data.get('grounding_id', '')).strip()
    expected = str(turn_state.get('world_state', {}).get('grounding_id', '')).strip()
    return {
        'grounding_id': grounding if grounding == expected else '',
        'entity_ids': _str_list(data.get('entity_ids', [])),
        'kb_subjects': _str_list(data.get('kb_subjects', [])),
        'latest_result_ids': _str_list(data.get('latest_result_ids', [])),
    }


def _validate_required_arguments(
    intent: dict,
    *,
    turn_state: dict,
    entity_ids: set[str],
) -> tuple[list[str], list[str]]:
    skills = list(intent.get('intent_sequence', [])) or [str(intent.get('type', '')).strip()]
    arguments = dict(intent.get('arguments', {}))
    manifest = {
        str(item.get('name', '')).strip(): item
        for item in turn_state.get('available_skills', [])
        if isinstance(item, dict) and str(item.get('name', '')).strip()
    }
    missing: list[str] = []
    unknown: list[str] = []
    for skill_name in skills:
        skill = manifest.get(skill_name, {})
        for param in _str_list(skill.get('required_params', [])):
            value = str(arguments.get(param, '')).strip()
            if not value:
                missing.append(param)
            elif param in {
                'object',
                'object_id',
                'recipient',
                'recipient_id',
                'target',
                'location',
                'target_area',
                'destination',
                'destination_id',
                'support',
            } and value not in entity_ids:
                unknown.append(value)
    return list(dict.fromkeys(missing)), list(dict.fromkeys(unknown))


def _turn_state_entity_ids(turn_state: dict) -> set[str]:
    return {
        str(item.get('id', '')).strip()
        for item in turn_state.get('world_state', {}).get('entities', [])
        if isinstance(item, dict) and str(item.get('id', '')).strip()
    }


def _clarification_text(missing: list[str], unknown: list[str]) -> str:
    if missing:
        return 'Which %s should I use?' % str(missing[0]).replace('_', ' ')
    if unknown:
        return 'I cannot match that target to the current scene. Which target do you mean?'
    return 'Could you clarify the target for that request?'


def _has_evidence(evidence: dict) -> bool:
    return bool(
        evidence.get('grounding_id')
        or evidence.get('entity_ids')
        or evidence.get('kb_subjects')
        or evidence.get('latest_result_ids')
    )


def _confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _str_list(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
