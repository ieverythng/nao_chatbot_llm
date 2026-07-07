"""Response parsing and bounded spoken-text fallbacks for chatbot turns.

This module owns two related concerns that previously lived inline in
``turn_engine``:

- parsing model output into a usable verbal acknowledgement (JSON extraction,
  ack coercion, user-intent coercion, length limiting);
- producing bounded, route-safe spoken text when the model output is missing,
  unsafe, or must be post-processed (execution-ack sanitisation, planner
  completion/dialogue acks, and execution-report wording).

These helpers must never fabricate execution success or perception facts; they
only word what the structured payloads already assert, or fall back to a short
safe phrase.
"""

from __future__ import annotations

import json
import re

from chatbot_llm.route_heuristics import _EXECUTION_ROUTE
from chatbot_llm.route_heuristics import _KNOWLEDGE_QUERY_ROUTE
from chatbot_llm.route_heuristics import _route_is_contradictory


_MAX_VERBAL_ACK_CHARS = 900
_DIALOGUE_ROUTE_FALLBACK_ACK = 'I can talk about that without starting a robot action.'
_KNOWLEDGE_QUERY_ROUTE_FALLBACK_ACK = 'I will answer from the current grounded context.'


# ---------------------------------------------------------------------------
# Model-output parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_dict(payload: str) -> dict:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, str):
        return _parse_json_dict(parsed)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _extract_json_object(payload: str) -> dict:
    parsed = _parse_json_dict(payload)
    if parsed:
        return parsed

    decoder = json.JSONDecoder()
    for start in range(len(payload)):
        if payload[start] != '{':
            continue
        try:
            maybe_obj, _ = decoder.raw_decode(payload[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(maybe_obj, dict):
            return maybe_obj
    return {}


def _extract_ack_text(payload: str, _depth: int = 0) -> str:
    if _depth > 3:
        return ''
    parsed = _extract_json_object(payload)
    if parsed:
        for key in ('verbal_ack', 'ack_text'):
            text = str(parsed.get(key, '')).strip()
            if text:
                return text
        for value in parsed.values():
            if isinstance(value, dict):
                nested_text = _extract_ack_text(json.dumps(value), _depth + 1)
                if nested_text:
                    return nested_text
            elif isinstance(value, str):
                nested_text = _extract_ack_text(value, _depth + 1)
                if nested_text:
                    return nested_text
        response_text = str(parsed.get('response', '')).strip()
        if response_text:
            nested_text = _extract_ack_text(response_text, _depth + 1)
            if nested_text:
                return nested_text
            return response_text
        return ''

    # Be permissive when the model drifts into "almost JSON" formatting.
    for key in ('verbal_ack', 'ack_text', 'response'):
        patterns = (
            rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf"'{key}'\s*:\s*'((?:[^'\\]|\\.)*)'",
            rf'\b{key}\b\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf"\b{key}\b\s*:\s*'((?:[^'\\]|\\.)*)'",
        )
        for pattern in patterns:
            match = re.search(pattern, payload, re.DOTALL)
            if not match:
                continue
            text = bytes(match.group(1), 'utf-8').decode('unicode_escape').strip()
            if not text:
                continue
            if key == 'response':
                nested_text = _extract_ack_text(text, _depth + 1)
                if nested_text:
                    return nested_text
            return text
    return ''


def _ack_from_parsed_response(parsed: dict) -> str:
    for key in ('verbal_ack', 'ack_text'):
        text = str(parsed.get(key, '')).strip()
        if text:
            return text
    response_text = str(parsed.get('response', '')).strip()
    if response_text:
        nested_text = _extract_ack_text(response_text)
        return nested_text or response_text
    return ''


def _limit_verbal_ack(verbal_ack: str) -> str:
    text = ' '.join(str(verbal_ack or '').strip().split())
    if len(text) <= _MAX_VERBAL_ACK_CHARS:
        return text
    boundary = max(text.rfind('.', 0, _MAX_VERBAL_ACK_CHARS), text.rfind('?', 0, _MAX_VERBAL_ACK_CHARS))
    if boundary < int(_MAX_VERBAL_ACK_CHARS * 0.55):
        boundary = text.rfind(' ', 0, _MAX_VERBAL_ACK_CHARS)
    if boundary < int(_MAX_VERBAL_ACK_CHARS * 0.55):
        boundary = _MAX_VERBAL_ACK_CHARS
    summary = text[:boundary].strip(' ,;:.')
    return '%s. I can continue if you want more detail.' % summary


def _looks_like_json_payload(payload: str) -> bool:
    clean_payload = str(payload or '').strip()
    return clean_payload.startswith(('{', '"{', '```json', '```'))


def _coerce_user_intent(user_intent) -> dict:
    if isinstance(user_intent, dict):
        cleaned = {}
        for key in (
            'type',
            'object',
            'recipient',
            'input',
            'goal',
            'goal_text',
            'task',
            'ack_text',
            'request_kind',
            'goal_id',
            'parent_goal_id',
            'supersedes_goal_id',
            'dialogue_turn_id',
        ):
            value = str(user_intent.get(key, '')).strip()
            if value:
                cleaned[key] = value
        scene_targets = user_intent.get('scene_targets')
        if isinstance(scene_targets, str):
            parsed_targets = [item.strip() for item in scene_targets.split(',') if item.strip()]
            if parsed_targets:
                cleaned['scene_targets'] = parsed_targets
        elif isinstance(scene_targets, (list, tuple)):
            parsed_targets = [str(item).strip() for item in scene_targets if str(item).strip()]
            if parsed_targets:
                cleaned['scene_targets'] = parsed_targets

        intent_sequence = user_intent.get('intent_sequence')
        if isinstance(intent_sequence, str):
            parsed_intents = [
                item.strip() for item in intent_sequence.split(',') if item.strip()
            ]
            if parsed_intents:
                cleaned['intent_sequence'] = parsed_intents
        elif isinstance(intent_sequence, (list, tuple)):
            parsed_intents = [
                str(item).strip() for item in intent_sequence if str(item).strip()
            ]
            if parsed_intents:
                cleaned['intent_sequence'] = parsed_intents

        return cleaned
    if isinstance(user_intent, str) and user_intent.strip():
        return {'type': user_intent.strip()}
    return {}


# ---------------------------------------------------------------------------
# Route-safe acknowledgement sanitisers
# ---------------------------------------------------------------------------

def _sanitize_execution_ack(verbal_ack: str) -> str:
    """Keep execution acknowledgements from claiming the result before execution."""
    clean_ack = str(verbal_ack or '').strip()
    if not clean_ack:
        return 'Okay, I will do that.'
    if clean_ack == 'fallback':
        return clean_ack

    clean_ack = re.sub(
        r'\s*\((?:[^)]*\b(?:perform|performs|performed|move|moves|moved|'
        r'scan|scans|scanned|look|looks|looked|turn|turns|turned|wave|waves|'
        r'waved|navigate|navigates|navigated)[^)]*)\)',
        '',
        clean_ack,
        flags=re.IGNORECASE,
    ).strip()

    lowered = clean_ack.lower()
    if re.search(
        r"\b(i cannot|i can't|i could not|i couldn't|i do not have|i don't have|unable to)\b",
        lowered,
    ):
        return 'Okay, I will try that now.'

    result_markers = (
        'i can currently',
        'i currently',
        'i can see',
        'i see',
        'i have scanned',
        'i found',
        'i have found',
        'i detected',
        'i have detected',
        'i observed',
        'i have observed',
        'i am now',
        "i'm now",
        'i have arrived',
        "i've arrived",
        'i arrived',
        'i performed',
        'i have moved',
        'i have completed',
        'i completed',
        'i scanned',
        'the scan',
    )
    split_at = len(clean_ack)
    for marker in result_markers:
        match = re.search(
            r'(^|[.!?]\s+)%s\b' % re.escape(marker.lower()),
            lowered,
        )
        if match:
            split_at = min(split_at, match.start(1))
    clean_ack = clean_ack[:split_at].strip()
    clean_ack = re.sub(
        r'(?:\s*(?:,|;|:)?\s*\b(?:and|but|so)\b)+$',
        '',
        clean_ack,
        flags=re.IGNORECASE,
    ).strip()

    if not clean_ack:
        return 'Okay, I will do that.'
    if clean_ack[-1] not in '.!?':
        clean_ack += '.'
    return clean_ack


def _route_safe_fallback_ack(route: str) -> str:
    if route == _EXECUTION_ROUTE:
        return 'Okay, I will try that now.'
    if route == _KNOWLEDGE_QUERY_ROUTE:
        return _KNOWLEDGE_QUERY_ROUTE_FALLBACK_ACK
    return _DIALOGUE_ROUTE_FALLBACK_ACK


def _sanitize_locked_route_ack(
    *,
    route: str,
    user_text: str,
    verbal_ack: str,
) -> str:
    if route == _EXECUTION_ROUTE:
        return _sanitize_execution_ack(verbal_ack)
    if not _route_is_contradictory(
        user_text=user_text,
        verbal_ack=verbal_ack,
        route=route,
    ):
        return str(verbal_ack or '').strip()
    return _route_safe_fallback_ack(route)


# ---------------------------------------------------------------------------
# Planner completion / dialogue fallbacks
# ---------------------------------------------------------------------------

def _fallback_planner_completion_ack(completion_context: dict) -> str:
    text_hint = str(completion_context.get('text_hint', '')).strip()
    if text_hint:
        return text_hint
    result_summary = str(completion_context.get('result_summary', '')).strip()
    if result_summary:
        return result_summary
    result_payload = completion_context.get('result_payload', {})
    if isinstance(result_payload, dict):
        summary_text = str(result_payload.get('summary_text', '')).strip()
        if summary_text:
            return summary_text
    goal_text = str(completion_context.get('goal_text', '')).strip()
    if goal_text:
        return 'I completed the requested task.'
    return ''


def _friendly_step_label(step_name: str) -> str:
    clean = str(step_name or '').strip().lower()
    labels = {
        'look_at': 'looked at the target',
        'navigate_to': 'navigated to the target',
        'walk_to': 'walked to the target',
        'wave_greet': 'waved',
        'perform_motion': 'performed the motion',
        'scan': 'looked around',
        'inspect_scene': 'inspected the scene',
        'find_object': 'looked for the object',
        'pick_object': 'picked up the object',
        'place_object': 'placed the object',
        'bring_object': 'brought the object',
    }
    if clean in labels:
        return labels[clean]
    return clean.replace('_', ' ') if clean else 'completed the step'


def _fallback_execution_report_ack(report_context: dict) -> str:
    successful_steps = [
        step for step in report_context.get('steps', [])
        if isinstance(step, dict) and str(step.get('status', '')).strip().lower() == 'succeeded'
    ]
    outcome_report = _completed_target_outcome_report(report_context)
    if outcome_report:
        return outcome_report
    latest_summary = str(report_context.get('latest_result_summary', '')).strip()
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate':
        if latest_summary:
            return latest_summary
        if successful_steps:
            latest_step_summary = str(successful_steps[-1].get('result_summary', '')).strip()
            if latest_step_summary:
                return latest_step_summary

    if successful_steps:
        navigation_report = _ordered_navigation_report(successful_steps, report_context)
        if navigation_report:
            return navigation_report
        summaries = [
            str(step.get('result_summary', '')).strip()
            for step in successful_steps
            if str(step.get('result_summary', '')).strip()
            and not _is_placeholder_report_summary(str(step.get('result_summary', '')).strip())
        ]
        if summaries:
            return ' '.join(summaries[-2:])

    if latest_summary:
        return latest_summary
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        summary_text = str(latest_payload.get('summary_text', '')).strip()
        if summary_text:
            return summary_text
    step_names = [
        _friendly_step_label(str(step.get('name', '')).strip())
        for step in successful_steps[-2:]
        if isinstance(step, dict) and str(step.get('name', '')).strip()
    ]
    if step_names:
        if len(step_names) == 1:
            return 'I %s.' % step_names[0]
        return 'I %s and then %s.' % (step_names[0], step_names[1])
    if str(report_context.get('goal_text', '')).strip():
        return 'I completed the requested task.'
    return ''


def _completed_target_outcome_report(report_context: dict) -> str:
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate':
        return ''
    outcome = report_context.get('plan_outcome_summary', {})
    if not isinstance(outcome, dict):
        return ''
    completed_targets = _clean_unique_list(outcome.get('completed_targets', []))
    delivery_objects = _completed_delivery_objects(report_context)
    is_delivery = _is_delivery_report_context(report_context, delivery_objects=delivery_objects)
    report_targets = delivery_objects or completed_targets
    if len(report_targets) < 2:
        return ''
    failed_targets = _clean_unique_list(outcome.get('failed_targets', []))
    pending_targets = _clean_unique_list(outcome.get('pending_targets', []))
    labels = [_friendly_target_label(target) for target in report_targets]
    recipient = _latest_delivery_recipient(report_context)
    if recipient and is_delivery:
        report = 'I brought %s to %s.' % (_natural_join(labels), _friendly_recipient_label(recipient))
    else:
        report = 'I completed %s.' % _natural_join(labels)
    if failed_targets:
        report += ' I could not complete %s.' % _natural_join(
            [_friendly_target_label(target) for target in failed_targets]
        )
    if pending_targets:
        report += ' %s still pending.' % _natural_join(
            [_friendly_target_label(target) for target in pending_targets]
        )
    return report


def _postprocess_execution_report_ack(candidate: str, report_context: dict) -> str:
    """Reject report wording that duplicates or promises a future report."""
    clean = str(candidate or '').strip()
    if not clean:
        return ''
    lowered = clean.lower()
    if any(
        marker in lowered
        for marker in (
            'i finished:',
            'will report back',
            'will report on',
            'will let you know',
            'will tell you',
            'can report the current scene summary',
            'can report what i observed',
            'can report what i saw',
            'can report it',
            'going to report',
            "i'll report",
            "i'll let you know",
        )
    ):
        return ''
    if _is_placeholder_report_summary(clean):
        return ''
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate' and (
        'next object' in lowered
        or 'another object' in lowered
        or 'i am now walking' in lowered
        or "i'm now walking" in lowered
        or 'ready to move' in lowered
    ):
        return ''
    if _uses_stale_intermediate_arrival(clean, report_context):
        return ''
    clean = _rewrite_person_delivery_surface_phrase(clean, report_context)
    if _misclassifies_delivery_recipient_as_completed(clean, report_context):
        return ''
    if _omits_required_completed_targets(clean, report_context):
        return ''

    sentences = _split_sentences(clean)
    if len(sentences) <= 1:
        return clean
    successful_steps = _successful_non_report_steps(report_context)
    if len(successful_steps) == 1 and any(_is_generic_completion_sentence(item) for item in sentences):
        return ''
    return clean


def _omits_required_completed_targets(candidate: str, report_context: dict) -> bool:
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate':
        return False
    outcome = report_context.get('plan_outcome_summary', {})
    if not isinstance(outcome, dict):
        return False
    completed_targets = _clean_unique_list(outcome.get('completed_targets', []))
    delivery_objects = _completed_delivery_objects(report_context)
    if _is_delivery_report_context(report_context, delivery_objects=delivery_objects):
        completed_targets = delivery_objects or completed_targets
    if len(completed_targets) < 2:
        return False
    lowered = str(candidate or '').lower()
    mentioned = 0
    for target in completed_targets:
        target_text = target.lower()
        short = target_text
        for prefix in ('codex_probe_', 'codex_lab_', 'codex_kitchen_', 'detected_', 'object_'):
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        short = short.replace('_', ' ')
        if target_text in lowered or short in lowered:
            mentioned += 1
    return mentioned < len(completed_targets)


def _misclassifies_delivery_recipient_as_completed(candidate: str, report_context: dict) -> bool:
    delivery_objects = _completed_delivery_objects(report_context)
    if not _is_delivery_report_context(report_context, delivery_objects=delivery_objects):
        return False
    recipient = _latest_delivery_recipient(report_context)
    if not recipient:
        return False
    lowered = str(candidate or '').strip().lower()
    if 'completed' not in lowered and 'finished' not in lowered:
        return False
    recipient_forms = {
        recipient.lower(),
        recipient.lower().replace('_', ' '),
        _friendly_recipient_label(recipient).lower(),
    }
    return any(form and form in lowered for form in recipient_forms)


def _uses_stale_intermediate_arrival(candidate: str, report_context: dict) -> bool:
    if str(report_context.get('report_role', '')).strip().lower() != 'intermediate':
        return False
    lowered = str(candidate or '').strip().lower()
    if 'arrived at' not in lowered and 'arrived to' not in lowered:
        return False
    navigation_steps = [
        step for step in _successful_non_report_steps(report_context)
        if str(step.get('name', '')).strip().lower() in {'navigate_to', 'walk_to'}
    ]
    if len(navigation_steps) <= 1:
        return False
    if 'first object' in lowered:
        return True
    latest_target = _latest_step_target(navigation_steps[-1], report_context)
    if not latest_target:
        return False
    for step in navigation_steps[:-1]:
        target = _latest_step_target(step, report_context)
        if target and target != latest_target and target.lower() in lowered:
            return True
    return False


def _rewrite_person_delivery_surface_phrase(candidate: str, report_context: dict) -> str:
    person_labels = _person_delivery_destination_labels(report_context)
    if not person_labels:
        return candidate
    rewritten = str(candidate or '').strip()
    for label in sorted(person_labels, key=len, reverse=True):
        escaped = re.escape(label)
        rewritten = re.sub(
            r'\b(?:placed|put)\s+(?P<object>[^.?!;]+?)\s+(?:on|onto)\s+(?P<label>%s)\b'
            % escaped,
            lambda match: (
                'delivered %s to %s'
                % (match.group('object').strip(), match.group('label').strip())
            ),
            rewritten,
            flags=re.IGNORECASE,
        )
    return rewritten


def _person_delivery_destination_labels(report_context: dict) -> set[str]:
    entity_labels = _grounded_person_labels_by_id(report_context)
    labels: set[str] = set()
    for step in _successful_non_report_steps(report_context):
        if str(step.get('name', '')).strip().lower() != 'place_object':
            continue
        destination_id = _step_destination_id(step)
        if not destination_id:
            continue
        for label in entity_labels.get(destination_id, ()):
            clean = str(label or '').strip()
            if clean:
                labels.add(clean)
    return labels


def _grounded_person_labels_by_id(report_context: dict) -> dict[str, set[str]]:
    grounded_context = report_context.get('grounded_context', {})
    if not isinstance(grounded_context, dict):
        return {}
    labels_by_id: dict[str, set[str]] = {}
    for entity in grounded_context.get('entities', []):
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id or not _grounded_entity_is_person(entity):
            continue
        labels = {entity_id}
        for key in ('label', 'name', 'display_name'):
            value = str(entity.get(key, '')).strip()
            if value:
                labels.add(value)
        for relation in entity.get('relations', []):
            if not isinstance(relation, dict):
                continue
            predicate = str(relation.get('predicate', '')).strip().lower()
            if predicate.endswith('name') or predicate in {'name', 'label'}:
                value = str(relation.get('object', '')).strip()
                if value:
                    labels.add(value)
        labels_by_id[entity_id] = labels
    return labels_by_id


def _grounded_entity_is_person(entity: dict) -> bool:
    kind = str(entity.get('kind', '')).strip().lower()
    entity_class = str(entity.get('class', '')).strip().lower()
    if kind in {'person', 'human'} or entity_class in {'person', 'human'}:
        return True
    for relation in entity.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', '')).strip().lower()
        value = str(relation.get('object', '')).strip().lower()
        if predicate.endswith('type') and value in {'person', 'human', 'foaf:person'}:
            return True
    return False


def _step_destination_id(step: dict) -> str:
    for source in (step.get('args', {}), step.get('result_payload', {})):
        if not isinstance(source, dict):
            continue
        destination = str(
            source.get('destination')
            or source.get('recipient')
            or source.get('recipient_id')
            or source.get('destination_id')
            or ''
        ).strip()
        if destination:
            return destination
    return ''


def _latest_step_target(step: dict, report_context: dict) -> str:
    payload = step.get('result_payload', {})
    if isinstance(payload, dict):
        target = str(
            payload.get('target')
            or payload.get('object')
            or payload.get('location')
            or payload.get('target_frame')
            or ''
        ).strip()
        if target:
            return target
    args = step.get('args', {})
    if isinstance(args, dict):
        target = str(
            args.get('target')
            or args.get('object')
            or args.get('location')
            or args.get('target_frame')
            or ''
        ).strip()
        if target:
            return target
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        return str(
            latest_payload.get('target')
            or latest_payload.get('object')
            or latest_payload.get('location')
            or latest_payload.get('target_frame')
            or ''
        ).strip()
    return ''


def _successful_non_report_steps(report_context: dict) -> list[dict]:
    return [
        step
        for step in report_context.get('steps', [])
        if isinstance(step, dict)
        and str(step.get('status', '')).strip().lower() == 'succeeded'
        and str(step.get('name', '')).strip().lower() != 'report_result'
    ]


def _completed_delivery_objects(report_context: dict) -> list[str]:
    targets: list[str] = []
    for step in _successful_non_report_steps(report_context):
        if str(step.get('name', '')).strip().lower() not in {
            'bring_object',
            'deliver_object',
            'pick_object',
            'place_object',
        }:
            continue
        payload = step.get('result_payload', {})
        if not isinstance(payload, dict):
            payload = {}
        args = step.get('args', {})
        if not isinstance(args, dict):
            args = {}
        target = str(
            payload.get('object_id')
            or payload.get('object')
            or payload.get('target_object')
            or payload.get('target')
            or args.get('object_id')
            or args.get('object')
            or args.get('target_object')
            or args.get('target')
            or ''
        ).strip()
        if target and target not in targets:
            targets.append(target)
    return targets


def _is_delivery_report_context(
    report_context: dict,
    *,
    delivery_objects: list[str] | None = None,
) -> bool:
    goal_text = str(report_context.get('goal_text', '')).strip().lower()
    if any(marker in goal_text for marker in ('bring', 'deliver', 'handoff')):
        return True
    if delivery_objects is None:
        delivery_objects = _completed_delivery_objects(report_context)
    if not delivery_objects:
        return False
    return bool(_latest_delivery_recipient(report_context))


def _ordered_navigation_report(successful_steps: list[dict], report_context: dict) -> str:
    goal_text = str(report_context.get('goal_text', '')).strip().lower()
    if not any(marker in goal_text for marker in ('each object', 'every object', 'all objects')):
        return ''
    targets = []
    for step in successful_steps:
        if str(step.get('name', '')).strip().lower() not in {'navigate_to', 'walk_to'}:
            continue
        target = _latest_step_target(step, report_context)
        if target and target not in targets:
            targets.append(target)
    if len(targets) < 2:
        return ''
    labels = [_friendly_target_label(target) for target in targets]
    return 'I walked to %s and reported each arrival.' % _natural_join(labels)


def _friendly_target_label(target: str) -> str:
    clean = str(target or '').strip()
    if not clean:
        return 'the target'
    lowered = clean.lower()
    for prefix in ('codex_probe_', 'codex_lab_', 'codex_kitchen_', 'detected_', 'object_'):
        if lowered.startswith(prefix):
            clean = clean[len(prefix):]
            break
    return 'the %s' % clean.replace('_', ' ')


def _friendly_recipient_label(target: str) -> str:
    clean = str(target or '').strip()
    if not clean:
        return 'the recipient'
    lowered = clean.lower()
    for prefix in ('codex_lab_', 'codex_kitchen_', 'codex_recipient_', 'detected_', 'object_'):
        if lowered.startswith(prefix):
            clean = clean[len(prefix):]
            lowered = clean.lower()
            break
    if 'person' in lowered and not any(char.isdigit() for char in lowered):
        return 'the person'
    label = clean.replace('_', ' ').strip()
    if label and ' ' not in label:
        return label.upper()
    return 'the %s' % label if label else 'the recipient'


def _clean_unique_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        clean = str(item or '').strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def _latest_delivery_recipient(report_context: dict) -> str:
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        recipient = str(
            latest_payload.get('recipient')
            or latest_payload.get('recipient_id')
            or latest_payload.get('destination')
            or latest_payload.get('destination_id')
            or ''
        ).strip()
        if recipient:
            return recipient
    for step in reversed(_successful_non_report_steps(report_context)):
        args = step.get('args', {})
        if not isinstance(args, dict):
            continue
        recipient = str(
            args.get('recipient')
            or args.get('recipient_id')
            or args.get('destination')
            or args.get('destination_id')
            or ''
        ).strip()
        if recipient:
            return recipient
    return ''


def _natural_join(items: list[str]) -> str:
    clean_items = [str(item or '').strip() for item in items if str(item or '').strip()]
    if not clean_items:
        return ''
    if len(clean_items) == 1:
        return clean_items[0]
    if len(clean_items) == 2:
        return '%s and %s' % (clean_items[0], clean_items[1])
    return '%s, and %s' % (', '.join(clean_items[:-1]), clean_items[-1])


def _is_placeholder_report_summary(text: str) -> bool:
    lowered = str(text or '').strip().lower()
    return any(
        marker in lowered
        for marker in (
            'can report the current scene summary',
            'can report current scene summary',
            'can report what i observed',
            'can report what i saw',
            'can report it',
            'will report back',
            'will let you know',
        )
    )


def _split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r'(?<=[.!?])\s+', str(text or '').strip())
        if item.strip()
    ]


def _is_generic_completion_sentence(sentence: str) -> bool:
    lowered = str(sentence or '').strip().lower()
    return any(
        marker in lowered
        for marker in (
            'completed the task',
            'completed this task',
            'completed the',
            'have completed',
            'i finished the task',
            'i have finished the task',
        )
    )


def _fallback_planner_dialogue_ack(dialogue_context: dict) -> str:
    """Return bounded fallback wording without exposing planner/model prose."""
    completion_context = dialogue_context.get('completion_context', {})
    if isinstance(completion_context, dict):
        completion_text = _fallback_planner_completion_ack(completion_context)
        if completion_text:
            return completion_text

    act = str(dialogue_context.get('act', '')).strip().lower()
    return {
        'progress_update': 'I am working on it now.',
        'ask_clarification': 'I need a bit more detail before I continue.',
        'ask_for_help': 'I need help to continue this task.',
        'explain_failure': 'I could not complete that task.',
        'notify_completion': 'I finished that task.',
        'notify_cancellation': 'Okay, I will stop working on that.',
    }.get(act, '')
