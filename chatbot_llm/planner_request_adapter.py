"""Helpers for optional planner-mode handoff from ``chatbot_llm``."""

from __future__ import annotations

from dataclasses import asdict
import json
import re

from kb_skills.intent_labels import KB_QUERY_INTENTS
from chatbot_llm.prompt_pack import default_prompt_pack
from chatbot_llm.person_references import resolve_grounded_person_id
from chatbot_llm.person_references import resolve_grounded_person_in_text
from planner_common import PLANNER_REQUEST_KINDS
from planner_common import PlannerRequest
from planner_common import extract_json_object
from planner_common import IntentLabels
from planner_common import is_perform_motion_object_label
from planner_common import make_goal_id
from planner_common import normalize_grounded_context
from planner_common import project_llm_grounded_context

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
    pending_execution_context: dict | None = None,
    grounded_context: dict | None = None,
) -> bool:
    """Return true when the turn contains execution-oriented intents."""
    user_intent = dict(_turn_user_intent(turn_result))
    resolved_intent = getattr(turn_result, 'intent', '')
    if _pending_execution_for_turn(
        pending_execution_context,
        user_text=user_text,
        grounded_context=grounded_context or {},
    ):
        return True
    if _is_dialogue_only_request(
        user_text=user_text,
        user_intent=user_intent,
        resolved_intent=resolved_intent,
    ):
        return False

    if _normalize_token(getattr(turn_result, 'route', '')) == 'execution':
        return True

    if _contains_execution_intent(intents):
        return True

    return _is_multi_step_turn(
        user_intent=user_intent,
        resolved_intent=resolved_intent,
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
    pending_execution_context: dict | None = None,
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
    dialogue_context = _bounded_dialogue_context(
        getattr(turn_result, 'updated_history', []),
        max_history_entries=max_history_entries,
    )
    normalized_intents = _normalized_intents_for_turn(turn_result)
    grounded_payload = _grounded_context_payload(
        knowledge_context,
        grounded_context=grounded_context,
    )
    return_destination_id = _robot_return_destination_id(user_text, grounded_payload)
    if return_destination_id:
        normalized_intents = _robot_return_intents(normalized_intents)
        report_policy = str(
            user_intent.get('target_selection', {}).get('report_policy', 'none')
            if isinstance(user_intent.get('target_selection'), dict)
            else 'none'
        ).strip().lower()
        user_intent['intent_sequence'] = normalized_intents
        if str(user_intent.get('type', '')).strip().lower() in {
            'bring_object',
            'deliver_object',
        }:
            user_intent['type'] = 'navigate_to'
        user_intent['target_selection'] = {
            'selection_kind': 'explicit_members',
            'operation': 'visit',
            'source_location_id': '',
            'member_ids': [return_destination_id],
            'recipient_id': '',
            'ordering': 'sequential',
            'report_policy': report_policy if report_policy in {'none', 'final'} else 'none',
        }
    request_kind = _resolved_request_kind(user_intent, resolved_intent)
    pending_clarification = _pending_planner_clarification(
        dialogue_context,
        active_goal_id=active_goal_id,
    )
    if (
        request_kind == 'new_goal'
        and 'request_kind' not in user_intent
        and not normalized_intents
        and pending_clarification
    ):
        request_kind = 'clarification_answer'
        user_intent = _clarification_answer_intent(
            user_intent,
            user_text=user_text,
            grounded_context=grounded_payload,
        )
        normalized_intents = list(pending_clarification.get('requested_intents', []))
    pending_execution = _pending_execution_for_turn(
        pending_execution_context,
        user_text=user_text,
        grounded_context=grounded_payload,
    )
    if pending_execution:
        user_intent = _rehydrate_execution_context(
            user_intent,
            pending_execution,
            user_text=user_text,
            grounded_context=grounded_payload,
        )
        normalized_intents = list(
            pending_execution.get('normalized_intents', [])
        )
    goal_id = _resolved_goal_id(
        user_intent=user_intent,
        turn_id=turn_id,
        active_goal_id=active_goal_id,
        request_kind=request_kind,
    )
    scene_targets = _scene_targets_from_user_intent(user_intent)
    quantified_targets = _quantified_grounded_object_ids(
        _goal_text_from_user_intent(user_intent, user_text=user_text),
        grounded_payload.get('entities', []),
    )
    if quantified_targets:
        scene_targets = _merge_quantified_scene_targets(
            quantified_targets,
            scene_targets,
            grounded_payload.get('entities', []),
        )
    payload = {
        'request_id': str(turn_id).strip(),
        'goal_id': goal_id,
        'parent_goal_id': str(user_intent.get('parent_goal_id', '')).strip(),
        'supersedes_goal_id': _resolved_supersedes_goal_id(
            user_intent=user_intent,
            request_kind=request_kind,
            active_goal_id=active_goal_id,
            goal_id=goal_id,
        ),
        'request_kind': request_kind,
        'goal_text': _resolved_goal_text(
            user_intent,
            user_text=user_text,
            pending_clarification=pending_clarification
            if request_kind == 'clarification_answer'
            else {},
        ),
        'normalized_intents': normalized_intents,
        'scene_targets': scene_targets,
        'dialogue_context': dialogue_context,
        'grounded_context': grounded_payload,
        'planner_mode': resolved_planner_mode,
        'dialogue_turn_id': str(user_intent.get('dialogue_turn_id', turn_id)).strip()
        or str(turn_id).strip(),
    }
    if isinstance(user_intent.get('target_selection'), dict):
        payload['target_selection'] = user_intent['target_selection']
    elif _target_selection_salvage_allowed(
        turn_result,
        request_kind=request_kind,
        has_pending_execution=bool(pending_execution),
    ):
        target_selection = _derive_target_selection(payload, user_intent)
        if target_selection:
            payload['target_selection'] = target_selection
    return _planner_request_payload(payload)


def _target_selection_salvage_allowed(
    turn_result,
    *,
    request_kind: str,
    has_pending_execution: bool,
) -> bool:
    """Keep deterministic scope derivation behind explicit fallback states."""
    if has_pending_execution or request_kind == 'clarification_answer':
        return True
    source = str(getattr(turn_result, 'intent_source', '') or '').strip().lower()
    return (
        'retry_exhausted' in source
        or 'route_repair' in source
        or source.startswith('rules')
        or '+rules_' in source
    )


def _pending_planner_clarification(
    dialogue_context: list[str],
    *,
    active_goal_id: str,
) -> dict:
    for entry in reversed(dialogue_context):
        role, separator, content = str(entry or '').partition(':')
        if separator != ':' or role.strip().lower() != 'system':
            continue
        payload = extract_json_object(content)
        planner_dialogue = payload.get('planner_dialogue', {}) if payload else {}
        if not isinstance(planner_dialogue, dict):
            continue
        if str(planner_dialogue.get('act', '')).strip() != 'ask_clarification':
            return {}
        if not bool(planner_dialogue.get('await_user_response', False)):
            return {}
        goal_id = str(planner_dialogue.get('goal_id', '')).strip()
        if active_goal_id and goal_id and goal_id != str(active_goal_id).strip():
            return {}
        context = planner_dialogue.get('context', {})
        return dict(context) if isinstance(context, dict) else {}
    return {}


def _clarification_answer_intent(
    user_intent: dict,
    *,
    user_text: str,
    grounded_context: dict,
) -> dict:
    normalized = dict(user_intent or {})
    person_id = resolve_grounded_person_id(grounded_context, user_text)
    if not person_id:
        return normalized
    normalized['recipient'] = person_id
    scene_targets = _scene_targets_from_user_intent(normalized)
    if person_id not in scene_targets:
        scene_targets.append(person_id)
    normalized['scene_targets'] = scene_targets
    return normalized


def _pending_execution_for_turn(
    pending_execution_context: dict | None,
    *,
    user_text: str,
    grounded_context: dict,
) -> dict:
    """Return a pending route-conflict task only for an entity correction."""
    if not isinstance(pending_execution_context, dict):
        return {}
    if not _looks_like_entity_correction(user_text):
        return {}
    person_id = resolve_grounded_person_in_text(grounded_context, user_text)
    if not person_id:
        return {}
    normalized_intents = [
        _normalize_token(value)
        for value in pending_execution_context.get('normalized_intents', [])
        if _normalize_token(value)
    ]
    if not normalized_intents:
        intent_name = _normalize_token(pending_execution_context.get('intent', ''))
        if intent_name:
            normalized_intents = [intent_name]
    if not normalized_intents:
        return {}
    return {
        'goal_text': str(pending_execution_context.get('goal_text', '')).strip(),
        'normalized_intents': list(dict.fromkeys(normalized_intents)),
        'scene_targets': pending_execution_context.get('scene_targets', []),
        'requested_person': str(
            pending_execution_context.get('requested_person', '')
        ).strip(),
        'recipient_id': person_id,
    }


def _looks_like_entity_correction(user_text: str) -> bool:
    normalized = _normalize_grounded_words(user_text)
    return bool(
        re.match(
            r'^(?:i\s+(?:meant|mean)|use|the\s+(?:person|human|recipient)|'
            r'(?:person|human|recipient))\b',
            normalized,
        )
    )


def _rehydrate_execution_context(
    user_intent: dict,
    pending_execution: dict,
    *,
    user_text: str,
    grounded_context: dict,
) -> dict:
    normalized = dict(user_intent or {})
    original_goal = normalize_goal_text(pending_execution.get('goal_text', ''))
    grounded_goal = _replace_pending_recipient(
        original_goal,
        requested_person=pending_execution.get('requested_person', ''),
        recipient_id=pending_execution['recipient_id'],
    )
    if grounded_goal:
        normalized['goal_text'] = grounded_goal
    else:
        current_answer = normalize_goal_text(user_text)
        if original_goal and current_answer:
            normalized['goal_text'] = '%s Clarification answer: %s' % (
                original_goal,
                current_answer,
            )
        elif original_goal:
            normalized['goal_text'] = original_goal
        else:
            normalized['goal_text'] = current_answer
    normalized['type'] = pending_execution['normalized_intents'][0]
    normalized['recipient'] = pending_execution['recipient_id']
    target_ids = _pending_scene_target_ids(
        original_goal,
        pending_execution.get('scene_targets', []),
        grounded_context.get('entities', []),
    )
    if pending_execution['recipient_id'] not in target_ids:
        target_ids.append(pending_execution['recipient_id'])
    normalized['scene_targets'] = target_ids
    return normalized


def _replace_pending_recipient(
    goal_text: str,
    *,
    requested_person: str,
    recipient_id: str,
) -> str:
    goal = str(goal_text or '').strip()
    requested = str(requested_person or '').strip()
    recipient = str(recipient_id or '').strip()
    if not goal or not requested or not recipient:
        return ''
    replaced = re.sub(
        r'(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])' % re.escape(requested),
        recipient,
        goal,
        flags=re.IGNORECASE,
    )
    return normalize_goal_text(replaced) if replaced != goal else ''


def _pending_scene_target_ids(goal_text: str, raw_targets, entities) -> list[str]:
    target_ids = _quantified_grounded_object_ids(goal_text, entities)
    if target_ids:
        return target_ids
    for target in raw_targets if isinstance(raw_targets, (list, tuple)) else ():
        record = _match_grounded_record(target, entities)
        if record:
            target_id = str(record.get('id', '')).strip()
            if target_id and target_id not in target_ids:
                target_ids.append(target_id)
            continue
        folded_target = _fold_grounded_name(target)
        if not folded_target or folded_target in {'object', 'item', 'thing'}:
            continue
        for entity in entities if isinstance(entities, list) else ():
            if not isinstance(entity, dict):
                continue
            if str(entity.get('kind', '')).strip().lower() != 'object':
                continue
            names = _grounded_reference_names(entity)
            if not any(
                folded_target == _fold_grounded_name(name)
                for name in names
            ):
                continue
            entity_id = str(entity.get('id', '')).strip()
            if entity_id and entity_id not in target_ids:
                target_ids.append(entity_id)
    return target_ids


def _merge_quantified_scene_targets(
    quantified_targets: list[str],
    raw_targets: list[str],
    entities,
) -> list[str]:
    merged = list(dict.fromkeys(quantified_targets))
    quantified_ids = set(quantified_targets)
    quantified_labels = set()
    for entity in entities if isinstance(entities, list) else ():
        if not isinstance(entity, dict):
            continue
        if str(entity.get('id', '')).strip() not in quantified_ids:
            continue
        quantified_labels.update(
            _fold_grounded_name(name) for name in _grounded_reference_names(entity)
        )
    for target in raw_targets:
        target_text = str(target or '').strip()
        if not target_text or target_text in merged:
            continue
        record = _match_grounded_record(target_text, entities)
        if record:
            if str(record.get('kind', '')).strip().lower() != 'object':
                merged.append(target_text)
            continue
        if _fold_grounded_name(target_text) not in quantified_labels:
            merged.append(target_text)
    return merged


def _resolved_goal_text(
    user_intent: dict,
    *,
    user_text: str,
    pending_clarification: dict,
) -> str:
    current = _goal_text_from_user_intent(user_intent, user_text=user_text)
    original = str(pending_clarification.get('goal_text', '')).strip()
    if not original:
        return current
    if not current:
        return original
    return '%s Clarification answer: %s' % (original, current)


def _derive_target_selection(payload: dict, user_intent: dict) -> dict:
    grounded = payload.get('grounded_context', {})
    locations = grounded.get('locations', []) if isinstance(grounded, dict) else []
    entities = grounded.get('entities', []) if isinstance(grounded, dict) else []
    source_hint = str(user_intent.get('object', '')).strip()
    recipient_hint = str(user_intent.get('recipient', '')).strip()
    goal_text = str(payload.get('goal_text', '')).strip()
    source = _match_grounded_record(source_hint, locations)
    if not source:
        source = _match_referenced_grounded_record(goal_text, locations)
    if not source:
        source = _single_grounded_record(
            _match_grounded_record(target, locations)
            for target in payload.get('scene_targets', [])
        )
    recipient = _match_grounded_record(recipient_hint, entities, required_kind='person')
    if not recipient:
        recipient = _match_referenced_grounded_record(
            goal_text,
            entities,
            required_kind='person',
        )
    scene_records = [
        _match_grounded_record(target, entities)
        for target in payload.get('scene_targets', [])
    ]
    if not recipient:
        recipient = _single_record_of_kind(scene_records, 'person')
    if not recipient and _references_generic_person(goal_text):
        recipient = _single_visible_record_of_kind(entities, 'person')
    if not recipient:
        recipient = _delivery_destination_from_goal(goal_text, entities)
    intents = set(payload.get('normalized_intents', []))
    report_policy = _selection_report_policy(user_intent, intents, goal_text=goal_text)

    if 'bring_object' in intents:
        if not recipient:
            return {}
        recipient_id = str(recipient.get('id', '')).strip()
        if source and str(source.get('id', '')).strip() == recipient_id:
            source = {}
        if source:
            members = _object_member_ids(source.get('contains', []))
            selection_kind = 'location_members'
            source_location_id = str(source.get('id', '')).strip()
        else:
            selected_object = _match_grounded_record(
                source_hint,
                entities,
                required_kind='object',
            )
            if not selected_object:
                selected_object = _match_referenced_grounded_record(
                    goal_text,
                    entities,
                    required_kind='object',
                )
            object_candidates = [
                record for record in scene_records
                if isinstance(record, dict)
                and str(record.get('id', '')).strip() != recipient_id
                and str(record.get('kind', '')).strip().lower() == 'object'
            ]
            members = _object_member_ids(object_candidates)
            if selected_object:
                members = _object_member_ids([selected_object])
            if not members and _requests_quantified_object_scope(goal_text):
                members = _object_member_ids(_visible_grounded_objects(entities))
            selection_kind = 'explicit_members'
            source_location_id = ''
        if not members:
            return {}
        return {
            'selection_kind': selection_kind,
            'operation': 'deliver',
            'source_location_id': source_location_id,
            'member_ids': members,
            'recipient_id': recipient_id,
            'ordering': 'none',
            'report_policy': report_policy,
        }

    if intents.intersection({'navigate_to', 'walk_to'}):
        object_scope = _requests_quantified_object_scope(goal_text)
        person_scope = _references_generic_person(goal_text) or bool(
            _match_referenced_grounded_record(
                goal_text,
                entities,
                required_kind='person',
            )
        )
        required_kind = 'object' if object_scope else ''
        selected_records = [
            _match_grounded_record(target, entities, required_kind=required_kind)
            for target in payload.get('scene_targets', [])
        ]
        if not object_scope:
            selected_records = [
                record or _match_grounded_record(target, locations)
                for target, record in zip(payload.get('scene_targets', []), selected_records)
            ]
        if person_scope:
            person_id = resolve_grounded_person_in_text(grounded, goal_text)
            person_record = _match_grounded_record(
                person_id,
                entities,
                required_kind='person',
            )
            if person_record and person_record not in selected_records:
                selected_records.append(person_record)
        members = (
            _object_member_ids(selected_records)
            if object_scope
            else _grounded_member_ids(selected_records)
        )
        if not members and object_scope:
            members = _object_member_ids(
                source.get('contains', []) if source else _visible_grounded_objects(entities)
            )
        if not members:
            return {}
        return {
            'selection_kind': 'explicit_members',
            'operation': 'visit',
            'source_location_id': '',
            'member_ids': members,
            'recipient_id': '',
            'ordering': 'sequential',
            'report_policy': report_policy,
        }
    return {}


def _selection_report_policy(
    user_intent: dict,
    intents: set[str],
    *,
    goal_text: str,
) -> str:
    explicit = str(user_intent.get('report_policy', '')).strip().lower()
    if explicit in {'none', 'per_target', 'final'}:
        return explicit
    if requests_per_target_reporting(goal_text):
        return 'per_target'
    if 'report_result' in intents or _requests_final_reporting(goal_text):
        return 'final'
    return 'none'


def _requests_quantified_object_scope(goal_text: str) -> bool:
    normalized = _normalize_grounded_words(goal_text)
    return bool(
        re.search(
            r'\b(?:all|each|every|both) (?:the )?(?:(?:visible|selected) )?objects?\b',
            normalized,
        )
        or re.search(r'\b(?:these|those) (?:(?:visible|selected) )?objects?\b', normalized)
        or bool(re.search(r'\bboth\s+[a-z0-9_-]+\b', normalized))
    )


def _quantified_grounded_object_ids(goal_text: str, entities) -> list[str]:
    normalized = _normalize_grounded_words(goal_text)
    match = re.search(
        r'\b(?:all|each|every|both)\s+(?:the\s+)?([a-z0-9_-]+)\b',
        normalized,
    )
    if not match:
        return []
    requested_class = _singular_grounded_word(match.group(1))
    if requested_class in {'object', 'item', 'thing'}:
        return []
    matches = []
    for entity in entities if isinstance(entities, list) else ():
        if not isinstance(entity, dict):
            continue
        if str(entity.get('kind', '')).strip().lower() != 'object':
            continue
        names = _grounded_reference_names(entity)
        words = {
            _singular_grounded_word(word)
            for name in names
            for word in _normalize_grounded_words(name).split()
        }
        entity_id = str(entity.get('id', '')).strip()
        if entity_id and requested_class in words:
            matches.append(entity_id)
    return list(dict.fromkeys(matches))


def _singular_grounded_word(value: str) -> str:
    word = str(value or '').strip().lower()
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 3 and word[-3] in {'s', 'x', 'z'}:
        return word[:-2]
    if word.endswith('s') and len(word) > 2:
        return word[:-1]
    return word


def _delivery_destination_from_goal(goal_text: str, entities) -> dict:
    normalized = _normalize_grounded_words(goal_text)
    match = re.search(r'\bto\s+(?:the\s+)?(.+)$', normalized)
    if not match:
        return {}
    destination = _match_referenced_grounded_record(match.group(1), entities)
    if not destination:
        return {}
    if str(destination.get('kind', '')).strip().lower() == 'person':
        return destination
    predicates = {
        str(relation.get('predicate', '')).strip().lower().split(':')[-1]
        for relation in destination.get('relations', [])
        if isinstance(relation, dict)
    }
    entity_class = _fold_grounded_name(destination.get('class', ''))
    if predicates.intersection({'contains', 'placeof'}):
        return destination
    if entity_class in {'location', 'place', 'room', 'spatialthinglocalized'}:
        return destination
    return {}


def _robot_return_destination_id(user_text: str, grounded_context: dict) -> str:
    """Resolve a robot destination from "return to X", not "return X to Y"."""
    normalized = _normalize_grounded_words(user_text)
    match = re.search(
        r'\b(?:return|go back|come back) to (?:the )?(.+?)(?: and |$)',
        normalized,
    )
    if not match:
        return ''
    return resolve_grounded_person_in_text(grounded_context, match.group(1))


def _robot_return_intents(intents: list[str]) -> list[str]:
    filtered = [
        intent
        for intent in intents
        if intent not in {'bring_object', 'deliver_object'}
    ]
    if 'navigate_to' not in filtered:
        report_index = filtered.index('report_result') if 'report_result' in filtered else len(filtered)
        filtered.insert(report_index, 'navigate_to')
    return list(dict.fromkeys(filtered))


def requests_per_target_reporting(goal_text: str) -> bool:
    normalized = _normalize_grounded_words(goal_text)
    if not re.search(r'\b(?:report|tell me|let me know)\b', normalized):
        return False
    return bool(
        re.search(
            r'\b(?:each|every) (?:arrival|one|target|time)\b',
            normalized,
        )
    )


def _requests_final_reporting(goal_text: str) -> bool:
    normalized = _normalize_grounded_words(goal_text)
    return bool(
        re.search(r'\b(?:report|summary|summarize|summarise)\b', normalized)
        or re.search(r'\b(?:tell me what happened|let me know)\b', normalized)
    )


def _visible_grounded_objects(entities) -> list[dict]:
    if not isinstance(entities, list):
        return []
    return [
        entity
        for entity in entities if isinstance(entity, dict)
        if str(entity.get('kind', '')).strip().lower() == 'object'
        and entity.get('visible', True) is not False
    ]


def _single_visible_record_of_kind(records, kind: str) -> dict:
    if not isinstance(records, list):
        return {}
    return _single_record_of_kind(
        [record for record in records if record.get('visible', True) is not False],
        kind,
    )


def _references_generic_person(text: str) -> bool:
    normalized = _normalize_grounded_words(text)
    return bool(re.search(r'\b(?:person|human|recipient)\b', normalized))


def _single_record_of_kind(records, kind: str) -> dict:
    matches = [
        record
        for record in records if isinstance(record, dict)
        if str(record.get('kind', '')).strip().lower() == kind
    ]
    return matches[0] if len(matches) == 1 else {}


def _single_grounded_record(records) -> dict:
    matches = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict) or not record:
            continue
        record_id = str(record.get('id', '')).strip()
        if record_id and record_id in seen_ids:
            continue
        if record_id:
            seen_ids.add(record_id)
        matches.append(record)
    return matches[0] if len(matches) == 1 else {}


def _object_member_ids(records) -> list[str]:
    return _entity_member_ids(records, allowed_kinds={'object'})


def _grounded_member_ids(records) -> list[str]:
    return list(
        dict.fromkeys(
            str(record.get('id', '')).strip()
            for record in records if isinstance(records, (list, tuple))
            if isinstance(record, dict) and str(record.get('id', '')).strip()
        )
    )


def _entity_member_ids(records, *, allowed_kinds: set[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(record.get('id', '')).strip()
            for record in records if isinstance(records, (list, tuple))
            if isinstance(record, dict)
            and str(record.get('kind', '')).strip().lower() in allowed_kinds
            and str(record.get('id', '')).strip()
        )
    )


def _match_grounded_record(hint: str, records, *, required_kind: str = '') -> dict:
    folded_hint = _fold_grounded_name(hint)
    if not folded_hint:
        return {}
    matches = []
    for record in records if isinstance(records, list) else ():
        if not isinstance(record, dict):
            continue
        if required_kind and str(record.get('kind', '')).strip().lower() != required_kind:
            continue
        names = [record.get('id', ''), record.get('label', '')]
        names.extend(record.get('aliases', []) if isinstance(record.get('aliases'), list) else ())
        if folded_hint in {_fold_grounded_name(name) for name in names}:
            matches.append(record)
    return matches[0] if len(matches) == 1 else {}


def _match_referenced_grounded_record(
    text: str,
    records,
    *,
    required_kind: str = '',
) -> dict:
    normalized_text = ' %s ' % _normalize_grounded_words(text)
    matches: list[tuple[tuple[int, int], dict]] = []
    for record in records if isinstance(records, list) else ():
        if not isinstance(record, dict):
            continue
        if required_kind and str(record.get('kind', '')).strip().lower() != required_kind:
            continue
        names = _grounded_reference_names(record)
        scores = []
        for name in names:
            normalized_name = _normalize_grounded_words(name)
            if normalized_name and ' %s ' % normalized_name in normalized_text:
                scores.append((len(normalized_name.split()), len(normalized_name)))
        if scores:
            matches.append((max(scores), record))
    if not matches:
        return {}
    best_score = max(score for score, _record in matches)
    best_matches = [record for score, record in matches if score == best_score]
    return best_matches[0] if len(best_matches) == 1 else {}


def _grounded_reference_names(record: dict) -> list[str]:
    names = [record.get('id', ''), record.get('label', ''), record.get('class', '')]
    names.extend(record.get('aliases', []) if isinstance(record.get('aliases'), list) else ())
    entity_class = str(record.get('class', '')).strip()
    for relation in record.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', '')).strip().lower()
        value = str(relation.get('object', '')).strip()
        if not value or predicate not in {'dbp:name', 'dbp:color', 'dbp:colour'}:
            continue
        names.append(value)
        if predicate in {'dbp:color', 'dbp:colour'} and entity_class:
            names.append(f'{value} {entity_class}')
    return names


def _normalize_grounded_words(value) -> str:
    return ' '.join(
        ''.join(
            character if character.isalnum() else ' '
            for character in str(value or '').lower()
        ).split()
    )


def _fold_grounded_name(value) -> str:
    return ''.join(character for character in str(value or '').lower() if character.isalnum())


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
    pending_execution_context: dict | None = None,
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
        pending_execution_context=pending_execution_context,
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
    if not normalized.get('target_selection'):
        normalized.pop('target_selection', None)
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


def _normalized_intents_for_turn(turn_result) -> list[str]:
    user_intent = _turn_user_intent(turn_result)
    intent_sequence = _coerce_str_list(user_intent.get('intent_sequence'))
    candidates = intent_sequence or [
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
    raw_targets = user_intent.get('scene_targets')
    if isinstance(raw_targets, (list, tuple)):
        scene_targets = []
        for item in raw_targets:
            if isinstance(item, dict):
                target_id = str(
                    item.get('id') or item.get('entity_id') or item.get('target') or ''
                ).strip()
            else:
                target_id = str(item).strip()
            if target_id and target_id not in scene_targets:
                scene_targets.append(target_id)
    else:
        scene_targets = _coerce_str_list(raw_targets)
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


def _goal_text_from_user_intent(user_intent: dict, *, user_text: str) -> str:
    for key in ('goal_text', 'goal', 'task'):
        clean_value = str(user_intent.get(key, '')).strip()
        if clean_value:
            return normalize_goal_text(clean_value)
    return normalize_goal_text(str(user_text or '').strip())


# Leading conversational filler stripped from goal statements. The chatbot prompt is the
# primary owner of a concise goal_text; this is a deterministic safety net so a verbatim
# utterance never leaks to the planner when the model omits or echoes the user phrasing.
_GOAL_TEXT_LEADING_FILLER = (
    'can you please',
    'could you please',
    'would you please',
    'please can you',
    'please could you',
    'can you',
    'could you',
    'would you',
    'will you',
    'please',
    'hey pop',
    'hey',
    'okay',
    'ok',
    'so',
    'now',
    'pop',
)


def normalize_goal_text(value: str) -> str:
    """Strip leading politeness/filler and trailing noise from a goal statement.

    Case and wording are preserved otherwise; this never tries to summarize meaning (that
    remains the chatbot's job), it only removes the obvious verbatim cruft that confuses
    the downstream planner.
    """
    text = str(value or '').strip().strip('"\'').strip()
    if not text:
        return ''

    changed = True
    while changed:
        changed = False
        if text and text[0] in ',.':
            text = text.lstrip(',. ').strip()
            changed = True
            continue
        lowered = text.lower()
        for filler in _GOAL_TEXT_LEADING_FILLER:
            if not lowered.startswith(filler):
                continue
            boundary = lowered[len(filler):len(filler) + 1]
            if boundary in ('', ' ', ',', '.'):
                text = text[len(filler):].lstrip(',. ').strip()
                changed = True
                break

    text = re.sub(r'\s+', ' ', text).strip()
    stripped_trailing = text.rstrip('!').strip()
    return stripped_trailing or text


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
    if 'entities' in payload:
        return payload

    clean_knowledge_context = str(knowledge_context or '').strip()
    if not clean_knowledge_context:
        return project_llm_grounded_context(payload)

    knowledge_snapshot = dict(payload.get('knowledge_snapshot', {}))
    references = knowledge_snapshot.get('references', [])
    has_structured_refs = isinstance(references, list) and bool(references)
    if not has_structured_refs:
        derived_references = _knowledge_references_from_text(clean_knowledge_context)
        if derived_references:
            knowledge_snapshot['references'] = derived_references
            payload['knowledge_snapshot'] = knowledge_snapshot
    return project_llm_grounded_context(payload)


def dialogue_turn_id(role_name: str, dialogue_id: tuple[int, ...] | None, request_count: int) -> str:
    """Build a planner-visible turn id scoped to one dialogue session."""
    return '%s:%s:%d' % (
        str(role_name or '__default__').strip() or '__default__',
        _short_uuid(dialogue_id),
        max(1, int(request_count or 1)),
    )


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
        if normalized_turn_id and not _is_local_dialogue_turn_id(normalized_turn_id):
            return 'goal_%s' % normalized_turn_id
    return make_goal_id()


def _is_local_dialogue_turn_id(value: str) -> bool:
    """Return true for ids that are not globally useful outside one dialogue."""
    normalized = str(value or '').strip().lower()
    return normalized in {'default', '__default__'}


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


def _short_uuid(dialogue_id: tuple[int, ...] | None) -> str:
    if not dialogue_id:
        return 'unknown'
    return ''.join('%02x' % value for value in dialogue_id[:4])


def _normalize_token(value) -> str:
    return str(value or '').strip().lower()


def _contains_execution_intent(intents: list[Intent]) -> bool:
    if not isinstance(intents, list) or not intents:
        return False
    return any(
        _normalize_token(getattr(intent, 'intent', '')) not in _NON_PLANNER_INTENT_NAMES
        for intent in intents
    )


def _is_dialogue_only_request(
    *,
    user_text: str,
    user_intent: dict,
    resolved_intent: str,
) -> bool:
    if not _is_dialogue_only_capability_question(user_text):
        return False
    clean_intent = _normalize_token(user_intent.get('type', '') or resolved_intent)
    return clean_intent in _NON_PLANNER_INTENT_NAMES or clean_intent in ('', 'fallback')


def _is_dialogue_only_capability_question(text: str) -> bool:
    normalized = ''.join(
        char
        for char in ' '.join(str(text or '').strip().lower().split())
        if char.isalnum() or char.isspace()
    ).strip()
    return any(
        marker in normalized
        for marker in (
            'what can you do',
            'what are you able to do',
            'what capabilities do you have',
            'what are your capabilities',
            'tell me what you can do',
            'what skills do you have',
            'which skills do you have',
            'what fake skills do you have',
            'do you have fake skills',
            'do you have any fake skills',
            'tell me about your skills',
        )
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


def _knowledge_references_from_text(knowledge_context: str) -> list[dict]:
    references: list[dict] = []
    seen_ids: set[str] = set()
    for raw_line in str(knowledge_context or '').splitlines():
        clean_line = str(raw_line).strip().lstrip('-').strip()
        if ' is currently classified as ' in clean_line:
            name, _, type_text = clean_line.partition(' is currently classified as ')
        elif ' is a ' in clean_line:
            name, _, type_text = clean_line.partition(' is a ')
        else:
            continue
        clean_name = str(name).strip()
        clean_type = str(type_text).split(',')[0].strip()
        if not clean_name:
            continue
        normalized_name = '_'.join(clean_name.lower().split())
        if normalized_name in seen_ids:
            continue
        seen_ids.add(normalized_name)
        references.append(
            {
                'normalized_name': normalized_name,
                'id': normalized_name,
                'type': clean_type,
            }
        )
    return references


def _coerce_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
