"""Planner request publishing and grounded-context assembly for ``chatbot_llm``."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

from planner_common import parse_json_object
from planner_common import project_llm_grounded_context
from planner_common import SceneSummary

try:  # pragma: no cover - ROS runtime dependency
    from hri_actions_msgs.msg import Intent
except ImportError:  # pragma: no cover - import-light unit tests
    class Intent:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.intent = ''
            self.source = ''
            self.modality = ''
            self.confidence = 0.0
            self.priority = 0
            self.data = ''


try:  # pragma: no cover - ROS runtime dependency
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - import-light unit tests
    class String:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.data = ''


try:  # pragma: no cover - ROS runtime dependency
    from hri_msgs.msg import IdsList
except ImportError:  # pragma: no cover - import-light unit tests
    IdsList = None

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.planner_request_adapter import build_planner_request_intent_from_payload
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import should_route_intents_through_planner

TraceFn = Callable[..., None]
GoalIdCallback = Callable[[Any, str], None]
_PERSON_LIKE_TOKENS = ('person', 'people', 'human', 'face', 'speaker', 'visitor', 'user')
_SCENE_PERSON_FRESHNESS_SEC = 5.0


def _planner_handoff_observability(result, planner_payload: dict) -> dict:
    """Project route and intent lineage for non-blocking turn diagnostics."""
    raw_intents = planner_payload.get('normalized_intents', [])
    if not isinstance(raw_intents, (list, tuple)):
        raw_intents = []
    normalized_intents = [
        str(intent).strip() for intent in raw_intents if str(intent).strip()
    ]
    route = str(getattr(result, 'route', '') or '').strip()
    intent = str(getattr(result, 'intent', '') or '').strip()
    intent_source = str(getattr(result, 'intent_source', '') or '').strip()
    return {
        'route': route,
        'intent': intent,
        'intent_source': intent_source,
        'route_repaired': 'route_repair' in intent_source.lower(),
        'normalized_intents': normalized_intents,
        'intent_gap': route == 'execution' and not normalized_intents,
        'goal_id': str(planner_payload.get('goal_id', '') or '').strip(),
        'dialogue_turn_id': str(
            planner_payload.get('dialogue_turn_id', '') or ''
        ).strip(),
    }


def _grounded_context_or_refresh(
    grounded_context: dict | None,
    refresh,
) -> dict:
    """Use a turn projection only when it contains grounded entities."""
    if isinstance(grounded_context, dict) and grounded_context.get('entities'):
        return grounded_context
    return refresh()


class PlannerHandoff:
    """Owns planner ingress publisher, subscriptions, and grounded-context assembly."""

    def __init__(
        self,
        node,
        config: ChatbotConfig,
        *,
        trace: TraceFn,
        on_planner_goal_id: GoalIdCallback | None = None,
    ) -> None:
        self._node = node
        self._config = config
        self._trace = trace
        self._on_planner_goal_id = on_planner_goal_id
        self._lock = threading.Lock()
        self._scene_summary_payload: dict = {}
        self._tracked_person_ids: set[str] | None = None

        self._publisher = node.create_publisher(
            Intent,
            config.planner_request_topic,
            10,
        )
        self._scene_summary_sub = node.create_subscription(
            String,
            config.planner_scene_summary_topic,
            self._on_scene_summary,
            10,
        )
        self._tracked_persons_sub = None
        if IdsList is not None:
            self._tracked_persons_sub = node.create_subscription(
                IdsList,
                '/humans/persons/tracked',
                self._on_tracked_persons,
                10,
            )

    def destroy(self) -> None:
        if getattr(self, '_publisher', None) is not None:
            self._node.destroy_publisher(self._publisher)
            self._publisher = None  # type: ignore[assignment]
        for attr in ('_scene_summary_sub', '_tracked_persons_sub'):
            sub = getattr(self, attr, None)
            if sub is not None:
                self._node.destroy_subscription(sub)
                setattr(self, attr, None)

    def grounded_context(
        self,
        knowledge_context: str,
        *,
        knowledge_rows: tuple[dict, ...] | list[dict] | None = None,
    ) -> dict:
        with self._lock:
            scene = dict(self._scene_summary_payload)
            active_person_ids = (
                None
                if self._tracked_person_ids is None
                else set(self._tracked_person_ids)
            )
        source_context = {
            'knowledge_snapshot': _knowledge_snapshot_payload(
                knowledge_context=knowledge_context,
                scene_summary=scene,
            ),
            'scene_summary': scene,
            'state_t0': _state_t0_payload(scene),
        }
        return project_llm_grounded_context(
            source_context,
            knowledge_rows=list(knowledge_rows or ()),
            active_person_ids=active_person_ids,
        )

    def publish_execution_turn_if_needed(
        self,
        *,
        session: Any,
        user_id: str,
        turn_id: str,
        user_text: str,
        knowledge_context: str,
        grounded_context: dict | None = None,
        result,
        direct_intents: list[Intent],
        pending_execution_context: dict | None = None,
    ) -> bool:
        """Publish a planner request when routing rules match; return True if published."""
        if self._publisher is None:
            self._node.get_logger().warn(
                'planner mode is enabled but planner request publisher is unavailable'
            )
            return False
        if not should_route_intents_through_planner(
            direct_intents,
            turn_result=result,
            user_text=user_text,
            multi_step_heuristics=self._config.planner_multi_step_heuristics,
            pending_execution_context=pending_execution_context,
            grounded_context=grounded_context,
        ):
            return False

        try:
            planner_payload = build_planner_request_payload(
                turn_id=turn_id,
                user_text=user_text,
                turn_result=result,
                knowledge_context=knowledge_context,
                grounded_context=_grounded_context_or_refresh(
                    grounded_context,
                    lambda: self.grounded_context(knowledge_context),
                ),
                multi_step_heuristics=self._config.planner_multi_step_heuristics,
                active_goal_id=session.active_planner_goal_id,
                pending_execution_context=pending_execution_context,
            )
            planner_msg = build_planner_request_intent_from_payload(
                payload=planner_payload,
                source_user_id=user_id,
                planner_request_intent=self._config.planner_request_intent,
                confidence=_planner_confidence(result),
            )
            self._publisher.publish(planner_msg)
        except Exception as err:  # pragma: no cover - ROS publish failures are runtime-only
            self._node.get_logger().warn('failed to publish planner request: %s' % err)
            self._trace(turn_id, 'PLANNER_REQUEST', 'publish failed: %s' % err, level='warn')
            return False

        if pending_execution_context:
            session.pending_execution_context = {}

        planner_goal_id = str(planner_payload.get('goal_id', '')).strip()
        if planner_goal_id and self._on_planner_goal_id is not None:
            self._on_planner_goal_id(session, planner_goal_id)

        handoff_evidence = _planner_handoff_observability(result, planner_payload)
        normalized_intents = json.dumps(
            handoff_evidence['normalized_intents'],
            separators=(',', ':'),
        )
        handoff_message = (
            'route=%s intent=%s intent_source=%s route_repaired=%s '
            'normalized_intents=%s goal_id=%s dialogue_turn_id=%s'
            % (
                handoff_evidence['route'] or '-',
                handoff_evidence['intent'] or '-',
                handoff_evidence['intent_source'] or '-',
                str(handoff_evidence['route_repaired']).lower(),
                normalized_intents,
                handoff_evidence['goal_id'] or '-',
                handoff_evidence['dialogue_turn_id'] or '-',
            )
        )
        self._trace(turn_id, 'ROUTE_INTENT_HANDOFF', handoff_message)
        self._trace(
            turn_id,
            'PLANNER_REQUEST',
            'published planner request on %s goal_id=%s kind=%s %s'
            % (
                self._config.planner_request_topic,
                handoff_evidence['goal_id'],
                planner_payload.get('request_kind', ''),
                handoff_message,
            ),
        )
        if handoff_evidence['intent_gap']:
            self._trace(
                turn_id,
                'ROUTE_INTENT_GAP',
                'execution admitted without normalized_intents goal_id=%s '
                'intent_source=%s route_repaired=%s dialogue_turn_id=%s'
                % (
                    handoff_evidence['goal_id'] or '-',
                    handoff_evidence['intent_source'] or '-',
                    str(handoff_evidence['route_repaired']).lower(),
                    handoff_evidence['dialogue_turn_id'] or '-',
                ),
                level='warn',
            )
        return True

    def _on_scene_summary(self, msg: String) -> None:
        payload = _scene_summary_payload(msg.data)
        with self._lock:
            self._scene_summary_payload = payload

    def _on_tracked_persons(self, msg: IdsList) -> None:
        tracked_ids = {
            str(person_id).strip()
            for person_id in getattr(msg, 'ids', [])
            if str(person_id).strip()
        }
        with self._lock:
            self._tracked_person_ids = tracked_ids
        self._trace(
            '',
            'TRACKED_PERSON_STATE',
            'authoritative_count=%d ids=%s'
            % (len(tracked_ids), ','.join(sorted(tracked_ids)) or 'none'),
            level='debug',
        )


def _scene_summary_payload(raw_payload) -> dict:
    raw_data = parse_json_object(raw_payload)
    summary = SceneSummary.from_payload(raw_payload)
    raw_objects = [
        {
            'label': obj.label,
            'entity_id': obj.entity_id,
            'kb_class': obj.kb_class,
            'score': obj.score,
            'tracker_id': obj.tracker_id,
            'source': obj.source,
            'center_x': obj.center_x,
            'center_y': obj.center_y,
            'last_seen_sec': obj.last_seen_sec,
        }
        for obj in summary.objects
    ]
    scene_objects = _scene_objects_payload(raw_objects)
    people = _merged_people_payload(
        _normalized_people_payload(raw_data.get('people', [])),
        _people_from_scene_objects(raw_objects),
    )
    people = _fresh_scene_people(people)
    payload = {
        'schema_version': 'scene_summary_v2',
        'observer': summary.observer,
        'backend': summary.backend,
        'captured_at_sec': _captured_at_sec(objects=scene_objects, people=people),
        'objects': scene_objects,
    }
    if people:
        payload['people'] = people
    return payload


def _knowledge_snapshot_payload(*, knowledge_context: str, scene_summary: dict) -> dict:
    references = _kb_references_from_scene(scene_summary)
    if not references:
        references = _kb_references_from_text(knowledge_context)

    payload = {
        'schema_version': 'knowledge_snapshot_v2',
        'references': references,
    }
    captured_at_sec = _coerce_float(scene_summary.get('captured_at_sec', 0.0))
    if captured_at_sec > 0.0:
        payload['captured_at_sec'] = captured_at_sec
    return payload


def _state_t0_payload(scene_summary: dict) -> dict:
    observer = str(scene_summary.get('observer', '')).strip()
    backend = str(scene_summary.get('backend', '')).strip()
    scene_objects = scene_summary.get('objects', [])
    scene_people = scene_summary.get('people', [])
    normalized_people = _normalized_people_payload(scene_people)

    object_entities = _state_entities_from_scene_objects(scene_objects)
    people_entities = _state_people_entries(normalized_people)
    entities = _merged_state_entities([*object_entities, *people_entities])

    return {
        'schema_version': 'state_t0_v2',
        'observer': observer,
        'backend': backend,
        'captured_at_sec': _captured_at_sec(objects=scene_objects, people=normalized_people),
        'entities': entities,
    }


def _kb_references_from_scene(scene_summary: dict) -> list[dict]:
    references: list[dict] = []
    seen_ids: set[str] = set()
    objects = scene_summary.get('objects', [])
    if isinstance(objects, list):
        for item in objects:
            _append_reference(references, seen_ids, item, fallback_type_key='kb_class')

    people = scene_summary.get('people', [])
    if isinstance(people, list):
        for item in people:
            _append_reference(
                references,
                seen_ids,
                item,
                fallback_type_key='type',
                default_type='Human',
            )
    return references


def _state_entities_from_scene_objects(scene_objects) -> list[dict]:
    if not isinstance(scene_objects, list):
        return []
    entities: list[dict] = []
    for item in scene_objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get('entity_id', item.get('id', ''))).strip()
        if not object_id:
            continue
        entities.append(
            {
                'normalized_name': _normalized_entity_name(
                    str(item.get('label', object_id)).strip()
                ),
                'id': object_id,
                'type': str(item.get('kb_class', '')).strip(),
                'kind': _entity_kind(
                    label=str(item.get('label', '')).strip(),
                    entity_type=str(item.get('kb_class', '')).strip(),
                ),
                'source': str(item.get('source', '')).strip(),
                'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
            }
        )
    return entities


def _scene_objects_payload(raw_objects: list[dict]) -> list[dict]:
    objects: list[dict] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        if _is_person_like_label(
            str(item.get('label', '')).strip(),
            str(item.get('kb_class', '')).strip(),
        ):
            continue
        objects.append(dict(item))
    return objects


def _people_from_scene_objects(raw_objects: list[dict]) -> list[dict]:
    people: list[dict] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get('entity_id', item.get('id', ''))).strip()
        if not entity_id:
            continue
        label = str(item.get('label', entity_id)).strip()
        entity_type = str(item.get('kb_class', item.get('type', 'Human'))).strip()
        if not _is_person_like_label(label, entity_type):
            continue
        people.append(
            {
                'id': entity_id,
                'label': label or entity_id,
                'type': entity_type or 'Human',
                'source': str(item.get('source', 'scene_summary')).strip() or 'scene_summary',
                'score': _coerce_float(item.get('score', 0.0)),
                'center_x': _coerce_float(item.get('center_x', 0.0)),
                'center_y': _coerce_float(item.get('center_y', 0.0)),
                'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
            }
        )
    return people


def _merged_people_payload(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for candidate in [*primary, *secondary]:
        if not isinstance(candidate, dict):
            continue
        person_id = str(candidate.get('id', candidate.get('entity_id', ''))).strip()
        if not person_id or person_id in seen_ids:
            continue
        seen_ids.add(person_id)
        merged.append(dict(candidate))
    return merged


def _fresh_scene_people(people: list[dict]) -> list[dict]:
    latest_seen = max(
        (_coerce_float(person.get('last_seen_sec', 0.0)) for person in people),
        default=0.0,
    )
    if latest_seen <= 0.0:
        return people
    return [
        person
        for person in people
        if _coerce_float(person.get('last_seen_sec', 0.0)) <= 0.0
        or latest_seen - _coerce_float(person.get('last_seen_sec', 0.0))
        <= _SCENE_PERSON_FRESHNESS_SEC
    ]


def _state_people_entries(normalized_people) -> list[dict]:
    people: list[dict] = []
    for item in normalized_people:
        person_id = str(item.get('id', item.get('entity_id', ''))).strip()
        if not person_id:
            continue
        people.append(
            {
                'normalized_name': _normalized_entity_name(
                    str(item.get('label', person_id)).strip()
                ),
                'id': person_id,
                'type': str(item.get('type', item.get('kb_class', 'Human'))).strip() or 'Human',
                'kind': 'person',
                'source': str(item.get('source', '')).strip(),
                'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
            }
        )
    return people


def _merged_state_entities(items: list[dict]) -> list[dict]:
    merged: list[dict] = []
    merged_by_id: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get('id', '')).strip()
        if not entity_id:
            continue
        existing_index = merged_by_id.get(entity_id)
        if existing_index is None:
            merged_by_id[entity_id] = len(merged)
            merged.append(dict(item))
            continue
        existing = merged[existing_index]
        existing_kind = str(existing.get('kind', '')).strip().lower()
        new_kind = str(item.get('kind', '')).strip().lower()
        if existing_kind != 'person' and new_kind == 'person':
            merged[existing_index] = dict(item)
    return merged


def _append_reference(
    references: list[dict],
    seen_ids: set[str],
    item,
    *,
    fallback_type_key: str,
    default_type: str = '',
) -> None:
    if not isinstance(item, dict):
        return
    entity_id = str(item.get('entity_id', item.get('id', ''))).strip()
    if not entity_id or entity_id in seen_ids:
        return
    label = str(item.get('label', entity_id)).strip()
    entity_type = str(item.get('type', item.get(fallback_type_key, default_type))).strip()
    references.append(
        {
            'normalized_name': _normalized_entity_name(label),
            'id': entity_id,
            'type': entity_type,
        }
    )
    seen_ids.add(entity_id)


def _normalized_people_payload(raw_people) -> list[dict]:
    if not isinstance(raw_people, list):
        return []
    normalized: list[dict] = []
    for item in raw_people:
        if isinstance(item, dict):
            person_id = str(item.get('id', item.get('entity_id', ''))).strip()
            if not person_id:
                continue
            normalized.append(
                {
                    'id': person_id,
                    'label': str(item.get('label', person_id)).strip() or person_id,
                    'type': str(item.get('type', item.get('kb_class', 'Human'))).strip()
                    or 'Human',
                    'source': str(item.get('source', 'scene_summary')).strip() or 'scene_summary',
                    'score': _coerce_float(item.get('score', 0.0)),
                    'center_x': _coerce_float(item.get('center_x', 0.0)),
                    'center_y': _coerce_float(item.get('center_y', 0.0)),
                    'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
                }
            )
            continue
        person_id = str(item).strip()
        if person_id:
            normalized.append(
                {
                    'id': person_id,
                    'label': person_id,
                    'type': 'Human',
                    'source': 'scene_summary',
                    'score': 0.0,
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'last_seen_sec': 0.0,
                }
            )
    return normalized


def _captured_at_sec(*, objects, people) -> float:
    timestamps: list[float] = []
    if isinstance(objects, list):
        timestamps.extend(
            _coerce_float(item.get('last_seen_sec', 0.0))
            for item in objects
            if isinstance(item, dict)
        )
    if isinstance(people, list):
        timestamps.extend(
            _coerce_float(item.get('last_seen_sec', 0.0))
            for item in people
            if isinstance(item, dict)
        )
    return max(timestamps, default=0.0)


def _entity_kind(*, label: str, entity_type: str) -> str:
    if _is_person_like_label(label, entity_type):
        return 'person'
    return 'object'


def _is_person_like_label(label: str, kb_class: str = '') -> bool:
    label_token = _normalized_entity_name(label)
    class_token = _normalized_entity_name(kb_class)
    return any(
        token and keyword in token
        for token in (label_token, class_token)
        for keyword in _PERSON_LIKE_TOKENS
    )


def _kb_references_from_text(knowledge_context: str) -> list[dict]:
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
        normalized_name = _normalized_entity_name(clean_name)
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


def _normalized_entity_name(value: str) -> str:
    return '_'.join(str(value or '').strip().lower().split())


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _planner_confidence(result) -> float:
    try:
        confidence = float(getattr(result, 'intent_confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    route = str(getattr(result, 'route', '')).strip().lower()
    if confidence <= 0.0 and route == 'execution':
        return 0.5
    return max(0.0, min(1.0, confidence))
