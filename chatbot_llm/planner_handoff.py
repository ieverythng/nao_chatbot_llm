"""Planner request publishing and grounded-context assembly for ``chatbot_llm``."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from hri_actions_msgs.msg import Intent
from planner_common import parse_json_object
from planner_common import project_llm_grounded_context
from planner_common import SceneSummary
from std_msgs.msg import String

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.planner_request_adapter import build_planner_request_intent_from_payload
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import should_route_intents_through_planner

TraceFn = Callable[..., None]
GoalIdCallback = Callable[[Any, str], None]
_TRACKED_PEOPLE_TOPIC = '/humans/persons/tracked'

try:  # pragma: no cover - runtime dependency
    from hri_msgs.msg import IdsList
except ImportError:  # pragma: no cover - runtime dependency
    IdsList = None


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
        self._tracked_people_ids: tuple[str, ...] = ()
        self._tracked_people_ts = 0.0

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
        self._tracked_people_sub = None
        if IdsList is not None:
            self._tracked_people_sub = node.create_subscription(
                IdsList,
                _TRACKED_PEOPLE_TOPIC,
                self._on_tracked_people,
                10,
            )

    def destroy(self) -> None:
        if getattr(self, '_publisher', None) is not None:
            self._node.destroy_publisher(self._publisher)
            self._publisher = None  # type: ignore[assignment]
        for attr in ('_scene_summary_sub', '_tracked_people_sub'):
            sub = getattr(self, attr, None)
            if sub is not None:
                self._node.destroy_subscription(sub)
                setattr(self, attr, None)

    def grounded_context(
        self,
        knowledge_context: str,
        *,
        knowledge_rows: list[dict] | tuple[dict, ...] | None = None,
    ) -> dict:
        with self._lock:
            scene = dict(self._scene_summary_payload)
            tracked_people_ids = tuple(self._tracked_people_ids)
            tracked_people_ts = float(self._tracked_people_ts)
        if not _has_scene_entities(scene) and not knowledge_rows:
            scene = _hydrate_scene_summary_from_knowledge_context(
                scene,
                knowledge_context=knowledge_context,
            )
        if tracked_people_ids:
            scene = _merge_tracked_people(
                scene,
                tracked_people_ids=tracked_people_ids,
                tracked_people_ts=tracked_people_ts,
            )
        raw_context = {
            'knowledge_snapshot': _knowledge_snapshot_payload(
                knowledge_context=knowledge_context,
                scene_summary=scene,
            ),
            'scene_summary': scene,
            'state_t0': _state_t0_payload(scene),
        }
        return project_llm_grounded_context(
            raw_context,
            knowledge_rows=[dict(item) for item in (knowledge_rows or []) if isinstance(item, dict)],
            include_state_t0=self._config.grounded_context_include_state_t0,
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
        ):
            return False

        try:
            planner_payload = build_planner_request_payload(
                turn_id=turn_id,
                user_text=user_text,
                turn_result=result,
                knowledge_context=knowledge_context,
                grounded_context=grounded_context or self.grounded_context(knowledge_context),
                multi_step_heuristics=self._config.planner_multi_step_heuristics,
                active_goal_id=session.active_planner_goal_id,
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

        planner_goal_id = str(planner_payload.get('goal_id', '')).strip()
        if planner_goal_id and self._on_planner_goal_id is not None:
            self._on_planner_goal_id(session, planner_goal_id)

        self._trace(
            turn_id,
            'PLANNER_REQUEST',
            'published planner request on %s goal_id=%s kind=%s'
            % (
                self._config.planner_request_topic,
                planner_payload.get('goal_id', ''),
                planner_payload.get('request_kind', ''),
            ),
        )
        return True

    def _on_scene_summary(self, msg: String) -> None:
        payload = _scene_summary_payload(msg.data)
        with self._lock:
            self._scene_summary_payload = payload

    def _on_tracked_people(self, msg: IdsList) -> None:
        people_ids = tuple(
            str(person_id).strip()
            for person_id in msg.ids
            if str(person_id).strip()
        )
        with self._lock:
            self._tracked_people_ids = people_ids
            self._tracked_people_ts = time.time()


def _scene_summary_payload(raw_payload) -> dict:
    raw_data = parse_json_object(raw_payload)
    summary = SceneSummary.from_payload(raw_payload)
    captured_at_sec = _captured_at_sec_from_scene(raw_data)
    return {
        'schema_version': 'scene_summary_v2',
        'observer': summary.observer,
        'backend': summary.backend,
        'captured_at_sec': captured_at_sec,
        'objects': [
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
        ],
        'people': _scene_people_payload(raw_data.get('people', [])),
    }


def _has_scene_entities(scene_summary: dict) -> bool:
    if not isinstance(scene_summary, dict):
        return False
    for key in ('objects', 'people'):
        items = scene_summary.get(key, [])
        if isinstance(items, list) and any(isinstance(item, dict) for item in items):
            return True
    return False


def _hydrate_scene_summary_from_knowledge_context(
    scene_summary: dict,
    *,
    knowledge_context: str,
) -> dict:
    """Backfill scene-summary entities from the current KB snapshot text."""
    merged = dict(scene_summary or {})
    objects = merged.get('objects', [])
    people = merged.get('people', [])
    if isinstance(objects, list) and objects:
        return merged
    if isinstance(people, list) and people:
        return merged

    references = _kb_references_from_text(knowledge_context)
    if not references:
        return merged

    hydrated_objects: list[dict] = []
    hydrated_people: list[dict] = []
    for item in references:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get('id', '')).strip()
        if not entity_id:
            continue
        entity_type = str(item.get('type', '')).strip()
        normalized_name = str(item.get('normalized_name', entity_id)).strip() or entity_id
        if _is_person_type(entity_type):
            hydrated_people.append(
                {
                    'id': entity_id,
                    'label': normalized_name,
                    'type': entity_type or 'Person',
                    'source': 'knowledge_snapshot',
                    'last_seen_sec': 0.0,
                }
            )
            continue
        hydrated_objects.append(
            {
                'entity_id': entity_id,
                'label': normalized_name,
                'kb_class': entity_type,
                'source': 'knowledge_snapshot',
                'score': 0.0,
                'center_x': 0.0,
                'center_y': 0.0,
                'last_seen_sec': 0.0,
            }
        )

    if hydrated_objects:
        merged['objects'] = hydrated_objects
    if hydrated_people:
        merged['people'] = hydrated_people
    return merged


def _knowledge_snapshot_payload(*, knowledge_context: str, scene_summary: dict) -> dict:
    references = _kb_references_from_scene(scene_summary)
    if not references:
        references = _kb_references_from_text(knowledge_context)
    counts = _reference_counts(references)
    return {
        'schema_version': 'knowledge_snapshot_v2',
        'captured_at_sec': _coerce_float(scene_summary.get('captured_at_sec', 0.0)),
        'references': references,
        'counts': counts,
    }


def _state_t0_payload(scene_summary: dict) -> dict:
    observer = str(scene_summary.get('observer', '')).strip()
    backend = str(scene_summary.get('backend', '')).strip()

    object_entities = []
    scene_objects = scene_summary.get('objects', [])
    if isinstance(scene_objects, list):
        for item in scene_objects:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get('entity_id', item.get('id', ''))).strip()
            if not object_id:
                continue
            object_entities.append(
                {
                    'normalized_name': _normalized_entity_name(
                        str(item.get('label', object_id)).strip()
                    ),
                    'id': object_id,
                    'type': str(item.get('kb_class', '')).strip(),
                    'kind': 'object',
                    'source': str(item.get('source', '')).strip(),
                    'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
                }
            )

    person_entities = []
    scene_people = scene_summary.get('people', [])
    if isinstance(scene_people, list):
        for item in scene_people:
            if not isinstance(item, dict):
                continue
            person_id = str(item.get('id', item.get('entity_id', ''))).strip()
            if not person_id:
                continue
            person_type = (
                str(item.get('type', item.get('kb_class', 'Person'))).strip()
                or 'Person'
            )
            person_entities.append(
                {
                    'normalized_name': _normalized_entity_name(
                        str(item.get('label', person_id)).strip()
                    ),
                    'id': person_id,
                    'type': person_type,
                    'kind': 'person',
                    'source': str(item.get('source', 'hri_tracked_persons')).strip()
                    or 'hri_tracked_persons',
                    'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
                }
            )

    entities = object_entities + person_entities

    captured_at_sec = 0.0
    if entities:
        captured_at_sec = max(
            (_coerce_float(item.get('last_seen_sec', 0.0)) for item in entities),
            default=0.0,
        )

    return {
        'schema_version': 'state_t0_v2',
        'observer': observer,
        'backend': backend,
        'captured_at_sec': captured_at_sec,
        'entity_counts': _entity_counts(entities),
        'entities': entities,
        'objects': [dict(item) for item in object_entities],
        'people': [dict(item) for item in person_entities],
    }


def _kb_references_from_scene(scene_summary: dict) -> list[dict]:
    references: list[dict] = []
    objects = scene_summary.get('objects', [])
    if not isinstance(objects, list):
        return references
    for item in objects:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get('entity_id', item.get('id', ''))).strip()
        if not entity_id:
            continue
        label = str(item.get('label', entity_id)).strip()
        references.append(
            {
                'normalized_name': _normalized_entity_name(label),
                'id': entity_id,
                'type': str(item.get('kb_class', '')).strip(),
            }
        )
    people = scene_summary.get('people', [])
    if isinstance(people, list):
        for item in people:
            if not isinstance(item, dict):
                continue
            person_id = str(item.get('id', item.get('entity_id', ''))).strip()
            if not person_id:
                continue
            label = str(item.get('label', person_id)).strip() or person_id
            person_type = (
                str(item.get('type', item.get('kb_class', 'Person'))).strip()
                or 'Person'
            )
            references.append(
                {
                    'normalized_name': _normalized_entity_name(label),
                    'id': person_id,
                    'type': person_type,
                }
            )
    return references


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


def _is_person_type(value: str) -> bool:
    clean = str(value or '').strip().lower()
    if not clean:
        return False
    return any(token in clean for token in ('person', 'human', 'face', 'speaker'))


def _captured_at_sec_from_scene(raw_data: dict) -> float:
    explicit_ts = _coerce_float(raw_data.get('captured_at_sec', 0.0))
    if explicit_ts > 0.0:
        return explicit_ts
    timestamps: list[float] = []
    for key in ('objects', 'people'):
        items = raw_data.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                timestamps.append(_coerce_float(item.get('last_seen_sec', 0.0)))
    return max(timestamps, default=0.0)


def _reference_counts(references: list[dict]) -> dict:
    people = 0
    objects = 0
    for item in references:
        item_type = str(item.get('type', '')).strip().lower()
        if item_type in {'person', 'human'} or 'person' in item_type or 'human' in item_type:
            people += 1
        else:
            objects += 1
    return {
        'entities': len(references),
        'people': people,
        'objects': objects,
    }


def _entity_counts(entities: list[dict]) -> dict:
    people = sum(1 for item in entities if str(item.get('kind', '')).strip() == 'person')
    objects = sum(1 for item in entities if str(item.get('kind', '')).strip() == 'object')
    return {
        'entities': len(entities),
        'people': people,
        'objects': objects,
    }


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scene_people_payload(raw_people) -> list[dict]:
    if not isinstance(raw_people, list):
        return []
    people: list[dict] = []
    for item in raw_people:
        if not isinstance(item, dict):
            continue
        person_id = str(item.get('id', item.get('entity_id', ''))).strip()
        if not person_id:
            continue
        people.append(
            {
                'id': person_id,
                'label': str(item.get('label', person_id)).strip() or person_id,
                'type': (
                    str(item.get('type', item.get('kb_class', 'Person'))).strip()
                    or 'Person'
                ),
                'source': str(item.get('source', 'scene_summary')).strip() or 'scene_summary',
                'score': _coerce_float(item.get('score', 0.0)),
                'center_x': _coerce_float(item.get('center_x', 0.0)),
                'center_y': _coerce_float(item.get('center_y', 0.0)),
                'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
            }
        )
    return people


def _merge_tracked_people(
    scene_summary: dict,
    *,
    tracked_people_ids: tuple[str, ...],
    tracked_people_ts: float,
) -> dict:
    merged = dict(scene_summary or {})
    current_people = merged.get('people', [])
    people = (
        [dict(item) for item in current_people if isinstance(item, dict)]
        if isinstance(current_people, list)
        else []
    )
    existing_ids = {
        str(item.get('id', item.get('entity_id', ''))).strip()
        for item in people
        if isinstance(item, dict)
    }
    now_sec = time.time()
    age_sec = (
        max(0.0, now_sec - tracked_people_ts)
        if tracked_people_ts > 0.0
        else 0.0
    )
    last_seen_sec = max(0.0, now_sec - age_sec)
    for person_id in tracked_people_ids:
        clean_person_id = str(person_id).strip()
        if not clean_person_id or clean_person_id in existing_ids:
            continue
        people.append(
            {
                'id': clean_person_id,
                'label': clean_person_id,
                'type': 'Person',
                'source': 'hri_tracked_persons',
                'last_seen_sec': last_seen_sec,
                'last_seen_age_sec': round(age_sec, 3),
            }
        )
        existing_ids.add(clean_person_id)
    if people:
        merged['people'] = people
    return merged


def _planner_confidence(result) -> float:
    try:
        confidence = float(getattr(result, 'intent_confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    route = str(getattr(result, 'route', '')).strip().lower()
    if confidence <= 0.0 and route == 'execution':
        return 0.5
    return max(0.0, min(1.0, confidence))
