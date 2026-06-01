"""Planner request publishing and grounded-context assembly for ``chatbot_llm``."""

from __future__ import annotations

import threading
from typing import Any, Callable

from hri_actions_msgs.msg import Intent
from planner_common import SceneSummary
from std_msgs.msg import String

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.planner_request_adapter import build_planner_request_intent_from_payload
from chatbot_llm.planner_request_adapter import build_planner_request_payload
from chatbot_llm.planner_request_adapter import should_route_intents_through_planner

TraceFn = Callable[..., None]
GoalIdCallback = Callable[[Any, str], None]


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

    def destroy(self) -> None:
        if getattr(self, '_publisher', None) is not None:
            self._node.destroy_publisher(self._publisher)
            self._publisher = None  # type: ignore[assignment]
        for attr in ('_scene_summary_sub',):
            sub = getattr(self, attr, None)
            if sub is not None:
                self._node.destroy_subscription(sub)
                setattr(self, attr, None)

    def grounded_context(self, knowledge_context: str) -> dict:
        with self._lock:
            scene = dict(self._scene_summary_payload)
        return {
            'knowledge_snapshot': _knowledge_snapshot_payload(
                knowledge_context=knowledge_context,
                scene_summary=scene,
            ),
            'scene_summary': scene,
            'state_t0': _state_t0_payload(scene),
        }

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

def _scene_summary_payload(raw_payload) -> dict:
    summary = SceneSummary.from_payload(raw_payload)
    return {
        'observer': summary.observer,
        'backend': summary.backend,
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
    }


def _knowledge_snapshot_payload(*, knowledge_context: str, scene_summary: dict) -> dict:
    references = _kb_references_from_scene(scene_summary)
    if not references:
        references = _kb_references_from_text(knowledge_context)
    return {'references': references} if references else {}


def _state_t0_payload(scene_summary: dict) -> dict:
    observer = str(scene_summary.get('observer', '')).strip()
    backend = str(scene_summary.get('backend', '')).strip()

    entities = []
    scene_objects = scene_summary.get('objects', [])
    if isinstance(scene_objects, list):
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
                    'source': str(item.get('source', '')).strip(),
                    'last_seen_sec': _coerce_float(item.get('last_seen_sec', 0.0)),
                }
            )

    captured_at_sec = 0.0
    if entities:
        captured_at_sec = max(
            (_coerce_float(item.get('last_seen_sec', 0.0)) for item in entities),
            default=0.0,
        )

    return {
        'observer': observer,
        'backend': backend,
        'captured_at_sec': captured_at_sec,
        'entities': entities,
        'objects': [dict(item) for item in entities],
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
