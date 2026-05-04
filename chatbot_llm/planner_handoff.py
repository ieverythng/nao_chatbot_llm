"""Planner request publishing and cached world-model inputs for ``chatbot_llm``."""

from __future__ import annotations

import threading
from typing import Any, Callable

from hri_actions_msgs.msg import Intent
from planner_common import EnrichedSnapshot
from planner_common import SceneSummary
from planner_common import build_world_model_text
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
        self._world_model_snapshot: dict = {}
        self._world_model_text = ''

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
        self._world_snapshot_sub = node.create_subscription(
            String,
            config.planner_world_model_snapshot_topic,
            self._on_world_model_snapshot,
            10,
        )
        self._world_text_sub = node.create_subscription(
            String,
            config.planner_world_model_text_topic,
            self._on_world_model_text,
            10,
        )

    def destroy(self) -> None:
        if getattr(self, '_publisher', None) is not None:
            self._node.destroy_publisher(self._publisher)
            self._publisher = None  # type: ignore[assignment]
        for attr in ('_scene_summary_sub', '_world_snapshot_sub', '_world_text_sub'):
            sub = getattr(self, attr, None)
            if sub is not None:
                self._node.destroy_subscription(sub)
                setattr(self, attr, None)

    def grounded_context(self, knowledge_context: str) -> dict:
        knowledge_snapshot = {}
        clean_knowledge_context = str(knowledge_context or '').strip()
        if clean_knowledge_context:
            knowledge_snapshot['summary_text'] = clean_knowledge_context
        with self._lock:
            scene = dict(self._scene_summary_payload)
            snapshot = dict(self._world_model_snapshot)
            text = self._world_model_text
        if snapshot and not text:
            text = build_world_model_text(EnrichedSnapshot.from_payload(snapshot))
        return {
            'knowledge_snapshot': knowledge_snapshot,
            'scene_summary': scene,
            'world_model_snapshot': snapshot,
            'world_model_text': text,
        }

    def publish_execution_turn_if_needed(
        self,
        *,
        session: Any,
        user_id: str,
        turn_id: str,
        user_text: str,
        knowledge_context: str,
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
                grounded_context=self.grounded_context(knowledge_context),
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

    def _on_world_model_snapshot(self, msg: String) -> None:
        payload = _world_model_snapshot_payload(msg.data)
        with self._lock:
            self._world_model_snapshot = payload

    def _on_world_model_text(self, msg: String) -> None:
        with self._lock:
            self._world_model_text = str(msg.data or '').strip()


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


def _world_model_snapshot_payload(raw_payload) -> dict:
    snapshot = EnrichedSnapshot.from_payload(raw_payload)
    return {
        'observer': snapshot.observer,
        'backend': snapshot.backend,
        'active_plan_id': snapshot.active_plan_id,
        'execution_status': snapshot.execution_status,
        'execution_reason': snapshot.execution_reason,
        'scene_targets': list(snapshot.scene_targets),
        'entities': [
            {
                'entity_id': entity.entity_id,
                'label': entity.label,
                'kb_class': entity.kb_class,
                'state': entity.state,
                'score': entity.score,
                'source': entity.source,
                'last_seen_sec': entity.last_seen_sec,
                'age_sec': entity.age_sec,
                'is_plan_relevant': entity.is_plan_relevant,
                'risk_tags': list(entity.risk_tags),
            }
            for entity in snapshot.entities
        ],
        'kb_rows': list(snapshot.kb_rows),
        'timestamp_sec': snapshot.timestamp_sec,
    }


def _planner_confidence(result) -> float:
    try:
        confidence = float(getattr(result, 'intent_confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    route = str(getattr(result, 'route', '')).strip().lower()
    if confidence <= 0.0 and route == 'execution':
        return 0.5
    return max(0.0, min(1.0, confidence))
