"""ROS client wrapper for read-only KnowledgeCore snapshots."""

from __future__ import annotations

import threading

from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot

try:  # pragma: no cover - optional dependency in unit tests
    from kb_msgs.srv import Query
except ImportError:  # pragma: no cover - optional dependency in unit tests
    Query = None


class KnowledgeSnapshotClient:
    """Query `/kb/query` and format the result into prompt-ready text."""

    def __init__(self, node, callback_group, service_name: str, timeout_sec: float) -> None:
        self._node = node
        self._service_name = str(service_name or '/kb/query').strip() or '/kb/query'
        self._timeout_sec = max(0.05, float(timeout_sec))
        self._client = None
        self._warned_import = False
        self._warned_unavailable = False

        if Query is None:
            self._warned_import = True
            self._node.get_logger().warn(
                'kb_msgs is unavailable; knowledge snapshots are disabled'
            )
            return

        self._client = self._node.create_client(
            Query,
            self._service_name,
            callback_group=callback_group,
        )

    def fetch_snapshot(
        self,
        settings: KnowledgeSnapshotSettings,
        *,
        turn_id: str = '',
        trace=None,
    ) -> str:
        """Return one formatted snapshot for the current turn or an empty string."""
        if not settings.enabled or self._client is None:
            return ''

        if not self._client.service_is_ready():
            if not self._warned_unavailable:
                self._node.get_logger().warn(
                    'Knowledge snapshot service is unavailable at %s'
                    % self._service_name
                )
                self._warned_unavailable = True
            return ''

        request = Query.Request()
        request.patterns = list(settings.patterns)
        request.vars = list(settings.vars)
        request.models = list(settings.models)

        future = self._client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())

        if not completed.wait(timeout=self._timeout_sec):
            future.cancel()
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'timeout waiting for %s' % self._service_name,
                level='warn',
            )
            return ''

        try:
            response = future.result()
        except Exception as err:  # pragma: no cover - rclpy failure path
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'query failure: %s' % err,
                level='warn',
            )
            return ''

        if not getattr(response, 'success', False):
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'query returned failure: %s' % getattr(response, 'error_msg', ''),
                level='warn',
            )
            return ''

        snapshot = format_knowledge_snapshot(
            getattr(response, 'json', ''),
            settings,
        )
        if snapshot:
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'loaded %d chars from %s' % (len(snapshot), self._service_name),
            )
        return snapshot

    @staticmethod
    def _trace(trace, turn_id: str, stage: str, message: str, level: str = 'info') -> None:
        if callable(trace):
            trace(turn_id, stage, message, level=level)
