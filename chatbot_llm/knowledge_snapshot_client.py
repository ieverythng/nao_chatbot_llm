"""ROS client wrapper for read-only KnowledgeCore snapshots."""

from __future__ import annotations

import json
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

        all_rows: list[dict] = []
        groups = settings.query_groups or [list(settings.patterns)]
        for group in groups:
            response = self._query_group(
                group,
                settings,
                turn_id=turn_id,
                trace=trace,
            )
            if response is None:
                continue
            all_rows.extend(self._parse_response_rows(getattr(response, 'json', '')))

        snapshot = format_knowledge_snapshot(
            json.dumps(self._dedupe_rows(all_rows)),
            settings,
        )
        if snapshot:
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'loaded %d chars from %s' % (len(snapshot), self._service_name),
            )
        else:
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'query returned no snapshot rows from %s' % self._service_name,
            )
        return snapshot

    def _query_group(
        self,
        patterns: list[str],
        settings: KnowledgeSnapshotSettings,
        *,
        turn_id: str,
        trace=None,
    ):
        request = Query.Request()
        request.patterns = list(patterns)
        request.vars = list(settings.query_vars)
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
            return None

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
            return None

        if not getattr(response, 'success', False):
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'query returned failure: %s' % getattr(response, 'error_msg', ''),
                level='warn',
            )
            return None
        return response

    @staticmethod
    def _parse_response_rows(json_payload: str) -> list[dict]:
        payload = str(json_payload or '').strip()
        if not payload:
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [row for row in parsed if isinstance(row, dict)]

    @staticmethod
    def _dedupe_rows(rows: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            key = json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    @staticmethod
    def _trace(trace, turn_id: str, stage: str, message: str, level: str = 'info') -> None:
        if callable(trace):
            trace(turn_id, stage, message, level=level)
