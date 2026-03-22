"""Formatting adapter over the reusable `kb_skills` query client."""

from __future__ import annotations

import json

from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot
from kb_skills.query_client import KnowledgeCoreQueryClient


class KnowledgeSnapshotClient:
    """Fetch KnowledgeCore rows and format them into prompt-ready snapshots."""

    def __init__(self, node, callback_group, service_name: str, timeout_sec: float) -> None:
        self._query_client = KnowledgeCoreQueryClient(
            node=node,
            callback_group=callback_group,
            service_name=service_name,
            timeout_sec=timeout_sec,
        )

    def fetch_snapshot(
        self,
        settings: KnowledgeSnapshotSettings,
        *,
        turn_id: str = '',
        trace=None,
    ) -> str:
        """Return one formatted snapshot for the current turn or an empty string."""
        if not settings.enabled:
            return ''

        all_rows: list[dict] = []
        groups = settings.query_groups or [list(settings.patterns)]
        for group in groups:
            all_rows.extend(
                self._query_client.query_rows(
                    patterns=group,
                    query_vars=list(settings.query_vars),
                    models=list(settings.models),
                    turn_id=turn_id,
                    trace=trace,
                    trace_stage='KB_SNAPSHOT',
                )
            )

        snapshot = format_knowledge_snapshot(
            json.dumps(KnowledgeCoreQueryClient.dedupe_rows(all_rows)),
            settings,
        )
        if snapshot:
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'loaded %d chars from %s'
                % (len(snapshot), self._query_client.service_name),
            )
        else:
            self._trace(
                trace,
                turn_id,
                'KB_SNAPSHOT',
                'query returned no snapshot rows from %s'
                % self._query_client.service_name,
            )
        return snapshot

    @staticmethod
    def _trace(trace, turn_id: str, stage: str, message: str, level: str = 'info') -> None:
        if callable(trace):
            trace(turn_id, stage, message, level=level)
