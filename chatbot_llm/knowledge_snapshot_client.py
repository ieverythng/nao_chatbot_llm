"""Formatting adapter over the reusable `kb_skills` query client."""

from __future__ import annotations

import json
import re

from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot
from kb_skills.query_client import KnowledgeCoreQueryClient

_VARIABLE_TOKEN_RE = re.compile(r'\?[A-Za-z_][A-Za-z0-9_]*')


def _query_vars_for_group(group_patterns: list[str], configured_vars: list[str]) -> list[str]:
    """Keep query vars bounded to variables present in one pattern group.

    Passing unbound vars to `/kb/query` can trigger large combinatorial scans in
    KnowledgeCore. Each grouped query should request only the vars it actually binds.
    """
    clean_patterns = [str(item).strip() for item in group_patterns if str(item).strip()]
    if not clean_patterns:
        return []

    discovered_vars: list[str] = []
    for pattern in clean_patterns:
        for token in _VARIABLE_TOKEN_RE.findall(pattern):
            if token not in discovered_vars:
                discovered_vars.append(token)
    if not discovered_vars:
        return []

    clean_configured_vars = [str(item).strip() for item in configured_vars if str(item).strip()]
    if not clean_configured_vars:
        return discovered_vars

    filtered = [item for item in clean_configured_vars if item in discovered_vars]
    if filtered:
        return filtered
    return discovered_vars


# ---------------------------------------------------------------------------
# Snapshot query adapter
# ---------------------------------------------------------------------------

class KnowledgeSnapshotClient:
    """Fetch KnowledgeCore rows and format them into prompt-ready snapshots."""

    def __init__(self, node, callback_group, service_name: str, timeout_sec: float) -> None:
        """Bind the reusable kb_skills client for one chatbot node instance."""
        self._query_client = KnowledgeCoreQueryClient(
            node=node,
            callback_group=callback_group,
            service_name=service_name,
            timeout_sec=timeout_sec,
        )
        self.last_rows: tuple[dict, ...] = ()

    def fetch_snapshot(
        self,
        settings: KnowledgeSnapshotSettings,
        *,
        turn_id: str = '',
        trace=None,
    ) -> str:
        """Return one formatted snapshot for the current turn or an empty string."""
        if not settings.enabled:
            self.last_rows = ()
            return ''

        all_rows: list[dict] = []
        groups = settings.query_groups or [list(settings.patterns)]
        for group in groups:
            clean_group = [str(item).strip() for item in group if str(item).strip()]
            if not clean_group:
                continue
            group_query_vars = _query_vars_for_group(clean_group, list(settings.query_vars))
            all_rows.extend(
                self._query_client.query_rows(
                    patterns=clean_group,
                    query_vars=group_query_vars,
                    models=list(settings.models),
                    turn_id=turn_id,
                    trace=trace,
                    trace_stage='KB_SNAPSHOT',
                )
            )

        deduped_rows = KnowledgeCoreQueryClient.dedupe_rows(all_rows)
        self.last_rows = tuple(dict(item) for item in deduped_rows)
        snapshot = format_knowledge_snapshot(
            json.dumps(deduped_rows),
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

    def enrich_snapshot_for_subjects(
        self,
        snapshot: str,
        settings: KnowledgeSnapshotSettings,
        subject_ids: list[str],
        *,
        turn_id: str = '',
        trace=None,
    ) -> str:
        """Add bounded relation rows for canonical subjects already resolved in state."""
        if not settings.enabled or not subject_ids:
            return snapshot
        rows = list(self.last_rows)
        for subject_id in subject_ids[:4]:
            rows.extend(
                self._query_client.query_rows(
                    patterns=['%s ?predicate ?object' % subject_id],
                    query_vars=['?predicate', '?object'],
                    models=list(settings.models),
                    turn_id=turn_id,
                    trace=trace,
                    trace_stage='KB_SUBJECT_LOOKUP',
                )
            )
        deduped_rows = KnowledgeCoreQueryClient.dedupe_rows(rows)
        self.last_rows = tuple(dict(item) for item in deduped_rows)
        enriched = format_knowledge_snapshot(json.dumps(deduped_rows), settings)
        self._trace(
            trace,
            turn_id,
            'KB_SUBJECT_LOOKUP',
            'resolved subjects=%s rows=%d' % (','.join(subject_ids[:4]), len(deduped_rows)),
        )
        return enriched or snapshot

    @staticmethod
    def _trace(trace, turn_id: str, stage: str, message: str, level: str = 'info') -> None:
        """Forward trace hooks without forcing callers to provide one."""
        if callable(trace):
            trace(turn_id, stage, message, level=level)
