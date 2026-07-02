"""Formatting adapter over the reusable `kb_skills` query client."""

from __future__ import annotations

import json
import re

from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import format_knowledge_snapshot
from kb_skills.query_client import KnowledgeCoreQueryClient

_VARIABLE_TOKEN_RE = re.compile(r'\?[A-Za-z_][A-Za-z0-9_]*')
_KB_SUBJECT_TOKEN_RE = re.compile(r'\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b')
_KB_NAME_TOKEN_RE = re.compile(r'\b[A-Z][A-Z0-9_]{2,}\b')
_SUBJECT_QUERY_CUES = (
    'about',
    'remember',
    'know',
    'knowledge',
    'name',
    'named',
    'called',
    'color',
    'colour',
)
_MAX_SUBJECT_QUERY_GROUPS = 4


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
        user_text: str = '',
        turn_id: str = '',
        trace=None,
    ) -> str:
        """Return one formatted snapshot for the current turn or an empty string."""
        if not settings.enabled:
            self.last_rows = ()
            return ''

        all_rows: list[dict] = []
        groups = list(settings.query_groups or [list(settings.patterns)])
        groups.extend(_subject_query_groups(user_text))
        for group in groups:
            clean_group = [str(item).strip() for item in group if str(item).strip()]
            if not clean_group:
                continue
            group_query_vars = _query_vars_for_group(clean_group, list(settings.query_vars))
            rows = self._query_client.query_rows(
                patterns=clean_group,
                query_vars=group_query_vars,
                models=list(settings.models),
                turn_id=turn_id,
                trace=trace,
                trace_stage='KB_SNAPSHOT',
            )
            subject_entity = _constant_subject_from_group(clean_group)
            if subject_entity:
                rows = _annotate_constant_subject_rows(rows, subject_entity)
            all_rows.extend(rows)

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


def _subject_query_groups(user_text: str) -> list[list[str]]:
    """Build narrow KB queries for explicit subject-recall turns.

    The default snapshot intentionally follows the current visible scene. A
    question that names a KB subject, however, should query that subject directly
    so facts can be recalled even when the object is not in the visible scene.
    """
    clean_text = str(user_text or '').strip()
    if not clean_text or not _contains_subject_query_cue(clean_text):
        return []

    groups: list[list[str]] = []
    seen: set[str] = set()
    for token in _KB_SUBJECT_TOKEN_RE.findall(clean_text):
        clean_token = token.strip()
        if clean_token and clean_token not in seen:
            seen.add(clean_token)
            groups.append([f'{clean_token} ?predicate ?object'])

    for token in _KB_NAME_TOKEN_RE.findall(clean_text):
        clean_token = token.strip()
        if clean_token and clean_token not in seen:
            seen.add(clean_token)
            groups.append([f'?entity dbp:name {clean_token}', '?entity ?predicate ?object'])

    return groups[:_MAX_SUBJECT_QUERY_GROUPS]


def _constant_subject_from_group(group: list[str]) -> str:
    if len(group) != 1:
        return ''
    parts = str(group[0]).strip().split()
    if len(parts) != 3:
        return ''
    subject = parts[0].strip()
    if not subject or subject.startswith('?'):
        return ''
    return subject


def _annotate_constant_subject_rows(rows: list[dict], subject: str) -> list[dict]:
    clean_subject = str(subject or '').strip()
    if not clean_subject:
        return rows
    annotated = []
    for row in rows:
        if not isinstance(row, dict):
            annotated.append(row)
            continue
        next_row = dict(row)
        next_row.setdefault('entity', clean_subject)
        annotated.append(next_row)
    return annotated


def _contains_subject_query_cue(user_text: str) -> bool:
    lowered = ' %s ' % str(user_text or '').lower()
    return any(' %s' % cue in lowered for cue in _SUBJECT_QUERY_CUES)
