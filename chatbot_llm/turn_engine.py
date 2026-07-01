"""Two-stage turn execution policy for the migrated chatbot backend."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.backend_config import coerce_float
from chatbot_llm.chat_history import history_to_messages
from chatbot_llm.chat_history import messages_to_history
from chatbot_llm.chat_history import trim_messages
from chatbot_llm.intent_rules import build_rule_response
from chatbot_llm.intent_rules import detect_intent
from chatbot_llm.intent_rules import is_execution_intent_label
from chatbot_llm.intent_rules import normalize_intent
from chatbot_llm.prompt_builders import build_intent_prompt
from chatbot_llm.prompt_builders import build_response_prompt
from chatbot_llm.prompt_builders import load_persona_prompt
from chatbot_llm.response_fallbacks import _ack_from_parsed_response
from chatbot_llm.response_fallbacks import _coerce_user_intent
from chatbot_llm.response_fallbacks import _extract_ack_text
from chatbot_llm.response_fallbacks import _extract_json_object
from chatbot_llm.response_fallbacks import _fallback_execution_report_ack
from chatbot_llm.response_fallbacks import _fallback_planner_completion_ack
from chatbot_llm.response_fallbacks import _fallback_planner_dialogue_ack
from chatbot_llm.response_fallbacks import _limit_verbal_ack
from chatbot_llm.response_fallbacks import _looks_like_json_payload
from chatbot_llm.response_fallbacks import _postprocess_execution_report_ack
from chatbot_llm.response_fallbacks import _route_safe_fallback_ack
from chatbot_llm.response_fallbacks import _sanitize_execution_ack
from chatbot_llm.response_fallbacks import _sanitize_locked_route_ack
from chatbot_llm.route_heuristics import _ack_implies_execution
from chatbot_llm.route_heuristics import _dialogue_user_intent
from chatbot_llm.route_heuristics import _DIALOGUE_INTENTS
from chatbot_llm.route_heuristics import _DIALOGUE_ROUTE
from chatbot_llm.route_heuristics import _execution_intent_from_text
from chatbot_llm.route_heuristics import _EXECUTION_ROUTE
from chatbot_llm.route_heuristics import _has_explicit_perception_action_request
from chatbot_llm.route_heuristics import _infer_kb_query_intent_from_text
from chatbot_llm.route_heuristics import _is_advice_or_idea_request
from chatbot_llm.route_heuristics import _is_capability_query
from chatbot_llm.route_heuristics import _is_greeting_intent
from chatbot_llm.route_heuristics import _is_information_only_action_word_question
from chatbot_llm.route_heuristics import _is_non_immediate_action_discussion
from chatbot_llm.route_heuristics import _is_personal_preference_question
from chatbot_llm.route_heuristics import _is_reflective_execution_question
from chatbot_llm.route_heuristics import _is_repeat_action_request
from chatbot_llm.route_heuristics import _is_social_turn
from chatbot_llm.route_heuristics import _KNOWLEDGE_QUERY_ROUTE
from chatbot_llm.route_heuristics import _looks_like_execution_text
from chatbot_llm.route_heuristics import _looks_like_non_mutating_kb_question
from chatbot_llm.route_heuristics import _normalize_route
from chatbot_llm.route_heuristics import _repair_response_route
from chatbot_llm.route_heuristics import _route_is_contradictory
from chatbot_llm.route_heuristics import _rules_execution_intent_allowed
from chatbot_llm.system_turn import _EXECUTION_REPORT_RESPONSE_ADDENDUM
from chatbot_llm.system_turn import _extract_execution_report_context
from chatbot_llm.system_turn import _extract_planner_completion_context
from chatbot_llm.system_turn import _extract_planner_dialogue_context
from chatbot_llm.system_turn import _join_prompt_addenda
from chatbot_llm.system_turn import _locked_route_prompt_block
from chatbot_llm.system_turn import _PLANNER_COMPLETION_RESPONSE_ADDENDUM
from chatbot_llm.system_turn import _PLANNER_DIALOGUE_RESPONSE_ADDENDUM
from chatbot_llm.system_turn import _system_task_response_addendum
from kb_skills.intent_labels import KB_QUERY_INTENTS


# ---------------------------------------------------------------------------
# Turn execution result model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnExecutionResult:
    """Result tuple returned by ``DialogueTurnEngine.execute_turn``."""

    success: bool
    verbal_ack: str
    updated_history: list[str]
    intent: str
    intent_source: str
    intent_confidence: float
    user_intent: dict
    route: str


_UNCLEAR_RESPONSE_ACK = 'I could not understand the request clearly enough. Could you rephrase it?'
_MISSING_NAMED_PERSON_ACK = (
    'I cannot confirm that person in the current grounded context. '
    'Which person should I use for the task?'
)


def _compact_match_text(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


def _iter_context_dicts(value):
    if isinstance(value, dict):
        yield value
        for nested_value in value.values():
            yield from _iter_context_dicts(nested_value)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_context_dicts(item)


def _relation_texts(entity: dict) -> list[str]:
    texts: list[str] = []
    relations = entity.get('relations', [])
    if not isinstance(relations, list):
        return texts
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', '')).lower()
        if 'name' not in predicate and predicate not in {'dbp:name', 'rdfs:label'}:
            continue
        for key in ('object', 'value', 'target', 'label'):
            value = str(relation.get(key, '')).strip()
            if value:
                texts.append(value)
    return texts


def _looks_like_person_entity(entity: dict) -> bool:
    descriptor = ' '.join(
        str(entity.get(key, ''))
        for key in ('kind', 'class', 'type', 'entity_class', 'label', 'id')
    ).lower()
    if 'location' in descriptor or 'place' in descriptor or 'room' in descriptor:
        return False
    return any(marker in descriptor for marker in ('person', 'human', 'recipient'))


def _person_name_texts(entity: dict) -> list[str]:
    texts = []
    for key in ('id', 'label', 'name', 'display_name'):
        value = str(entity.get(key, '')).strip()
        if value:
            texts.append(value)
    texts.extend(_relation_texts(entity))
    return texts


def _extract_grounded_context_dict(knowledge_snapshot: str) -> dict:
    parsed = _extract_json_object(str(knowledge_snapshot or ''))
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get('grounded_context'), dict):
        return parsed['grounded_context']
    if any(key in parsed for key in ('entities', 'relations', 'locations')):
        return parsed
    return {}


def _grounded_context_has_person_named(
    *,
    knowledge_snapshot: str,
    requested_name: str,
) -> bool:
    compact_name = _compact_match_text(requested_name)
    if not compact_name:
        return True
    context = _extract_grounded_context_dict(knowledge_snapshot)
    for entity in _iter_context_dicts(context):
        if not _looks_like_person_entity(entity):
            continue
        for candidate in _person_name_texts(entity):
            compact_candidate = _compact_match_text(candidate)
            if compact_candidate and compact_candidate == compact_name:
                return True
    return False


def _requested_named_people(user_text: str, user_intent: dict) -> list[str]:
    candidates: list[str] = []
    text = str(user_text or '')
    for pattern in (
        r'\b(?:person|human|recipient)\s+(?:named|called)\s+([A-Za-z][A-Za-z0-9_-]*)',
        r'\bto\s+(?:the\s+)?(?:person|human|recipient)\s+(?!named\b|called\b)([A-Za-z][A-Za-z0-9_-]*)',
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append(match.group(1))

    for key in ('recipient', 'target_person'):
        value = str(user_intent.get(key, '')).strip()
        if value:
            candidates.append(value)

    if re.search(r'\b(?:person|human|recipient)\b', text, flags=re.IGNORECASE):
        scene_targets = user_intent.get('scene_targets', [])
        if isinstance(scene_targets, str):
            scene_targets = [scene_targets]
        if isinstance(scene_targets, (list, tuple)):
            for target in scene_targets:
                value = str(target or '').strip()
                if value and not value.lower().startswith('codex_'):
                    candidates.append(value)

    unique: list[str] = []
    seen = set()
    for candidate in candidates:
        clean = str(candidate or '').strip(' ,.;:!?')
        compact = _compact_match_text(clean)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        unique.append(clean)
    return unique


# ---------------------------------------------------------------------------
# Two-stage turn engine
# ---------------------------------------------------------------------------

class DialogueTurnEngine:
    """Implement rules/LLM policies for ``chatbot_llm`` interactions."""

    def __init__(
        self,
        config: ChatbotConfig,
        transport,
        logger,
        skill_catalog_text: str,
    ) -> None:
        """Create one reusable two-stage turn engine."""
        self._config = config
        self._transport = transport
        self._logger = logger
        self._skill_catalog_text = str(skill_catalog_text or '').strip()
        self._persona_prompt = load_persona_prompt(config.persona_prompt_path, logger=logger)
        self._handled_requests = 0

    # -----------------------------------------------------------------------
    # Public turn execution path
    # -----------------------------------------------------------------------

    def execute_turn(
        self,
        user_text: str,
        history: list[str],
        user_id: str,
        knowledge_snapshot: str = '',
        progress_callback=None,
        turn_id: str = '',
        trace=None,
        cancel_requested=None,
    ) -> TurnExecutionResult:
        """Execute one full turn with policy/fallback logic."""
        cancel_requested = cancel_requested or (lambda: False)
        self._trace(trace, turn_id, 'TURN_START', f'user="{self._preview_text(user_text)}"')
        self._publish_progress(progress_callback, 'thinking', 0.1)
        if cancel_requested():
            return self._cancelled_result(history)

        if user_id == '__system__':
            execution_report = _extract_execution_report_context(user_text)
            if execution_report:
                result = self._execute_execution_report_turn(
                    history=history,
                    report_context=execution_report,
                )
                self._publish_progress(progress_callback, 'complete', 1.0)
                self._trace(trace, turn_id, 'TURN_DONE', 'execution report wording complete')
                return result
            planner_dialogue = _extract_planner_dialogue_context(user_text)
            if planner_dialogue:
                result = self._execute_planner_dialogue_turn(
                    history=history,
                    dialogue_context=planner_dialogue,
                )
                self._publish_progress(progress_callback, 'complete', 1.0)
                self._trace(trace, turn_id, 'TURN_DONE', 'planner dialogue wording complete')
                return result
            planner_completion = _extract_planner_completion_context(user_text)
            if planner_completion:
                result = self._execute_planner_completion_turn(
                    history=history,
                    completion_context=planner_completion,
                )
                self._publish_progress(progress_callback, 'complete', 1.0)
                self._trace(trace, turn_id, 'TURN_DONE', 'planner completion wording complete')
                return result

        if self._config.intent_detection_mode == 'rules':
            result = self._execute_rule_turn(user_text=user_text, history=history, source='rules')
            self._publish_progress(progress_callback, 'complete', 1.0)
            self._trace(trace, turn_id, 'TURN_DONE', 'rule path complete')
            return result

        if not self._config.enabled:
            if self._config.intent_detection_mode == 'llm_with_rules_fallback':
                result = self._execute_rule_turn(
                    user_text=user_text,
                    history=history,
                    source='rules_llm_disabled',
                )
                self._publish_progress(progress_callback, 'complete', 1.0)
                self._trace(trace, turn_id, 'TURN_DONE', 'llm disabled -> rules path')
                return result
            result = self._execute_disabled_turn(user_text=user_text, history=history)
            self._publish_progress(progress_callback, 'complete', 1.0)
            self._trace(trace, turn_id, 'TURN_DONE', 'llm disabled fallback response')
            return result

        history_messages = history_to_messages(
            history,
            max_history_messages=self._config.max_history_messages,
        )
        history_messages = self._inject_identity_reminder(history_messages)

        if (
            self._config.planner_mode_enabled
            and self._config.turn_pipeline_mode == 'intent_first'
        ):
            result = self._execute_intent_first_planner_turn(
                user_text=user_text,
                history=history,
                history_messages=history_messages,
                user_id=user_id,
                knowledge_snapshot=knowledge_snapshot,
                progress_callback=progress_callback,
                turn_id=turn_id,
                trace=trace,
                cancel_requested=cancel_requested,
            )
            self._publish_progress(progress_callback, 'complete', 1.0)
            self._trace(trace, turn_id, 'TURN_DONE', 'intent-first planner-mode complete')
            return result

        self._publish_progress(progress_callback, 'generating_response', 0.35)
        if cancel_requested():
            return self._cancelled_result(history)

        stage1_timeout_sec = (
            self._config.first_request_timeout_sec
            if self._handled_requests == 0
            else self._config.request_timeout_sec
        )
        self._trace(
            trace,
            turn_id,
            'LLM_REQUEST',
            'stage=response model=%s history=%d timeout=%.1fs'
            % (self._config.model, len(history_messages), stage1_timeout_sec),
        )
        response_payload = self._query_response(
            history_messages=history_messages,
            user_text=user_text,
            user_id=user_id,
            knowledge_snapshot=knowledge_snapshot,
            timeout_sec=stage1_timeout_sec,
        )
        verbal_ack = str(response_payload.get('verbal_ack', '')).strip()
        if not verbal_ack:
            if (
                self._config.intent_detection_mode == 'llm_with_rules_fallback'
                and not self._config.planner_mode_enabled
            ):
                result = self._execute_rule_turn(
                    user_text=user_text,
                    history=history,
                    source='rules_llm_response_fallback',
                )
                self._publish_progress(progress_callback, 'complete', 1.0)
                self._trace(trace, turn_id, 'TURN_DONE', 'llm response fallback -> rules')
                return result
            result = self._execute_llm_failure_turn(user_text=user_text, history=history)
            self._publish_progress(progress_callback, 'complete', 1.0)
            self._trace(trace, turn_id, 'TURN_DONE', 'llm response failed fallback')
            return result

        if self._config.planner_mode_enabled:
            llm_route = _normalize_route(response_payload.get('route', ''))
            (
                route,
                resolved_intent,
                intent_source,
                intent_confidence,
                user_intent,
            ) = self._resolve_planner_mode_turn(
                user_text=user_text,
                verbal_ack=verbal_ack,
                response_payload=response_payload,
                knowledge_snapshot=knowledge_snapshot,
            )
            if route == _EXECUTION_ROUTE:
                verbal_ack = _sanitize_execution_ack(verbal_ack)
            elif (
                isinstance(user_intent.get('route_conflict'), dict)
                and user_intent['route_conflict'].get('reason')
                == 'missing_named_person_in_grounded_context'
            ):
                verbal_ack = _MISSING_NAMED_PERSON_ACK
            self._trace(
                trace,
                turn_id,
                'ROUTE_RESOLVED',
                'route=%s intent=%s source=%s confidence=%.2f'
                % (route, resolved_intent or '-', intent_source, intent_confidence),
            )
            self._trace_route_decision(
                trace,
                turn_id,
                llm_route=llm_route,
                final_route=route,
                intent_source=intent_source,
            )

            updated_history = messages_to_history(
                history_messages
                + [
                    {'role': 'user', 'content': user_text},
                    {'role': 'assistant', 'content': verbal_ack},
                ],
                max_history_messages=self._config.max_history_messages,
            )

            self._handled_requests += 1
            self._publish_progress(progress_callback, 'complete', 1.0)
            self._trace(trace, turn_id, 'TURN_DONE', 'planner-mode response complete')
            return self._build_result(
                success=True,
                verbal_ack=verbal_ack,
                updated_history=updated_history,
                intent=resolved_intent,
                intent_source=intent_source,
                intent_confidence=intent_confidence,
                user_intent=user_intent,
                route=route,
            )

        self._publish_progress(progress_callback, 'extracting_intent', 0.7)
        if cancel_requested():
            return self._cancelled_result(history)

        self._trace(
            trace,
            turn_id,
            'LLM_REQUEST',
            'stage=intent model=%s timeout=%.1fs'
            % (self._config.intent_model, self._config.intent_request_timeout_sec),
        )
        intent_payload = self._query_intent(
            history_messages=history_messages,
            user_text=user_text,
            assistant_response=verbal_ack,
            user_id=user_id,
            knowledge_snapshot=knowledge_snapshot,
            timeout_sec=self._config.intent_request_timeout_sec,
        )

        (
            resolved_intent,
            intent_source,
            intent_confidence,
            user_intent,
        ) = self._resolve_intent(
            user_text=user_text,
            verbal_ack=verbal_ack,
            intent_payload=intent_payload,
        )
        self._trace(
            trace,
            turn_id,
            'INTENT_RESOLVED',
            'intent=%s source=%s confidence=%.2f'
            % (resolved_intent, intent_source, intent_confidence),
        )

        updated_history = messages_to_history(
            history_messages
            + [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': verbal_ack},
            ],
            max_history_messages=self._config.max_history_messages,
        )

        self._handled_requests += 1
        self._publish_progress(progress_callback, 'complete', 1.0)
        self._trace(trace, turn_id, 'TURN_DONE', 'chat backend complete')
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent=resolved_intent,
            intent_source=intent_source,
            intent_confidence=intent_confidence,
            user_intent=user_intent,
            route=self._route_for_intent(resolved_intent),
        )

    # -----------------------------------------------------------------------
    # Intent resolution and fallback paths
    # -----------------------------------------------------------------------

    def _execute_intent_first_planner_turn(
        self,
        *,
        user_text: str,
        history: list[str],
        history_messages: list[dict],
        user_id: str,
        knowledge_snapshot: str,
        progress_callback,
        turn_id: str,
        trace,
        cancel_requested,
    ) -> TurnExecutionResult:
        self._publish_progress(progress_callback, 'extracting_intent', 0.25)
        if cancel_requested():
            return self._cancelled_result(history)

        self._trace(
            trace,
            turn_id,
            'LLM_REQUEST',
            'stage=intent_first model=%s timeout=%.1fs'
            % (self._config.intent_model, self._config.intent_request_timeout_sec),
        )
        intent_payload = self._query_intent(
            history_messages=history_messages,
            user_text=user_text,
            assistant_response='',
            user_id=user_id,
            knowledge_snapshot=knowledge_snapshot,
            timeout_sec=self._config.intent_request_timeout_sec,
        )
        (
            resolved_intent,
            intent_source,
            intent_confidence,
            user_intent,
        ) = self._resolve_intent(
            user_text=user_text,
            verbal_ack='',
            intent_payload=intent_payload,
        )
        llm_route = self._route_for_intent(
            str(user_intent.get('type', '')).strip() or resolved_intent
        )
        route, resolved_intent, user_intent, intent_source = self._lock_intent_first_route(
            user_text=user_text,
            resolved_intent=resolved_intent,
            user_intent=user_intent,
            intent_source=intent_source,
            intent_confidence=intent_confidence,
        )
        self._trace(
            trace,
            turn_id,
            'ROUTE_LOCKED',
            'mode=intent_first route=%s intent=%s source=%s confidence=%.2f'
            % (route, resolved_intent or '-', intent_source, intent_confidence),
        )
        self._trace_route_decision(
            trace,
            turn_id,
            llm_route=llm_route,
            final_route=route,
            intent_source=intent_source,
        )

        self._publish_progress(progress_callback, 'generating_response', 0.65)
        if cancel_requested():
            return self._cancelled_result(history)

        response_payload = self._query_response(
            history_messages=history_messages,
            user_text=user_text,
            user_id=user_id,
            knowledge_snapshot=knowledge_snapshot,
            timeout_sec=self._config.request_timeout_sec,
            locked_route_context={
                'route': route,
                'resolved_intent': resolved_intent,
                'user_intent': user_intent,
            },
        )
        verbal_ack = str(response_payload.get('verbal_ack', '')).strip()
        if not verbal_ack:
            verbal_ack = _route_safe_fallback_ack(route)
        verbal_ack = _sanitize_locked_route_ack(
            route=route,
            user_text=user_text,
            verbal_ack=verbal_ack,
        )

        updated_history = messages_to_history(
            history_messages
            + [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': verbal_ack},
            ],
            max_history_messages=self._config.max_history_messages,
        )
        self._handled_requests += 1
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent=resolved_intent,
            intent_source=intent_source,
            intent_confidence=intent_confidence,
            user_intent=user_intent,
            route=route,
        )

    def _lock_intent_first_route(
        self,
        *,
        user_text: str,
        resolved_intent: str,
        user_intent: dict,
        intent_source: str,
        intent_confidence: float,
    ) -> tuple[str, str, dict, str]:
        if (
            _is_reflective_execution_question(user_text)
            or _is_social_turn(user_text)
            or _is_capability_query(user_text)
            or _is_personal_preference_question(user_text)
            or _is_information_only_action_word_question(user_text)
            or _is_non_immediate_action_discussion(user_text)
        ):
            return _DIALOGUE_ROUTE, '', _dialogue_user_intent(user_intent), (
                intent_source + '_route_lock'
            )

        locked_intent_route = self._route_for_intent(str(user_intent.get('type', '')).strip())
        if _looks_like_execution_text(user_text) and locked_intent_route == _KNOWLEDGE_QUERY_ROUTE:
            fallback_intent = _execution_intent_from_text(user_text)
            if (
                fallback_intent
                and fallback_intent != 'fallback'
                and is_execution_intent_label(fallback_intent)
                and _rules_execution_intent_allowed(user_text)
            ):
                locked_user_intent = dict(user_intent)
                locked_user_intent['type'] = fallback_intent
                locked_user_intent['goal'] = str(user_text or '').strip()
                locked_user_intent['goal_text'] = str(user_text or '').strip()
                return (
                    _EXECUTION_ROUTE,
                    fallback_intent,
                    locked_user_intent,
                    intent_source + '_route_lock',
                )
        kb_query_intent = _infer_kb_query_intent_from_text(user_text)
        if kb_query_intent:
            locked_intent = kb_query_intent
            locked_user_intent = dict(user_intent)
            locked_user_intent['type'] = locked_intent
            return (
                _KNOWLEDGE_QUERY_ROUTE,
                locked_intent,
                locked_user_intent,
                intent_source + '_route_lock',
            )

        route = self._route_for_intent(str(user_intent.get('type', '')).strip() or resolved_intent)
        if route == _EXECUTION_ROUTE and not _rules_execution_intent_allowed(user_text):
            return _DIALOGUE_ROUTE, '', _dialogue_user_intent(user_intent), (
                intent_source + '_route_lock'
            )
        if route == _DIALOGUE_ROUTE and _looks_like_execution_text(user_text):
            fallback_intent = normalize_intent(detect_intent(user_text), default='')
            if (
                fallback_intent
                and fallback_intent != 'fallback'
                and is_execution_intent_label(fallback_intent)
                and _rules_execution_intent_allowed(user_text)
            ):
                locked_user_intent = dict(user_intent)
                locked_user_intent['type'] = fallback_intent
                if not str(locked_user_intent.get('goal', '')).strip():
                    locked_user_intent['goal'] = str(user_text or '').strip()
                return (
                    _EXECUTION_ROUTE,
                    fallback_intent,
                    locked_user_intent,
                    intent_source + '_route_lock',
                )
        return route, resolved_intent, user_intent, intent_source + '_route_lock'

    def _resolve_intent(
        self,
        user_text: str,
        verbal_ack: str,
        intent_payload: dict,
    ) -> tuple[str, str, float, dict]:
        if intent_payload:
            raw_user_intent = intent_payload.get('user_intent', intent_payload)
            user_intent = _coerce_user_intent(raw_user_intent)
            hint_text = ' '.join(
                [
                    user_intent.get('object', ''),
                    user_intent.get('goal', ''),
                    user_intent.get('input', ''),
                    verbal_ack,
                ]
            ).strip()
            resolved = normalize_intent(
                user_intent.get('type', ''),
                default='',
                hint_text=hint_text,
            )
            confidence = coerce_float(
                intent_payload.get(
                    'intent_confidence',
                    intent_payload.get('confidence', 0.0),
                )
            )
            if resolved:
                return resolved, 'llm_intent', confidence, user_intent

        if self._config.intent_detection_mode == 'llm_with_rules_fallback':
            fallback_intent = detect_intent(user_text)
            if (
                is_execution_intent_label(fallback_intent)
                and not _rules_execution_intent_allowed(user_text)
            ):
                fallback_intent = 'fallback'
            fallback_user_intent = (
                {'type': fallback_intent}
                if fallback_intent != 'fallback'
                else {}
            )
            fallback_confidence = 1.0 if fallback_intent != 'fallback' else 0.0
            return (
                fallback_intent,
                'rules_llm_intent_fallback',
                fallback_confidence,
                fallback_user_intent,
            )

        return 'fallback', 'llm_intent_failed', 0.0, {}

    def _resolve_planner_mode_turn(
        self,
        *,
        user_text: str,
        verbal_ack: str,
        response_payload: dict,
        knowledge_snapshot: str = '',
    ) -> tuple[str, str, str, float, dict]:
        """Resolve the final route for a response-first planner-mode turn.

        This is an ordered route policy applied on top of the LLM-provided
        ``route``/``user_intent``. The phases run in this fixed order and may
        each rewrite the in-progress ``inferred_route``/``resolved_intent``/
        ``user_intent``:

        1. seed from the LLM route plus a rules-based inference;
        2. dialogue-first guards (reflective/greeting/social/capability/
           preference/information-only questions stay conversational);
        3. execution-repair guards (repeat-action and ack-implied execution);
        4. contradiction repair when the route fights the acknowledgement;
        5. rules-fallback intent backfill;
        6. knowledge-query guard;
        7. non-immediate action discussion guard;
        8. intent-source labelling.

        Phase 2 (SkillOpt-gated) is where these guards are reduced toward
        "LLM leads, fallbacks only nudge"; here they stay behaviour-preserving.
        """
        user_intent = _coerce_user_intent(response_payload.get('user_intent', {}))
        explicit_route = _normalize_route(response_payload.get('route', ''))
        resolved_intent = self._resolve_user_intent_label(
            user_text=user_text,
            verbal_ack=verbal_ack,
            user_intent=user_intent,
        )
        inferred_route = self._infer_route(
            requested_route=response_payload.get('route', ''),
            user_text=user_text,
            resolved_intent=resolved_intent,
            user_intent=user_intent,
        )
        route_repaired = False
        if _is_reflective_execution_question(user_text):
            inferred_route = _DIALOGUE_ROUTE
            resolved_intent = ''
            user_intent = _dialogue_user_intent(user_intent)
        # Keep greeting-like, short-social, and capability-question turns
        # dialogue-first unless the user clearly asked for an execution action.
        # These three guards each only force the dialogue route with no other
        # side effect, so they collapse into one ordered check.
        if (
            (
                _is_greeting_intent(resolved_intent, user_intent)
                and not _looks_like_execution_text(user_text)
            )
            or (_is_social_turn(user_text) and not _looks_like_execution_text(user_text))
            or _is_capability_query(user_text)
        ):
            inferred_route = _DIALOGUE_ROUTE
        if _is_personal_preference_question(user_text):
            inferred_route = _DIALOGUE_ROUTE
            resolved_intent = ''
            user_intent = {}
            route_repaired = explicit_route == _EXECUTION_ROUTE
        if _is_information_only_action_word_question(user_text):
            inferred_route = _DIALOGUE_ROUTE
            resolved_intent = ''
            user_intent = _dialogue_user_intent(user_intent)
            route_repaired = explicit_route == _EXECUTION_ROUTE
        if (
            inferred_route == _DIALOGUE_ROUTE
            and explicit_route == _DIALOGUE_ROUTE
            and _is_repeat_action_request(user_text)
            and _ack_implies_execution(verbal_ack)
            and not _is_reflective_execution_question(user_text)
            and not _is_capability_query(user_text)
        ):
            inferred_route = _EXECUTION_ROUTE
            if not str(user_intent.get('goal', '')).strip():
                user_intent = dict(user_intent)
                user_intent['goal'] = str(user_text or '').strip()
        if (
            inferred_route == _DIALOGUE_ROUTE
            and not explicit_route
            and not _is_social_turn(user_text)
            and not _is_capability_query(user_text)
            and not _is_advice_or_idea_request(user_text)
            and not _is_reflective_execution_question(user_text)
            and not _is_information_only_action_word_question(user_text)
            and _ack_implies_execution(verbal_ack)
        ):
            inferred_route = _EXECUTION_ROUTE
            if not str(user_intent.get('goal', '')).strip():
                user_intent = dict(user_intent)
                user_intent['goal'] = str(user_text or '').strip()

        # If the LLM routed to dialogue (e.g., mapped "wave at me" to greet),
        # but the rules-fallback detects an executable skill intent, force
        # execution so action requests do not get swallowed as conversation.
        if (
            inferred_route == _DIALOGUE_ROUTE
            and not explicit_route
            and not _is_reflective_execution_question(user_text)
            and not _is_advice_or_idea_request(user_text)
            and not _is_information_only_action_word_question(user_text)
        ):
            fb_intent = normalize_intent(detect_intent(user_text), default='')
            if (
                fb_intent
                and fb_intent != 'fallback'
                and is_execution_intent_label(fb_intent)
                and _rules_execution_intent_allowed(user_text)
            ):
                inferred_route = _EXECUTION_ROUTE
                resolved_intent = fb_intent
                user_intent = dict(user_intent)
                user_intent['type'] = fb_intent
                if not str(user_intent.get('goal', '')).strip():
                    user_intent['goal'] = str(user_text or '').strip()

        response_confidence = coerce_float(
            response_payload.get(
                'intent_confidence',
                response_payload.get('confidence', 0.0),
            )
        )
        if response_payload.get('_route_missing') or _route_is_contradictory(
            user_text=user_text,
            verbal_ack=verbal_ack,
            route=inferred_route,
        ):
            repaired_route = _repair_response_route(
                user_text=user_text,
                verbal_ack=verbal_ack,
                route=inferred_route,
            )
            if repaired_route:
                inferred_route = repaired_route
                route_repaired = True
                if repaired_route == _DIALOGUE_ROUTE:
                    user_intent = _dialogue_user_intent(user_intent)
                elif repaired_route == _EXECUTION_ROUTE:
                    user_intent = dict(user_intent)
                    repaired_intent = normalize_intent(detect_intent(user_text), default='')
                    if (
                        repaired_intent
                        and repaired_intent != 'fallback'
                        and str(user_intent.get('type', '')).strip().lower()
                        in {'', 'fallback'}
                    ):
                        resolved_intent = repaired_intent
                        user_intent['type'] = repaired_intent
                    if not str(user_intent.get('goal', '')).strip():
                        user_intent['goal'] = str(user_text or '').strip()

        if not resolved_intent:
            fallback_intent = ''
            if not _is_reflective_execution_question(user_text):
                combined_hint_text = ' '.join(
                    item.strip()
                    for item in (str(user_text or ''), str(verbal_ack or ''))
                    if str(item or '').strip()
                ).strip()
                fallback_intent = normalize_intent(
                    detect_intent(combined_hint_text),
                    default='',
                )
            if fallback_intent and fallback_intent != 'fallback':
                if is_execution_intent_label(
                    fallback_intent
                ) and not (
                    _rules_execution_intent_allowed(user_text)
                    or (
                        inferred_route == _EXECUTION_ROUTE
                        and _is_repeat_action_request(user_text)
                        and _ack_implies_execution(verbal_ack)
                    )
                ):
                    fallback_intent = ''
            if fallback_intent and fallback_intent != 'fallback':
                resolved_intent = fallback_intent
                if not user_intent:
                    user_intent = {'type': fallback_intent}
                elif not str(user_intent.get('type', '')).strip():
                    user_intent = dict(user_intent)
                    user_intent['type'] = fallback_intent
                # If route came back as dialogue but fallback intent is executable,
                # prefer execution so action intents do not get spoken-only.
                if (
                    inferred_route == _DIALOGUE_ROUTE
                    and not route_repaired
                    and is_execution_intent_label(fallback_intent)
                    and _rules_execution_intent_allowed(user_text)
                ):
                    inferred_route = _EXECUTION_ROUTE

        kb_query_intent = (
            ''
            if (
                _is_personal_preference_question(user_text)
                or _is_information_only_action_word_question(user_text)
            )
            else _infer_kb_query_intent_from_text(user_text)
        )
        if kb_query_intent and not route_repaired:
            stated_intent = str(user_intent.get('type', '')).strip().lower()
            non_mutating_kb_question = _looks_like_non_mutating_kb_question(user_text)
            explicit_non_kb_execution = (
                inferred_route == _EXECUTION_ROUTE
                and _looks_like_execution_text(user_text)
                and stated_intent not in {'kb_add', 'kb_revise', 'kb_remove'}
            )
            if not explicit_non_kb_execution and (non_mutating_kb_question or (
                not _has_explicit_perception_action_request(user_text)
                and stated_intent in {'', 'fallback', 'inspect_scene'}
                and str(resolved_intent or '').strip().lower()
                in {'', 'fallback', 'inspect_scene', kb_query_intent}
            )):
                inferred_route = _KNOWLEDGE_QUERY_ROUTE
                resolved_intent = kb_query_intent
                user_intent = dict(user_intent)
                user_intent['type'] = kb_query_intent

        if _is_non_immediate_action_discussion(user_text):
            inferred_route = _DIALOGUE_ROUTE
            resolved_intent = ''
            user_intent = _dialogue_user_intent(user_intent)
            route_repaired = True

        if inferred_route == _EXECUTION_ROUTE:
            missing_people = [
                name
                for name in _requested_named_people(user_text, user_intent)
                if not _grounded_context_has_person_named(
                    knowledge_snapshot=knowledge_snapshot,
                    requested_name=name,
                )
            ]
            if missing_people:
                user_intent = dict(user_intent)
                user_intent['route_conflict'] = {
                    'requested_person': missing_people[0],
                    'reason': 'missing_named_person_in_grounded_context',
                }
                user_intent = _dialogue_user_intent(user_intent)
                inferred_route = _DIALOGUE_ROUTE
                resolved_intent = ''
                route_repaired = True

        if inferred_route == _DIALOGUE_ROUTE and not resolved_intent and not user_intent:
            intent_source = 'llm_response_route'
        elif route_repaired:
            intent_source = 'llm_response_route_repair'
        elif _normalize_route(response_payload.get('route', '')):
            intent_source = 'llm_response_route'
        else:
            intent_source = 'llm_response_inferred_route'

        return inferred_route, resolved_intent, intent_source, response_confidence, user_intent

    def _resolve_user_intent_label(
        self,
        *,
        user_text: str,
        verbal_ack: str,
        user_intent: dict,
    ) -> str:
        if not user_intent:
            return ''
        hint_text = ' '.join(
            [
                user_intent.get('object', ''),
                user_intent.get('goal', ''),
                user_intent.get('input', ''),
                verbal_ack,
                user_text,
            ]
        ).strip()
        return normalize_intent(
            user_intent.get('type', ''),
            default='',
            hint_text=hint_text,
        )

    def _infer_route(
        self,
        *,
        requested_route,
        user_text: str,
        resolved_intent: str,
        user_intent: dict,
    ) -> str:
        clean_route = _normalize_route(requested_route)
        if clean_route:
            return clean_route

        if _looks_like_execution_text(user_text):
            return _EXECUTION_ROUTE

        intent_route = self._route_for_intent(
            str(user_intent.get('type', '')).strip() or resolved_intent
        )
        if intent_route != _DIALOGUE_ROUTE or resolved_intent or user_intent.get('type'):
            return intent_route

        fallback_intent = normalize_intent(detect_intent(user_text), default='')
        if fallback_intent and fallback_intent != 'fallback':
            return self._route_for_intent(fallback_intent)

        if _looks_like_execution_text(user_text):
            return _EXECUTION_ROUTE
        return _DIALOGUE_ROUTE

    @staticmethod
    def _route_for_intent(intent_name: str) -> str:
        clean_intent = str(intent_name or '').strip().lower()
        if clean_intent in KB_QUERY_INTENTS:
            return _KNOWLEDGE_QUERY_ROUTE
        if clean_intent in _DIALOGUE_INTENTS or clean_intent in ('', 'fallback'):
            return _DIALOGUE_ROUTE
        return _EXECUTION_ROUTE

    def _execute_rule_turn(
        self,
        user_text: str,
        history: list[str],
        source: str,
    ) -> TurnExecutionResult:
        intent = detect_intent(user_text)
        verbal_ack = build_rule_response(intent)
        user_intent = {'type': intent} if intent != 'fallback' else {}
        updated_history = self._history_with_turn(history, user_text, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent=intent,
            intent_source=source,
            intent_confidence=1.0 if intent != 'fallback' else 0.0,
            user_intent=user_intent,
            route=self._route_for_intent(intent),
        )

    def _execute_disabled_turn(self, user_text: str, history: list[str]) -> TurnExecutionResult:
        return self._execute_fallback_turn(
            user_text=user_text,
            history=history,
            intent_source='llm_disabled',
        )

    def _execute_planner_completion_turn(
        self,
        *,
        history: list[str],
        completion_context: dict,
    ) -> TurnExecutionResult:
        verbal_ack = self._query_planner_completion_ack(completion_context)
        if not verbal_ack:
            verbal_ack = _fallback_planner_completion_ack(completion_context)
        if not verbal_ack:
            verbal_ack = self._config.fallback_response
        updated_history = self._history_with_assistant_text(history, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent='',
            intent_source='planner_completion',
            intent_confidence=1.0,
            user_intent={},
            route=_DIALOGUE_ROUTE,
        )

    def _execute_execution_report_turn(
        self,
        *,
        history: list[str],
        report_context: dict,
    ) -> TurnExecutionResult:
        verbal_ack = self._query_execution_report_ack(report_context)
        if not verbal_ack:
            verbal_ack = _fallback_execution_report_ack(report_context)
        if not verbal_ack:
            verbal_ack = self._config.fallback_response
        updated_history = self._history_with_assistant_text(history, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent='',
            intent_source='execution_report',
            intent_confidence=1.0,
            user_intent={},
            route=_DIALOGUE_ROUTE,
        )

    def _execute_planner_dialogue_turn(
        self,
        *,
        history: list[str],
        dialogue_context: dict,
    ) -> TurnExecutionResult:
        verbal_ack = self._query_planner_dialogue_ack(dialogue_context)
        if not verbal_ack:
            verbal_ack = _fallback_planner_dialogue_ack(dialogue_context)
        if not verbal_ack:
            verbal_ack = self._config.fallback_response
        updated_history = self._history_with_assistant_text(history, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=True,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent='',
            intent_source='planner_dialogue',
            intent_confidence=1.0,
            user_intent={},
            route=_DIALOGUE_ROUTE,
        )

    def _execute_llm_failure_turn(self, user_text: str, history: list[str]) -> TurnExecutionResult:
        if self._config.planner_mode_enabled and _looks_like_execution_text(user_text):
            return self._execute_planner_timeout_turn(user_text=user_text, history=history)
        return self._execute_fallback_turn(
            user_text=user_text,
            history=history,
            intent_source='llm_response_failed',
        )

    def _execute_planner_timeout_turn(self, user_text: str, history: list[str]) -> TurnExecutionResult:
        verbal_ack = 'I will try that now.'
        updated_history = self._history_with_turn(history, user_text, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=False,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent='fallback',
            intent_source='llm_response_failed_execution_handoff',
            intent_confidence=0.0,
            user_intent={'type': 'fallback', 'goal_text': str(user_text or '').strip()},
            route=_EXECUTION_ROUTE,
        )

    def _execute_fallback_turn(
        self,
        *,
        user_text: str,
        history: list[str],
        intent_source: str,
    ) -> TurnExecutionResult:
        verbal_ack = self._config.fallback_response
        updated_history = self._history_with_turn(history, user_text, verbal_ack)
        self._handled_requests += 1
        return self._build_result(
            success=False,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent='fallback',
            intent_source=intent_source,
            intent_confidence=0.0,
            user_intent={},
            route=_DIALOGUE_ROUTE,
        )

    def _cancelled_result(self, history: list[str]) -> TurnExecutionResult:
        return self._build_result(
            success=False,
            verbal_ack='',
            updated_history=list(history),
            intent='fallback',
            intent_source='cancelled',
            intent_confidence=0.0,
            user_intent={},
            route=_DIALOGUE_ROUTE,
        )

    # -----------------------------------------------------------------------
    # LLM querying stages
    # -----------------------------------------------------------------------

    def _query_response(
        self,
        history_messages: list[dict],
        user_text: str,
        user_id: str,
        knowledge_snapshot: str,
        timeout_sec: float,
        locked_route_context: dict | None = None,
    ) -> dict:
        prompt = build_response_prompt(
            robot_name=self._config.robot_name,
            user_id=user_id,
            system_prompt=self._config.system_prompt,
            environment_description=self._config.environment_description,
            knowledge_snapshot=knowledge_snapshot,
            response_prompt_addendum=self._config.response_prompt_addendum,
            skill_catalog_text=self._skill_catalog_text,
            persona_prompt=self._persona_prompt,
            planner_mode_enabled=self._config.planner_mode_enabled,
        )
        if locked_route_context:
            prompt = _join_prompt_addenda(
                prompt,
                _locked_route_prompt_block(locked_route_context),
            )
        messages = [{'role': 'system', 'content': prompt}]
        messages.extend(history_messages)
        messages.append({'role': 'user', 'content': user_text})
        messages = trim_messages(messages, max_history_messages=self._config.max_history_messages)

        raw_response = self._transport.query(
            messages=messages,
            timeout_sec=timeout_sec,
            model=self._config.model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            think=self._config.think,
            max_tokens=self._config.response_max_tokens,
            response_format=self._config.response_schema,
        )
        if not raw_response:
            return {}

        parsed = _extract_json_object(raw_response)
        if parsed:
            verbal_ack = _ack_from_parsed_response(parsed)
            if verbal_ack:
                payload = {'verbal_ack': verbal_ack}
                route = _normalize_route(parsed.get('route', ''))
                if route:
                    payload['route'] = route
                else:
                    payload['_route_missing'] = True
                user_intent = _coerce_user_intent(parsed.get('user_intent', {}))
                if user_intent:
                    payload['user_intent'] = user_intent
                confidence = coerce_float(
                    parsed.get('confidence', parsed.get('intent_confidence', 0.0))
                )
                if confidence > 0.0:
                    payload['confidence'] = confidence
                return payload
        ack_text = _extract_ack_text(raw_response)
        if ack_text:
            return {'verbal_ack': ack_text, '_route_missing': True}
        if _looks_like_json_payload(raw_response):
            _warn(self._logger, 'Response JSON did not include a safe verbal acknowledgement')
            return {
                'verbal_ack': _UNCLEAR_RESPONSE_ACK,
                '_route_missing': True,
                '_invalid_response_payload': True,
            }
        return {'verbal_ack': str(raw_response).strip(), '_route_missing': True}

    def _query_intent(
        self,
        history_messages: list[dict],
        user_text: str,
        assistant_response: str,
        user_id: str,
        knowledge_snapshot: str,
        timeout_sec: float,
    ) -> dict:
        prompt = build_intent_prompt(
            robot_name=self._config.robot_name,
            user_id=user_id,
            system_prompt=self._config.system_prompt,
            environment_description=self._config.environment_description,
            knowledge_snapshot=knowledge_snapshot,
            intent_prompt_addendum=self._config.intent_prompt_addendum,
            skill_catalog_text=self._skill_catalog_text,
            persona_prompt=self._persona_prompt,
        )
        messages = [{'role': 'system', 'content': prompt}]
        messages.extend(history_messages)
        messages.extend(
            [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': assistant_response},
                {
                    'role': 'user',
                    'content': json.dumps(
                        {
                            'task': 'Extract user intent in canonical JSON form',
                            'user_text': user_text,
                            'assistant_response': assistant_response,
                        },
                        separators=(',', ':'),
                    ),
                },
            ]
        )
        messages = trim_messages(messages, max_history_messages=self._config.max_history_messages)

        raw_response = self._transport.query(
            messages=messages,
            timeout_sec=timeout_sec,
            model=self._config.intent_model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            think=self._config.think,
            max_tokens=self._config.intent_max_tokens,
            response_format=self._config.intent_schema,
        )
        if not raw_response:
            return {}

        parsed = _extract_json_object(raw_response)
        if not parsed:
            self._logger.warn('Intent extraction response was not valid JSON')
            return {}
        return parsed

    def _query_planner_completion_ack(self, completion_context: dict) -> str:
        if not self._config.enabled:
            return ''

        prompt = self._system_response_prompt(
            task_addendum=_PLANNER_COMPLETION_RESPONSE_ADDENDUM,
        )
        payload = json.dumps(
            {'planner_completion': completion_context},
            sort_keys=True,
            separators=(',', ':'),
        )
        raw_response = self._transport.query(
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': payload},
            ],
            timeout_sec=self._config.request_timeout_sec,
            model=self._config.model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            think=self._config.think,
            max_tokens=self._config.response_max_tokens,
            response_format=self._config.response_schema,
        )
        if not raw_response:
            return ''
        parsed = _extract_json_object(raw_response)
        if parsed:
            candidate = _ack_from_parsed_response(parsed)
            if candidate:
                return candidate
        return _extract_ack_text(raw_response) or str(raw_response).strip()

    def _query_execution_report_ack(self, report_context: dict) -> str:
        if not self._config.enabled:
            return ''

        prompt = self._system_response_prompt(
            task_addendum=_EXECUTION_REPORT_RESPONSE_ADDENDUM,
        )
        payload = json.dumps(
            {'execution_report': report_context},
            sort_keys=True,
            separators=(',', ':'),
        )
        raw_response = self._transport.query(
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': payload},
            ],
            timeout_sec=self._config.request_timeout_sec,
            model=self._config.model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            think=self._config.think,
            max_tokens=self._config.response_max_tokens,
            response_format=self._config.response_schema,
        )
        if not raw_response:
            return ''
        parsed = _extract_json_object(raw_response)
        if parsed:
            candidate = _ack_from_parsed_response(parsed)
            if candidate:
                return _postprocess_execution_report_ack(candidate, report_context)
        candidate = _extract_ack_text(raw_response) or str(raw_response).strip()
        return _postprocess_execution_report_ack(candidate, report_context)

    def _query_planner_dialogue_ack(self, dialogue_context: dict) -> str:
        if not self._config.enabled:
            return ''

        prompt = self._system_response_prompt(
            task_addendum=_PLANNER_DIALOGUE_RESPONSE_ADDENDUM,
        )
        payload = json.dumps(
            {'planner_dialogue': dialogue_context},
            sort_keys=True,
            separators=(',', ':'),
        )
        raw_response = self._transport.query(
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': payload},
            ],
            timeout_sec=self._config.request_timeout_sec,
            model=self._config.model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            think=self._config.think,
            max_tokens=self._config.response_max_tokens,
            response_format=self._config.response_schema,
        )
        if not raw_response:
            return ''
        parsed = _extract_json_object(raw_response)
        if parsed:
            candidate = _ack_from_parsed_response(parsed)
            if candidate:
                return candidate
        return _extract_ack_text(raw_response) or str(raw_response).strip()

    # -----------------------------------------------------------------------
    # Prompt-history helper methods
    # -----------------------------------------------------------------------

    def _system_response_prompt(self, *, task_addendum: str) -> str:
        return build_response_prompt(
            robot_name=self._config.robot_name,
            user_id='__system__',
            system_prompt=self._config.system_prompt,
            environment_description=self._config.environment_description,
            knowledge_snapshot='',
            response_prompt_addendum=_system_task_response_addendum(
                self._config.response_prompt_addendum,
                task_addendum,
            ),
            skill_catalog_text=self._skill_catalog_text,
            persona_prompt=self._persona_prompt,
            planner_mode_enabled=False,
        )

    def _inject_identity_reminder(self, history_messages: list[dict]) -> list[dict]:
        if self._config.identity_reminder_every_n_turns <= 0:
            return list(history_messages)
        if self._handled_requests <= 0:
            return list(history_messages)
        if self._handled_requests % self._config.identity_reminder_every_n_turns != 0:
            return list(history_messages)

        reminder = {
            'role': 'system',
            'content': (
                f'Reminder: You are {self._config.robot_name}. '
                'Keep your personality and stay concise for spoken responses.'
            ),
        }
        return list(history_messages) + [reminder]

    def _history_with_turn(
        self,
        history: list[str],
        user_text: str,
        assistant_text: str,
    ) -> list[str]:
        return messages_to_history(
            history_to_messages(
                history,
                max_history_messages=self._config.max_history_messages,
            )
            + [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': assistant_text},
            ],
            max_history_messages=self._config.max_history_messages,
        )

    def _history_with_assistant_text(self, history: list[str], assistant_text: str) -> list[str]:
        messages = history_to_messages(
            history,
            max_history_messages=self._config.max_history_messages,
        )
        messages.append({'role': 'assistant', 'content': assistant_text})
        return messages_to_history(
            messages,
            max_history_messages=self._config.max_history_messages,
        )

    @staticmethod
    def _build_result(
        *,
        success: bool,
        verbal_ack: str,
        updated_history: list[str],
        intent: str,
        intent_source: str,
        intent_confidence: float,
        user_intent: dict,
        route: str,
    ) -> TurnExecutionResult:
        verbal_ack = _limit_verbal_ack(verbal_ack)
        return TurnExecutionResult(
            success=success,
            verbal_ack=verbal_ack,
            updated_history=updated_history,
            intent=intent,
            intent_source=intent_source,
            intent_confidence=intent_confidence,
            user_intent=user_intent,
            route=route,
        )

    @staticmethod
    def _publish_progress(callback, status: str, progress: float) -> None:
        if callable(callback):
            callback(status, progress)

    @staticmethod
    def _trace(trace, turn_id: str, stage: str, message: str) -> None:
        if callable(trace):
            trace(turn_id, stage, message)

    @staticmethod
    def _trace_route_decision(
        trace,
        turn_id: str,
        *,
        llm_route: str,
        final_route: str,
        intent_source: str,
    ) -> None:
        """Emit an observability-only record of LLM route vs final route.

        This does not change any routing decision; it makes the override rate
        between the model's own route and the deterministically resolved route
        measurable from the turn trace.
        """
        overridden = bool(llm_route) and llm_route != final_route
        DialogueTurnEngine._trace(
            trace,
            turn_id,
            'ROUTE_DECISION',
            'llm_route=%s final_route=%s overridden=%s reason=%s'
            % (llm_route or '-', final_route, str(overridden).lower(), intent_source),
        )

    @staticmethod
    def _preview_text(text: str, max_len: int = 72) -> str:
        clean = ' '.join(str(text).split())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3] + '...'


# ---------------------------------------------------------------------------
# Module-local logging helper
# ---------------------------------------------------------------------------


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)
