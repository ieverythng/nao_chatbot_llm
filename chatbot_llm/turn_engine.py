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
from kb_skills.intent_labels import KB_QUERY_INTENTS, KB_QUERY_VISIBLE_OBJECTS


_DIALOGUE_ROUTE = 'dialogue'
_KNOWLEDGE_QUERY_ROUTE = 'knowledge_query'
_EXECUTION_ROUTE = 'execution'
_SUPPORTED_ROUTES = {
    _DIALOGUE_ROUTE,
    _KNOWLEDGE_QUERY_ROUTE,
    _EXECUTION_ROUTE,
}
_DIALOGUE_INTENTS = {'greet', 'identity', 'wellbeing', 'help'}
_EXECUTION_HINT_MARKERS = (
    ' and then ',
    ' then ',
    ' after that ',
    ' before that ',
    ' while also ',
    ' also ',
    ' stand',
    ' sit',
    ' kneel',
    ' look ',
    ' move ',
    ' scan ',
    ' search ',
    ' find ',
    ' locate ',
    ' detect ',
    ' turn ',
    ' head ',
    ' bring ',
    ' grab ',
    ' pick ',
    ' place ',
    ' guide ',
    ' navigate ',
    ' walk ',
    ' wave ',
    ' add ',
    ' remember ',
    ' store ',
    ' save ',
    ' revise ',
    ' update ',
    ' change ',
    ' correct ',
    ' remove ',
    ' delete ',
    ' forget ',
    ' knowledge base',
    ' kb ',
    ' rdf:type',
    ' dbp:',
    ' oro:',
)
_REFLECTIVE_EXECUTION_QUESTION_MARKERS = (
    'what did you',
    'what have you',
    'what were you able to',
    'how many',
    'which direction',
    'which directions',
    'did you',
    'have you',
)
_SOCIAL_TURN_MARKERS = {
    'hi',
    'hello',
    'hey',
    'hey there',
    'good morning',
    'good afternoon',
    'good evening',
    'how are you',
    'thanks',
    'thank you',
}


_PLANNER_COMPLETION_KEYS = (
    'goal_text',
    'result_summary',
    'text_hint',
    'requested_intents',
    'result_payload',
)
_PLANNER_DIALOGUE_KEYS = (
    'act',
    'reason',
    'text_hint',
    'slots_needed',
    'context',
    'completion_context',
)
_EXECUTION_REPORT_KEYS = (
    'goal_text',
    'requested_intents',
    'dialogue_context',
    'scene_targets',
    'grounded_context',
    'requested_summary',
    'report_role',
    'future_steps',
    'steps',
    'latest_result_summary',
    'latest_result_payload',
)
_SYSTEM_TASK_RESPONSE_MODE_ADDENDUM = """
System wording mode:
- This turn is a closed internal wording task, not a new user request.
- Keep the spoken answer short, natural for text-to-speech, and consistent with
  the configured robot identity.
- Do not apply route admission, planner handoff, or execution-acknowledgement
  policy here. This prompt exists only to word an already-classified system
  payload.
""".strip()
_PLANNER_COMPLETION_RESPONSE_ADDENDUM = """
Planner completion wording task:
- You are wording one spoken completion sentence after robot task execution.
- Use only facts from the planner_completion JSON payload.
- If facts are missing, briefly say no confirmed result was available.
- Return only the normal response JSON with verbal_ack as the TTS-ready sentence.
- Do not mention JSON, planner internals, routes, or policies.
""".strip()
_PLANNER_DIALOGUE_RESPONSE_ADDENDUM = """
Planner dialogue wording task:
- You are wording one spoken robot sentence for a planner dialogue act.
- Use only facts from the planner_dialogue JSON payload.
- Keep it concise and first-person as the robot.
- Return only the normal response JSON with verbal_ack as the TTS-ready sentence.
- Do not mention JSON, planner internals, routes, or policies.
""".strip()
_EXECUTION_REPORT_RESPONSE_ADDENDUM = """
Execution report wording task:
- You are wording a spoken report for robot task execution.
- Use only facts from the execution_report JSON payload.
- If report_role is "intermediate", report only the latest completed action or
  observation needed at this point. Do not summarize future or not-yet-executed
  steps, and do not claim the whole goal is finished.
- For intermediate reports, use latest_result_summary and latest_result_payload
  as the primary evidence. Older steps are context only and must not replace the
  latest completed action.
- Intermediate reports must not preview the next movement. Do not say "next
  object", "another object", "I am now walking", or equivalent continuation
  wording. Report only the arrival or observation that just completed.
- If report_role is "final" or absent, summarize the whole executed step chain,
  not only the last step.
- Synthesize related routine steps into natural language when producing a final
  report; do not recite each internal motion or execution result as a separate
  ledger sentence.
- Use goal_text and dialogue_context to produce a coherent continuation and to
  decide which outcomes matter to the user.
- Do not answer with "I finished:" followed by the user's request. Synthesize
  what was actually done from steps, latest_result_summary, and grounded_context.
- Use grounded_context to translate entity handles into meaningful user-facing
  labels and relations when the facts are present.
- When a place_object or bring_object result uses a person or Human as the
  destination, describe it as delivering or handing the object to that person,
  not placing the object on that person. Keep "on" only for support surfaces
  such as tables, shelves, counters, or named places.
- Treat requested_summary as evidence or a wording hint, not as text that must
  be repeated verbatim.
- Use step results as the authority for factual execution claims, respecting
  success or failure flags and the status of the previous steps in the chain.
- Mention relevant observations from scan/perception results.
- Use one report for this report_result call. Preserve useful dialogue context,
  avoid repeated information, and match the report_role.
- Return only the normal response JSON with verbal_ack as the TTS-ready report.
- Do not mention JSON, planner internals, report_result, routes, or policies.
""".strip()


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


_MAX_VERBAL_ACK_CHARS = 900


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
            )
            if route == _EXECUTION_ROUTE:
                verbal_ack = _sanitize_execution_ack(verbal_ack)
            self._trace(
                trace,
                turn_id,
                'ROUTE_RESOLVED',
                'route=%s intent=%s source=%s confidence=%.2f'
                % (route, resolved_intent or '-', intent_source, intent_confidence),
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
        route, resolved_intent, user_intent, intent_source = self._lock_intent_first_route(
            user_text=user_text,
            resolved_intent=resolved_intent,
            user_intent=user_intent,
            intent_source=intent_source,
        )
        self._trace(
            trace,
            turn_id,
            'ROUTE_LOCKED',
            'mode=intent_first route=%s intent=%s source=%s confidence=%.2f'
            % (route, resolved_intent or '-', intent_source, intent_confidence),
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
            verbal_ack = (
                'Okay, I will try that now.'
                if route == _EXECUTION_ROUTE
                else self._config.fallback_response
            )
        if route == _EXECUTION_ROUTE:
            verbal_ack = _sanitize_execution_ack(verbal_ack)

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

        route = self._route_for_intent(
            str(user_intent.get('type', '')).strip() or resolved_intent
        )
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
    ) -> tuple[str, str, str, float, dict]:
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
        # Keep greeting-like turns dialogue-first unless the user clearly asked
        # for an execution action. This avoids planner handoff on short social openers.
        if _is_greeting_intent(resolved_intent, user_intent) and not _looks_like_execution_text(
            user_text
        ):
            inferred_route = _DIALOGUE_ROUTE
        if _is_social_turn(user_text) and not _looks_like_execution_text(user_text):
            inferred_route = _DIALOGUE_ROUTE
        if _is_capability_query(user_text):
            inferred_route = _DIALOGUE_ROUTE
        if _is_personal_preference_question(user_text):
            inferred_route = _DIALOGUE_ROUTE
            resolved_intent = ''
            user_intent = {}
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
            and not _is_reflective_execution_question(user_text)
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
            if _is_personal_preference_question(user_text)
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
            return {'verbal_ack': self._config.fallback_response, '_route_missing': True}
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
    def _preview_text(text: str, max_len: int = 72) -> str:
        clean = ' '.join(str(text).split())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3] + '...'


# ---------------------------------------------------------------------------
# Module-local JSON coercion helpers
# ---------------------------------------------------------------------------

def _join_prompt_addenda(*parts: str) -> str:
    return '\n\n'.join(str(part).strip() for part in parts if str(part).strip())


def _system_task_response_addendum(base_addendum: str, task_addendum: str) -> str:
    return _join_prompt_addenda(
        _SYSTEM_TASK_RESPONSE_MODE_ADDENDUM,
        _extract_system_task_response_rules(base_addendum),
        task_addendum,
    )


def _locked_route_prompt_block(context: dict) -> str:
    route = _normalize_route(context.get('route', '')) or _DIALOGUE_ROUTE
    payload = {
        'route': route,
        'resolved_intent': str(context.get('resolved_intent', '')).strip(),
        'user_intent': _coerce_user_intent(context.get('user_intent', {})),
    }
    return _join_prompt_addenda(
        'Locked route context:',
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')),
        (
            'The route above was selected by the intent stage before this wording '
            'request. Do not change it. Return the same route in the response JSON. '
            'If route is dialogue or knowledge_query, do not promise physical action. '
            'If route is execution, acknowledge the request in future tense only and '
            'do not claim completion or observations.'
        ),
    )


def _extract_system_task_response_rules(base_addendum: str) -> str:
    sections = _response_addendum_sections(base_addendum)
    parts: list[str] = []

    intro_lines = [
        line.strip()
        for line in sections.get('__intro__', '').splitlines()
        if 'text-to-speech' in line.lower()
    ]
    if intro_lines:
        parts.append('\n'.join(intro_lines))

    perception = sections.get('Perception and knowledge policy', '').strip()
    if perception:
        parts.append('Perception and knowledge policy:\n%s' % perception)

    shared_style = _shared_response_style_rules(sections.get('Response style', ''))
    if shared_style:
        parts.append('Response style:\n%s' % shared_style)

    extracted = _join_prompt_addenda(*parts)
    if extracted:
        return extracted
    return str(base_addendum or '').strip()


def _response_addendum_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {'__intro__': []}
    current = '__intro__'
    for raw_line in str(text or '').splitlines():
        stripped = raw_line.strip()
        if stripped.endswith(':') and stripped and not stripped.startswith('- '):
            current = stripped[:-1]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line.rstrip())
    return {
        key: '\n'.join(value).strip()
        for key, value in sections.items()
        if any(line.strip() for line in value)
    }


def _shared_response_style_rules(section_text: str) -> str:
    shared_lines: list[str] = []
    skip_route_block = False
    for raw_line in str(section_text or '').splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if not skip_route_block:
                shared_lines.append('')
            continue
        if stripped.startswith('- If route="'):
            skip_route_block = True
            continue
        if skip_route_block:
            if stripped.startswith('- ') and not raw_line.startswith('  '):
                skip_route_block = False
            else:
                continue
        if not skip_route_block:
            shared_lines.append(raw_line.rstrip())
    return '\n'.join(shared_lines).strip()


def _parse_json_dict(payload: str) -> dict:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, str):
        return _parse_json_dict(parsed)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _extract_json_object(payload: str) -> dict:
    parsed = _parse_json_dict(payload)
    if parsed:
        return parsed

    decoder = json.JSONDecoder()
    for start in range(len(payload)):
        if payload[start] != '{':
            continue
        try:
            maybe_obj, _ = decoder.raw_decode(payload[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(maybe_obj, dict):
            return maybe_obj
    return {}


def _extract_ack_text(payload: str, _depth: int = 0) -> str:
    if _depth > 3:
        return ''
    parsed = _extract_json_object(payload)
    if parsed:
        for key in ('verbal_ack', 'ack_text'):
            text = str(parsed.get(key, '')).strip()
            if text:
                return text
        for value in parsed.values():
            if isinstance(value, dict):
                nested_text = _extract_ack_text(json.dumps(value), _depth + 1)
                if nested_text:
                    return nested_text
            elif isinstance(value, str):
                nested_text = _extract_ack_text(value, _depth + 1)
                if nested_text:
                    return nested_text
        response_text = str(parsed.get('response', '')).strip()
        if response_text:
            nested_text = _extract_ack_text(response_text, _depth + 1)
            if nested_text:
                return nested_text
            return response_text
        return ''

    # Be permissive when the model drifts into "almost JSON" formatting.
    for key in ('verbal_ack', 'ack_text', 'response'):
        patterns = (
            rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf"'{key}'\s*:\s*'((?:[^'\\]|\\.)*)'",
            rf'\b{key}\b\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf"\b{key}\b\s*:\s*'((?:[^'\\]|\\.)*)'",
        )
        for pattern in patterns:
            match = re.search(pattern, payload, re.DOTALL)
            if not match:
                continue
            text = bytes(match.group(1), 'utf-8').decode('unicode_escape').strip()
            if not text:
                continue
            if key == 'response':
                nested_text = _extract_ack_text(text, _depth + 1)
                if nested_text:
                    return nested_text
            return text
    return ''


def _ack_from_parsed_response(parsed: dict) -> str:
    for key in ('verbal_ack', 'ack_text'):
        text = str(parsed.get(key, '')).strip()
        if text:
            return text
    response_text = str(parsed.get('response', '')).strip()
    if response_text:
        nested_text = _extract_ack_text(response_text)
        return nested_text or response_text
    return ''


def _limit_verbal_ack(verbal_ack: str) -> str:
    text = ' '.join(str(verbal_ack or '').strip().split())
    if len(text) <= _MAX_VERBAL_ACK_CHARS:
        return text
    boundary = max(text.rfind('.', 0, _MAX_VERBAL_ACK_CHARS), text.rfind('?', 0, _MAX_VERBAL_ACK_CHARS))
    if boundary < int(_MAX_VERBAL_ACK_CHARS * 0.55):
        boundary = text.rfind(' ', 0, _MAX_VERBAL_ACK_CHARS)
    if boundary < int(_MAX_VERBAL_ACK_CHARS * 0.55):
        boundary = _MAX_VERBAL_ACK_CHARS
    summary = text[:boundary].strip(' ,;:.')
    return '%s. I can continue if you want more detail.' % summary


def _looks_like_json_payload(payload: str) -> bool:
    clean_payload = str(payload or '').strip()
    return clean_payload.startswith(('{', '"{', '```json', '```'))


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)


def _coerce_user_intent(user_intent) -> dict:
    if isinstance(user_intent, dict):
        cleaned = {}
        for key in (
            'type',
            'object',
            'recipient',
            'input',
            'goal',
            'goal_text',
            'task',
            'ack_text',
            'request_kind',
            'goal_id',
            'parent_goal_id',
            'supersedes_goal_id',
            'dialogue_turn_id',
        ):
            value = str(user_intent.get(key, '')).strip()
            if value:
                cleaned[key] = value
        scene_targets = user_intent.get('scene_targets')
        if isinstance(scene_targets, str):
            parsed_targets = [item.strip() for item in scene_targets.split(',') if item.strip()]
            if parsed_targets:
                cleaned['scene_targets'] = parsed_targets
        elif isinstance(scene_targets, (list, tuple)):
            parsed_targets = [str(item).strip() for item in scene_targets if str(item).strip()]
            if parsed_targets:
                cleaned['scene_targets'] = parsed_targets

        intent_sequence = user_intent.get('intent_sequence')
        if isinstance(intent_sequence, str):
            parsed_intents = [
                item.strip() for item in intent_sequence.split(',') if item.strip()
            ]
            if parsed_intents:
                cleaned['intent_sequence'] = parsed_intents
        elif isinstance(intent_sequence, (list, tuple)):
            parsed_intents = [
                str(item).strip() for item in intent_sequence if str(item).strip()
            ]
            if parsed_intents:
                cleaned['intent_sequence'] = parsed_intents

        return cleaned
    if isinstance(user_intent, str) and user_intent.strip():
        return {'type': user_intent.strip()}
    return {}


def _normalize_route(value) -> str:
    clean_value = str(value or '').strip().lower()
    if clean_value in _SUPPORTED_ROUTES:
        return clean_value
    return ''


def _looks_like_execution_text(user_text: str) -> bool:
    if _is_information_only_action_word_question(user_text):
        return False
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(marker in lowered for marker in _EXECUTION_HINT_MARKERS)


def _rules_execution_intent_allowed(user_text: str) -> bool:
    return _looks_like_execution_text(user_text) and not _is_information_only_action_word_question(
        user_text
    )


def _is_repeat_action_request(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized in {
        'again',
        'do it again',
        'do that again',
        'repeat it',
        'repeat that',
        'repeat the action',
        'one more time',
    }


def _is_information_only_action_word_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    asks_information = (
        '?' in clean
        or normalized.startswith(('what ', 'how ', 'why ', 'explain ', 'define '))
        or normalized.startswith('tell me about ')
    )
    if not asks_information:
        return False
    if any(
        marker in clean
        for marker in (
            'wave at',
            'wave to',
            'please wave',
            'can you wave',
            'could you wave',
            'will you wave',
            'would you wave',
            'look at',
            'navigate to',
            'walk to',
            'move to',
            'go to',
        )
    ):
        return False
    return any(
        marker in normalized
        for marker in (
            'wave particle',
            'wave equation',
            'particle equation',
            'wave duality',
            'physics',
            'wavelength',
        )
    ) or normalized.startswith(('what is wave', 'what are waves', 'explain wave'))


def _route_is_contradictory(*, user_text: str, verbal_ack: str, route: str) -> bool:
    if route == _EXECUTION_ROUTE:
        return _is_non_immediate_action_discussion(user_text)
    if route in {_DIALOGUE_ROUTE, _KNOWLEDGE_QUERY_ROUTE}:
        return (
            _looks_like_execution_text(user_text)
            and _ack_implies_execution(verbal_ack)
            and not _is_non_immediate_action_discussion(user_text)
            and not _is_reflective_execution_question(user_text)
            and not _is_capability_query(user_text)
        )
    return False


def _repair_response_route(*, user_text: str, verbal_ack: str, route: str) -> str:
    if _is_reflective_execution_question(user_text):
        return _DIALOGUE_ROUTE
    if _is_capability_query(user_text):
        return _DIALOGUE_ROUTE
    if _is_personal_preference_question(user_text):
        return _DIALOGUE_ROUTE
    if _is_social_turn(user_text) and not _looks_like_execution_text(user_text):
        return _DIALOGUE_ROUTE
    if _is_non_immediate_action_discussion(user_text):
        return _DIALOGUE_ROUTE
    if _looks_like_execution_text(user_text) and _ack_implies_execution(verbal_ack):
        return _EXECUTION_ROUTE
    if route in _SUPPORTED_ROUTES:
        return route
    return _DIALOGUE_ROUTE


def _is_non_immediate_action_discussion(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean or not _looks_like_execution_text(clean):
        return False
    if not any(marker in clean for marker in ('could', 'would', 'can we', 'could we')):
        return False
    return any(
        marker in clean
        for marker in (
            'later',
            'some other time',
            'at some point',
            'afterwards',
            'eventually',
        )
    )


def _is_social_turn(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch for ch in clean if ch.isalnum() or ch.isspace()).strip()
    if normalized in _SOCIAL_TURN_MARKERS:
        return True
    token_count = len([token for token in normalized.split(' ') if token])
    if token_count <= 2 and normalized in {'hi', 'hello', 'hey', 'thanks'}:
        return True
    return False


def _is_capability_query(user_text: str) -> bool:
    """Return whether the user is asking what the robot can do, not requesting it."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch for ch in clean if ch.isalnum() or ch.isspace()).strip()
    capability_markers = (
        'what can you do',
        'what are you able to do',
        'what capabilities do you have',
        'what are your capabilities',
        'tell me what you can do',
        'what skills do you have',
        'which skills do you have',
        'what fake skills do you have',
        'do you have fake skills',
        'do you have any fake skills',
        'tell me about your skills',
    )
    return any(marker in normalized for marker in capability_markers)


def _is_personal_preference_question(user_text: str) -> bool:
    """Return whether the user is asking for conversation, not robot execution."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    if not (
        '?' in clean
        or normalized.startswith(('what ', 'which ', 'who ', 'do you ', 'tell me '))
    ):
        return False
    preference_markers = (
        'your favorite',
        'your favourite',
        'do you like',
        'what do you like',
        'which do you like',
        'what do you prefer',
        'which do you prefer',
        'your preference',
        'your opinion',
    )
    return any(marker in normalized for marker in preference_markers)


def _is_reflective_execution_question(user_text: str) -> bool:
    """Return whether the user is asking about prior execution evidence."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    if not any(marker in clean for marker in _REFLECTIVE_EXECUTION_QUESTION_MARKERS):
        return False
    return _looks_like_execution_text(clean)


def _dialogue_user_intent(user_intent: dict) -> dict:
    if not user_intent:
        return {}
    normalized = dict(user_intent)
    normalized['type'] = 'fallback'
    for key in ('goal', 'goal_text', 'intent_sequence'):
        normalized.pop(key, None)
    return normalized


def _ack_implies_execution(verbal_ack: str) -> bool:
    clean_ack = ' %s ' % ' '.join(str(verbal_ack or '').strip().lower().split())
    if not clean_ack.strip():
        return False
    commitment_markers = (
        " i will ",
        " i'll ",
        ' let me ',
        ' i can ',
    )
    if not any(marker in clean_ack for marker in commitment_markers):
        return False
    return _looks_like_execution_text(clean_ack)


def _infer_kb_query_intent_from_text(user_text: str) -> str:
    inferred = normalize_intent(detect_intent(user_text), default='')
    if inferred in KB_QUERY_INTENTS:
        return inferred
    if _looks_like_visible_scene_question(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    if _looks_like_non_mutating_kb_question(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    # Catch specific-object attribute queries like "What is the name/color of X?"
    # These are KB lookups even when detect_intent returns 'fallback'.
    if _looks_like_object_attribute_query(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    return ''


def _looks_like_visible_scene_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized.startswith(
        (
            'what can you see',
            'what do you see',
            'what are you seeing',
            'who can you see',
            'who do you see',
            'what objects can you see',
            'what objects do you see',
            'how many objects can you see',
            'how many objects do you see',
        )
    )


def _looks_like_non_mutating_kb_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized.startswith(
        (
            'what do you remember',
            'what can you remember',
            'do you remember',
            'what do you know',
            'what is',
            'what are',
            'which facts',
            'tell me what',
        )
    )


def _looks_like_object_attribute_query(user_text: str) -> bool:
    """Detect queries about object attributes: name, color, type, position."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    # Must ask about an attribute
    attr_markers = ('name', 'color', 'type', 'position', 'location', 'id')
    has_attr = any(m in clean for m in attr_markers)
    # Must reference a specific object (not general "what do you see")
    obj_patterns = [
        r'of\s+\w+',           # "of the cup", "of X"
        r'the\s+\w+',          # "the probe cup"
        r'is\s+the\s+',        # "is the name"
    ]
    has_object = any(re.search(p, clean) for p in obj_patterns)
    return has_attr and has_object





def _is_greeting_intent(resolved_intent: str, user_intent: dict) -> bool:
    if str(resolved_intent or '').strip().lower() == 'greet':
        return True
    return str(user_intent.get('type', '')).strip().lower() == 'greet'


def _has_explicit_perception_action_request(user_text: str) -> bool:
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(
        marker in lowered
        for marker in (
            ' scan ',
            ' look around ',
            ' inspect the scene ',
            ' inspect scene ',
            ' search the area ',
            ' search around ',
            ' check the area ',
            ' perform a scan ',
        )
    )


def _sanitize_execution_ack(verbal_ack: str) -> str:
    """Keep execution acknowledgements from claiming the result before execution."""
    clean_ack = str(verbal_ack or '').strip()
    if not clean_ack:
        return 'Okay, I will do that.'
    if clean_ack == 'fallback':
        return clean_ack

    clean_ack = re.sub(
        r'\s*\((?:[^)]*\b(?:perform|performs|performed|move|moves|moved|'
        r'scan|scans|scanned|look|looks|looked|turn|turns|turned|wave|waves|'
        r'waved|navigate|navigates|navigated)[^)]*)\)',
        '',
        clean_ack,
        flags=re.IGNORECASE,
    ).strip()

    lowered = clean_ack.lower()
    if re.search(
        r"\b(i cannot|i can't|i could not|i couldn't|i do not have|i don't have|unable to)\b",
        lowered,
    ):
        return 'Okay, I will try that now.'

    result_markers = (
        'i can currently',
        'i currently',
        'i can see',
        'i see',
        'i have scanned',
        'i found',
        'i have found',
        'i detected',
        'i have detected',
        'i observed',
        'i have observed',
        'i am now',
        "i'm now",
        'i have arrived',
        "i've arrived",
        'i arrived',
        'i performed',
        'i have moved',
        'i have completed',
        'i completed',
        'i scanned',
        'the scan',
    )
    split_at = len(clean_ack)
    for marker in result_markers:
        match = re.search(
            r'(^|[.!?]\s+)%s\b' % re.escape(marker.lower()),
            lowered,
        )
        if match:
            split_at = min(split_at, match.start(1))
    clean_ack = clean_ack[:split_at].strip()
    clean_ack = re.sub(
        r'(?:\s*(?:,|;|:)?\s*\b(?:and|but|so)\b)+$',
        '',
        clean_ack,
        flags=re.IGNORECASE,
    ).strip()

    if not clean_ack:
        return 'Okay, I will do that.'
    if clean_ack[-1] not in '.!?':
        clean_ack += '.'
    return clean_ack


def _extract_planner_completion_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('planner_completion', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    normalized = {
        'goal_text': str(context.get('goal_text', '')).strip(),
        'result_summary': str(context.get('result_summary', '')).strip(),
        'text_hint': str(context.get('text_hint', '')).strip(),
        'requested_intents': [
            str(item).strip()
            for item in context.get('requested_intents', [])
            if str(item).strip()
        ]
        if isinstance(context.get('requested_intents', []), list)
        else [],
        'result_payload': context.get('result_payload', {})
        if isinstance(context.get('result_payload', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _PLANNER_COMPLETION_KEYS):
        return {}
    return normalized


def _extract_execution_report_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('execution_report', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    steps = context.get('steps', [])
    normalized_steps = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            normalized_steps.append(
                {
                    'id': str(step.get('id', '')).strip(),
                    'name': str(step.get('name', '')).strip(),
                    'type': str(step.get('type', '')).strip(),
                    'args': step.get('args', {})
                    if isinstance(step.get('args', {}), dict)
                    else {},
                    'status': str(step.get('status', '')).strip(),
                    'reason': str(step.get('reason', '')).strip(),
                    'result_summary': str(step.get('result_summary', '')).strip(),
                    'result_payload': step.get('result_payload', {})
                    if isinstance(step.get('result_payload', {}), dict)
                    else {},
                }
            )
    future_steps = context.get('future_steps', [])
    normalized_future_steps = []
    if isinstance(future_steps, list):
        for step in future_steps:
            if not isinstance(step, dict):
                continue
            normalized_future_steps.append(
                {
                    'id': str(step.get('id', '')).strip(),
                    'name': str(step.get('name', '')).strip(),
                    'type': str(step.get('type', '')).strip(),
                    'args': step.get('args', {})
                    if isinstance(step.get('args', {}), dict)
                    else {},
                    'requires': [
                        str(item).strip()
                        for item in step.get('requires', [])
                        if str(item).strip()
                    ]
                    if isinstance(step.get('requires', []), list)
                    else [],
                }
            )
    normalized = {
        'goal_text': str(context.get('goal_text', '')).strip(),
        'requested_intents': [
            str(item).strip()
            for item in context.get('requested_intents', [])
            if str(item).strip()
        ]
        if isinstance(context.get('requested_intents', []), list)
        else [],
        'dialogue_context': [
            str(item).strip()
            for item in context.get('dialogue_context', [])
            if str(item).strip()
        ]
        if isinstance(context.get('dialogue_context', []), list)
        else [],
        'scene_targets': [
            str(item).strip()
            for item in context.get('scene_targets', [])
            if str(item).strip()
        ]
        if isinstance(context.get('scene_targets', []), list)
        else [],
        'grounded_context': context.get('grounded_context', {})
        if isinstance(context.get('grounded_context', {}), dict)
        else {},
        'requested_summary': str(context.get('requested_summary', '')).strip(),
        'report_role': str(context.get('report_role', '')).strip(),
        'future_steps': normalized_future_steps,
        'steps': normalized_steps,
        'latest_result_summary': str(context.get('latest_result_summary', '')).strip(),
        'latest_result_payload': context.get('latest_result_payload', {})
        if isinstance(context.get('latest_result_payload', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _EXECUTION_REPORT_KEYS):
        return {}
    return normalized


def _extract_planner_dialogue_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('planner_dialogue', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    normalized = {
        'act': str(context.get('act', '')).strip().lower(),
        'goal_id': str(context.get('goal_id', '')).strip(),
        'plan_id': str(context.get('plan_id', '')).strip(),
        'plan_version': int(coerce_float(context.get('plan_version', 0))),
        'reason': str(context.get('reason', '')).strip(),
        'text_hint': str(context.get('text_hint', '')).strip(),
        'await_user_response': bool(context.get('await_user_response', False)),
        'slots_needed': [
            str(item).strip()
            for item in context.get('slots_needed', [])
            if str(item).strip()
        ]
        if isinstance(context.get('slots_needed', []), list)
        else [],
        'context': context.get('context', {})
        if isinstance(context.get('context', {}), dict)
        else {},
        'completion_context': context.get('completion_context', {})
        if isinstance(context.get('completion_context', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _PLANNER_DIALOGUE_KEYS):
        return {}
    return normalized


def _fallback_planner_completion_ack(completion_context: dict) -> str:
    text_hint = str(completion_context.get('text_hint', '')).strip()
    if text_hint:
        return text_hint
    result_summary = str(completion_context.get('result_summary', '')).strip()
    if result_summary:
        return result_summary
    result_payload = completion_context.get('result_payload', {})
    if isinstance(result_payload, dict):
        summary_text = str(result_payload.get('summary_text', '')).strip()
        if summary_text:
            return summary_text
    goal_text = str(completion_context.get('goal_text', '')).strip()
    if goal_text:
        return 'I completed the requested task.'
    return ''


def _friendly_step_label(step_name: str) -> str:
    clean = str(step_name or '').strip().lower()
    labels = {
        'look_at': 'looked at the target',
        'navigate_to': 'navigated to the target',
        'walk_to': 'walked to the target',
        'wave_greet': 'waved',
        'perform_motion': 'performed the motion',
        'scan': 'looked around',
        'inspect_scene': 'inspected the scene',
        'find_object': 'looked for the object',
        'pick_object': 'picked up the object',
        'place_object': 'placed the object',
        'bring_object': 'brought the object',
    }
    if clean in labels:
        return labels[clean]
    return clean.replace('_', ' ') if clean else 'completed the step'


def _fallback_execution_report_ack(report_context: dict) -> str:
    successful_steps = [
        step for step in report_context.get('steps', [])
        if isinstance(step, dict) and str(step.get('status', '')).strip().lower() == 'succeeded'
    ]
    latest_summary = str(report_context.get('latest_result_summary', '')).strip()
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate':
        if latest_summary:
            return latest_summary
        if successful_steps:
            latest_step_summary = str(successful_steps[-1].get('result_summary', '')).strip()
            if latest_step_summary:
                return latest_step_summary

    if successful_steps:
        navigation_report = _ordered_navigation_report(successful_steps, report_context)
        if navigation_report:
            return navigation_report
        summaries = [
            str(step.get('result_summary', '')).strip()
            for step in successful_steps
            if str(step.get('result_summary', '')).strip()
            and not _is_placeholder_report_summary(str(step.get('result_summary', '')).strip())
        ]
        if summaries:
            return ' '.join(summaries[-2:])

    if latest_summary:
        return latest_summary
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        summary_text = str(latest_payload.get('summary_text', '')).strip()
        if summary_text:
            return summary_text
    step_names = [
        _friendly_step_label(str(step.get('name', '')).strip())
        for step in successful_steps[-2:]
        if isinstance(step, dict) and str(step.get('name', '')).strip()
    ]
    if step_names:
        if len(step_names) == 1:
            return 'I %s.' % step_names[0]
        return 'I %s and then %s.' % (step_names[0], step_names[1])
    if str(report_context.get('goal_text', '')).strip():
        return 'I completed the requested task.'
    return ''


def _postprocess_execution_report_ack(candidate: str, report_context: dict) -> str:
    """Reject report wording that duplicates or promises a future report."""
    clean = str(candidate or '').strip()
    if not clean:
        return ''
    lowered = clean.lower()
    if any(
        marker in lowered
        for marker in (
            'i finished:',
            'will report back',
            'will report on',
            'will let you know',
            'will tell you',
            'can report the current scene summary',
            'can report what i observed',
            'can report what i saw',
            'can report it',
            'going to report',
            "i'll report",
            "i'll let you know",
        )
    ):
        return ''
    if _is_placeholder_report_summary(clean):
        return ''
    if str(report_context.get('report_role', '')).strip().lower() == 'intermediate' and (
        'next object' in lowered
        or 'another object' in lowered
        or 'i am now walking' in lowered
        or "i'm now walking" in lowered
        or 'ready to move' in lowered
    ):
        return ''
    if _uses_stale_intermediate_arrival(clean, report_context):
        return ''
    clean = _rewrite_person_delivery_surface_phrase(clean, report_context)

    sentences = _split_sentences(clean)
    if len(sentences) <= 1:
        return clean
    successful_steps = _successful_non_report_steps(report_context)
    if len(successful_steps) == 1 and any(_is_generic_completion_sentence(item) for item in sentences):
        return ''
    return clean


def _uses_stale_intermediate_arrival(candidate: str, report_context: dict) -> bool:
    if str(report_context.get('report_role', '')).strip().lower() != 'intermediate':
        return False
    lowered = str(candidate or '').strip().lower()
    if 'arrived at' not in lowered and 'arrived to' not in lowered:
        return False
    navigation_steps = [
        step for step in _successful_non_report_steps(report_context)
        if str(step.get('name', '')).strip().lower() in {'navigate_to', 'walk_to'}
    ]
    if len(navigation_steps) <= 1:
        return False
    if 'first object' in lowered:
        return True
    latest_target = _latest_step_target(navigation_steps[-1], report_context)
    if not latest_target:
        return False
    for step in navigation_steps[:-1]:
        target = _latest_step_target(step, report_context)
        if target and target != latest_target and target.lower() in lowered:
            return True
    return False


def _rewrite_person_delivery_surface_phrase(candidate: str, report_context: dict) -> str:
    person_labels = _person_delivery_destination_labels(report_context)
    if not person_labels:
        return candidate
    rewritten = str(candidate or '').strip()
    for label in sorted(person_labels, key=len, reverse=True):
        escaped = re.escape(label)
        rewritten = re.sub(
            r'\b(?:placed|put)\s+(?P<object>[^.?!;]+?)\s+(?:on|onto)\s+(?P<label>%s)\b'
            % escaped,
            lambda match: (
                'delivered %s to %s'
                % (match.group('object').strip(), match.group('label').strip())
            ),
            rewritten,
            flags=re.IGNORECASE,
        )
    return rewritten


def _person_delivery_destination_labels(report_context: dict) -> set[str]:
    entity_labels = _grounded_person_labels_by_id(report_context)
    labels: set[str] = set()
    for step in _successful_non_report_steps(report_context):
        if str(step.get('name', '')).strip().lower() != 'place_object':
            continue
        destination_id = _step_destination_id(step)
        if not destination_id:
            continue
        for label in entity_labels.get(destination_id, ()):
            clean = str(label or '').strip()
            if clean:
                labels.add(clean)
    return labels


def _grounded_person_labels_by_id(report_context: dict) -> dict[str, set[str]]:
    grounded_context = report_context.get('grounded_context', {})
    if not isinstance(grounded_context, dict):
        return {}
    labels_by_id: dict[str, set[str]] = {}
    for entity in grounded_context.get('entities', []):
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get('id', '')).strip()
        if not entity_id or not _grounded_entity_is_person(entity):
            continue
        labels = {entity_id}
        for key in ('label', 'name', 'display_name'):
            value = str(entity.get(key, '')).strip()
            if value:
                labels.add(value)
        for relation in entity.get('relations', []):
            if not isinstance(relation, dict):
                continue
            predicate = str(relation.get('predicate', '')).strip().lower()
            if predicate.endswith('name') or predicate in {'name', 'label'}:
                value = str(relation.get('object', '')).strip()
                if value:
                    labels.add(value)
        labels_by_id[entity_id] = labels
    return labels_by_id


def _grounded_entity_is_person(entity: dict) -> bool:
    kind = str(entity.get('kind', '')).strip().lower()
    entity_class = str(entity.get('class', '')).strip().lower()
    if kind in {'person', 'human'} or entity_class in {'person', 'human'}:
        return True
    for relation in entity.get('relations', []):
        if not isinstance(relation, dict):
            continue
        predicate = str(relation.get('predicate', '')).strip().lower()
        value = str(relation.get('object', '')).strip().lower()
        if predicate.endswith('type') and value in {'person', 'human', 'foaf:person'}:
            return True
    return False


def _step_destination_id(step: dict) -> str:
    for source in (step.get('args', {}), step.get('result_payload', {})):
        if not isinstance(source, dict):
            continue
        destination = str(
            source.get('destination')
            or source.get('recipient')
            or source.get('recipient_id')
            or source.get('destination_id')
            or ''
        ).strip()
        if destination:
            return destination
    return ''


def _latest_step_target(step: dict, report_context: dict) -> str:
    payload = step.get('result_payload', {})
    if isinstance(payload, dict):
        target = str(
            payload.get('target')
            or payload.get('object')
            or payload.get('location')
            or payload.get('target_frame')
            or ''
        ).strip()
        if target:
            return target
    args = step.get('args', {})
    if isinstance(args, dict):
        target = str(
            args.get('target')
            or args.get('object')
            or args.get('location')
            or args.get('target_frame')
            or ''
        ).strip()
        if target:
            return target
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        return str(
            latest_payload.get('target')
            or latest_payload.get('object')
            or latest_payload.get('location')
            or latest_payload.get('target_frame')
            or ''
        ).strip()
    return ''


def _successful_non_report_steps(report_context: dict) -> list[dict]:
    return [
        step
        for step in report_context.get('steps', [])
        if isinstance(step, dict)
        and str(step.get('status', '')).strip().lower() == 'succeeded'
        and str(step.get('name', '')).strip().lower() != 'report_result'
    ]


def _ordered_navigation_report(successful_steps: list[dict], report_context: dict) -> str:
    goal_text = str(report_context.get('goal_text', '')).strip().lower()
    if not any(marker in goal_text for marker in ('each object', 'every object', 'all objects')):
        return ''
    targets = []
    for step in successful_steps:
        if str(step.get('name', '')).strip().lower() not in {'navigate_to', 'walk_to'}:
            continue
        target = _latest_step_target(step, report_context)
        if target and target not in targets:
            targets.append(target)
    if len(targets) < 2:
        return ''
    labels = [_friendly_target_label(target) for target in targets]
    return 'I walked to %s and reported each arrival.' % _natural_join(labels)


def _friendly_target_label(target: str) -> str:
    clean = str(target or '').strip()
    if not clean:
        return 'the target'
    lowered = clean.lower()
    for prefix in ('codex_probe_', 'detected_', 'object_'):
        if lowered.startswith(prefix):
            clean = clean[len(prefix):]
            break
    return 'the %s' % clean.replace('_', ' ')


def _natural_join(items: list[str]) -> str:
    clean_items = [str(item or '').strip() for item in items if str(item or '').strip()]
    if not clean_items:
        return ''
    if len(clean_items) == 1:
        return clean_items[0]
    if len(clean_items) == 2:
        return '%s and %s' % (clean_items[0], clean_items[1])
    return '%s, and %s' % (', '.join(clean_items[:-1]), clean_items[-1])


def _is_placeholder_report_summary(text: str) -> bool:
    lowered = str(text or '').strip().lower()
    return any(
        marker in lowered
        for marker in (
            'can report the current scene summary',
            'can report current scene summary',
            'can report what i observed',
            'can report what i saw',
            'can report it',
            'will report back',
            'will let you know',
        )
    )


def _split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r'(?<=[.!?])\s+', str(text or '').strip())
        if item.strip()
    ]


def _is_generic_completion_sentence(sentence: str) -> bool:
    lowered = str(sentence or '').strip().lower()
    return any(
        marker in lowered
        for marker in (
            'completed the task',
            'completed this task',
            'completed the',
            'have completed',
            'i finished the task',
            'i have finished the task',
        )
    )


def _fallback_planner_dialogue_ack(dialogue_context: dict) -> str:
    """Return bounded fallback wording without exposing planner/model prose."""
    completion_context = dialogue_context.get('completion_context', {})
    if isinstance(completion_context, dict):
        completion_text = _fallback_planner_completion_ack(completion_context)
        if completion_text:
            return completion_text

    act = str(dialogue_context.get('act', '')).strip().lower()
    return {
        'progress_update': 'I am working on it now.',
        'ask_clarification': 'I need a bit more detail before I continue.',
        'ask_for_help': 'I need help to continue this task.',
        'explain_failure': 'I could not complete that task.',
        'notify_completion': 'I finished that task.',
        'notify_cancellation': 'Okay, I will stop working on that.',
    }.get(act, '')
