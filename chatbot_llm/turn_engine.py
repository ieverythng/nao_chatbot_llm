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
    'steps',
    'latest_result_summary',
    'latest_result_payload',
)
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
- You are wording the final spoken report for a completed robot task.
- Use only facts from the execution_report JSON payload.
- Summarize the whole executed step chain, not only the last step.
- Synthesize related routine steps into natural language; do not recite each
  internal motion or execution result as a separate ledger sentence.
- Use goal_text and dialogue_context to produce a coherent continuation and to
  decide which outcomes matter to the user.
- Use grounded_context to translate entity handles into meaningful user-facing
  labels and relations when the facts are present.
- Treat requested_summary as evidence or a wording hint, not as text that must
  be repeated verbatim.
- Use step results as the authority for factual execution claims.
- If a step status is succeeded, do not imply it failed.
- Mention relevant observations from scan/perception results.
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
            if fb_intent and fb_intent != 'fallback' and is_execution_intent_label(fb_intent):
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
                    and is_execution_intent_label(fallback_intent)
                ):
                    inferred_route = _EXECUTION_ROUTE

        kb_query_intent = _infer_kb_query_intent_from_text(user_text)
        if kb_query_intent:
            stated_intent = str(user_intent.get('type', '')).strip().lower()
            if (
                not _has_explicit_perception_action_request(user_text)
                and stated_intent in {'', 'fallback', 'inspect_scene'}
                and str(resolved_intent or '').strip().lower()
                in {'', 'fallback', 'inspect_scene', kb_query_intent}
            ):
                inferred_route = _KNOWLEDGE_QUERY_ROUTE
                resolved_intent = kb_query_intent
                user_intent = dict(user_intent)
                user_intent['type'] = kb_query_intent

        if inferred_route == _DIALOGUE_ROUTE and not resolved_intent and not user_intent:
            intent_source = 'llm_response_route'
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
            return {'verbal_ack': ack_text}
        if _looks_like_json_payload(raw_response):
            _warn(self._logger, 'Response JSON did not include a safe verbal acknowledgement')
            return {'verbal_ack': self._config.fallback_response}
        return {'verbal_ack': str(raw_response).strip()}

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
                return candidate
        return _extract_ack_text(raw_response) or str(raw_response).strip()

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
            response_prompt_addendum=_join_prompt_addenda(
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
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(marker in lowered for marker in _EXECUTION_HINT_MARKERS)


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
    # Catch specific-object attribute queries like "What is the name/color of X?"
    # These are KB lookups even when detect_intent returns 'fallback'.
    if _looks_like_object_attribute_query(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    return ''


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
        return 'I finished: %s.' % goal_text.rstrip('.')
    return ''


def _fallback_execution_report_ack(report_context: dict) -> str:
    successful_steps = [
        step for step in report_context.get('steps', [])
        if isinstance(step, dict) and str(step.get('status', '')).strip().lower() == 'succeeded'
    ]
    if successful_steps:
        summaries = [
            str(step.get('result_summary', '')).strip()
            for step in successful_steps
            if str(step.get('result_summary', '')).strip()
        ]
        if summaries:
            return ' '.join(summaries[-2:])

    latest_summary = str(report_context.get('latest_result_summary', '')).strip()
    if latest_summary:
        return latest_summary
    latest_payload = report_context.get('latest_result_payload', {})
    if isinstance(latest_payload, dict):
        summary_text = str(latest_payload.get('summary_text', '')).strip()
        if summary_text:
            return summary_text
    goal_text = str(report_context.get('goal_text', '')).strip()
    if goal_text:
        return 'I finished: %s.' % goal_text.rstrip('.')
    return ''


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
