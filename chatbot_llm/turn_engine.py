"""Two-stage turn execution policy for the migrated chatbot backend."""

from __future__ import annotations

import json
from dataclasses import dataclass

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.backend_config import coerce_float
from chatbot_llm.chat_history import history_to_messages
from chatbot_llm.chat_history import messages_to_history
from chatbot_llm.chat_history import trim_messages
from chatbot_llm.intent_rules import build_rule_response
from chatbot_llm.intent_rules import detect_intent
from chatbot_llm.intent_rules import normalize_intent
from chatbot_llm.prompt_builders import build_intent_prompt
from chatbot_llm.prompt_builders import build_response_prompt
from chatbot_llm.prompt_builders import load_persona_prompt
from kb_skills.intent_labels import KB_QUERY_INTENTS


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
    ' turn ',
    ' head ',
    ' bring ',
    ' grab ',
    ' pick ',
    ' place ',
    ' guide ',
    ' walk ',
)


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
            if self._config.intent_detection_mode == 'llm_with_rules_fallback':
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

        response_confidence = coerce_float(
            response_payload.get(
                'intent_confidence',
                response_payload.get('confidence', 0.0),
            )
        )

        if not resolved_intent:
            fallback_intent = normalize_intent(detect_intent(user_text), default='')
            if fallback_intent and fallback_intent != 'fallback':
                resolved_intent = fallback_intent
                if not user_intent:
                    user_intent = {'type': fallback_intent}
                elif not str(user_intent.get('type', '')).strip():
                    user_intent = dict(user_intent)
                    user_intent['type'] = fallback_intent

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

        if _has_executable_plan(user_intent):
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

    def _execute_llm_failure_turn(self, user_text: str, history: list[str]) -> TurnExecutionResult:
        return self._execute_fallback_turn(
            user_text=user_text,
            history=history,
            intent_source='llm_response_failed',
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
            response_format=self._config.response_schema,
        )
        if not raw_response:
            return {}

        parsed = _extract_json_object(raw_response)
        if parsed:
            verbal_ack = str(parsed.get('verbal_ack', '')).strip()
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
            response_format=self._config.intent_schema,
        )
        if not raw_response:
            return {}

        parsed = _extract_json_object(raw_response)
        if not parsed:
            self._logger.warn('Intent extraction response was not valid JSON')
            return {}
        return parsed

    # -----------------------------------------------------------------------
    # Prompt-history helper methods
    # -----------------------------------------------------------------------

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

def _parse_json_dict(payload: str) -> dict:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
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


def _coerce_user_intent(user_intent) -> dict:
    if isinstance(user_intent, dict):
        cleaned = {}
        for key in ('type', 'object', 'recipient', 'input', 'goal', 'ack_text', 'ack_mode'):
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

        raw_plan = user_intent.get('plan')
        if isinstance(raw_plan, list):
            cleaned_plan = []
            for step in raw_plan:
                if not isinstance(step, dict):
                    continue
                step_type = str(step.get('type', '')).strip()
                if not step_type:
                    continue
                cleaned_plan.append(
                    {
                        'type': step_type,
                        'name': str(step.get('name', '')).strip(),
                        'args': step.get('args', {}) if isinstance(step.get('args'), dict) else {},
                    }
                )
            if cleaned_plan:
                cleaned['plan'] = cleaned_plan
        return cleaned
    if isinstance(user_intent, str) and user_intent.strip():
        return {'type': user_intent.strip()}
    return {}


def _normalize_route(value) -> str:
    clean_value = str(value or '').strip().lower()
    if clean_value in _SUPPORTED_ROUTES:
        return clean_value
    return ''


def _coerce_plan(user_intent: dict) -> list[dict]:
    if not isinstance(user_intent, dict):
        return []
    raw_plan = user_intent.get('plan')
    if not isinstance(raw_plan, list):
        return []
    clean_plan = []
    for step in raw_plan:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get('type', '')).strip().lower()
        if not step_type:
            continue
        clean_plan.append(step)
    return clean_plan


def _has_executable_plan(user_intent: dict) -> bool:
    for step in _coerce_plan(user_intent):
        if str(step.get('type', '')).strip().lower() in ('skill', 'look_at', 'say'):
            return True
    return False


def _looks_like_execution_text(user_text: str) -> bool:
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(marker in lowered for marker in _EXECUTION_HINT_MARKERS)
