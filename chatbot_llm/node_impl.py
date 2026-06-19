"""Upstream ``chatbot_llm`` backend adapted to the local Ollama policy."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from chatbot_msgs.msg import DialogueRole
from chatbot_msgs.srv import DialogueInteraction, PrepareDialogue
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.backend_config import declare_backend_parameters
from chatbot_llm.backend_config import load_backend_config
from chatbot_llm.intent_adapter import build_response_intents
from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import build_grounded_context_block
from chatbot_llm.knowledge_snapshot import build_scene_digest
from chatbot_llm.knowledge_snapshot import extract_scene_memory_entry
from chatbot_llm.knowledge_snapshot import resolve_knowledge_snapshot_settings
from chatbot_llm.knowledge_snapshot_client import KnowledgeSnapshotClient
from chatbot_llm.ollama_transport import OllamaTransport
from chatbot_llm.planner_handoff import PlannerHandoff
from chatbot_llm.skill_catalog import build_skill_catalog_text
from chatbot_llm.skill_catalog import build_skill_catalog_text_from_shared_registry
from chatbot_llm.turn_engine import DialogueTurnEngine
from chatbot_llm.turn_engine import _extract_ack_text
from chatbot_llm.turn_engine import _looks_like_json_payload
from hri_actions_msgs.msg import Intent
from std_msgs.msg import String

try:  # pragma: no cover - optional dependency
    from i18n_msgs.action import SetLocale
    from i18n_msgs.srv import GetLocales
except ImportError:  # pragma: no cover - optional dependency
    SetLocale = None
    GetLocales = None


SYSTEM_USER_ID = '__system__'
ASSISTANT_USER_ID = '__assistant__'
DEFAULT_ROLE = '__default__'

# Cap only the LLM-facing grounded-context serialization (visible entities first);
# the grounded-context contract and the authoritative scene-digest counts are untouched.
_GROUNDED_BLOCK_MAX_ENTITIES = 30


# ---------------------------------------------------------------------------
# Active dialogue session state
# ---------------------------------------------------------------------------

@dataclass
class DialogueSession:
    """In-memory state for the currently active dialogue."""

    dialogue_id: tuple[int, ...]
    role_name: str
    role_configuration: str
    knowledge_settings: KnowledgeSnapshotSettings
    locale: str
    history: list[str] = field(default_factory=list)
    recent_scene_memory: list[str] = field(default_factory=list)
    request_count: int = 0
    last_user_id: str = 'anonymous_user'
    active_planner_goal_id: str = ''


# ---------------------------------------------------------------------------
# Lifecycle chatbot node
# ---------------------------------------------------------------------------

class LLMChatbot(Node):
    """Lifecycle chatbot backend exposing the stateless chatbot_msgs v4 contract."""

    def __init__(self) -> None:
        """Declare parameters and initialize backend state containers."""
        super().__init__('chatbot_llm')

        declare_backend_parameters(self)

        self._callback_group = ReentrantCallbackGroup()
        self._session_lock = threading.Lock()

        self._prepare_dialogue_srv = None
        self._dialogue_interaction_srv = None
        self._get_supported_locales_server = None
        self._set_default_locale_server = None

        self._diag_pub = None
        self._diag_timer = None
        self._llm_keepalive_timer = None
        self._planner_handoff: PlannerHandoff | None = None
        self._turn_trace_pub = None

        self._config: ChatbotConfig | None = None
        self._transport = None
        self._turn_engine = None
        self._knowledge_snapshot_client = None
        self._skill_catalog_text = ''
        self._skill_catalog_size = 0
        self._default_locale = ''

        self._session: DialogueSession | None = None
        self._dialogue_sessions: dict[tuple[int, ...], DialogueSession] = {}

        self.get_logger().info('Chatbot backend created, awaiting lifecycle configuration.')

    # -----------------------------------------------------------------------
    # Stateless dialogue services
    # -----------------------------------------------------------------------

    def on_prepare_dialogue(
        self,
        request: PrepareDialogue.Request,
        response: PrepareDialogue.Response,
    ):
        """Seed per-dialogue metadata for later stateless interactions."""
        dialogue_id = tuple(request.dialogue_id.uuid)
        session = self._build_dialogue_session(
            dialogue_id=dialogue_id,
            role=request.role,
            locale='',
        )
        with self._session_lock:
            self._dialogue_sessions[dialogue_id] = session
            self._session = session
        self.get_logger().debug(
            '[CHATBOT] prepare_dialogue role=%s id=%s'
            % (session.role_name, _short_uuid(dialogue_id))
        )
        return response

    def _build_dialogue_session(
        self,
        dialogue_id: tuple[int, ...],
        role: DialogueRole,
        locale: str,
    ) -> DialogueSession:
        role_name = _normalize_role_name(getattr(role, 'name', ''))
        role_configuration = _normalize_role_configuration(
            getattr(role, 'configuration', ''),
        )
        return DialogueSession(
            dialogue_id=dialogue_id,
            role_name=role_name or DEFAULT_ROLE,
            role_configuration=role_configuration,
            knowledge_settings=resolve_knowledge_snapshot_settings(
                role_configuration,
                self._config,
                logger=self.get_logger(),
            ),
            locale=str(locale or self._default_locale).strip(),
            history=_seed_history(role_name or DEFAULT_ROLE, role_configuration),
        )

    def on_dialogue_interaction(
        self,
        request: DialogueInteraction.Request,
        response: DialogueInteraction.Response,
    ):
        """Process one stateless dialogue interaction request."""
        dialogue_id = tuple(request.dialogue_id.uuid)
        with self._session_lock:
            session = self._dialogue_sessions.get(dialogue_id)
        if session is None:
            session = self._build_dialogue_session(
                dialogue_id=dialogue_id,
                role=request.role,
                locale=str(getattr(request, 'locale', '') or ''),
            )
            with self._session_lock:
                self._dialogue_sessions[dialogue_id] = session
                self._session = session
        clean_locale = str(getattr(request, 'locale', '') or '').strip()
        if clean_locale:
            with self._session_lock:
                session.locale = clean_locale
                self._session = session

        history_entries = _history_entries_from_stateless_request(request)
        if not history_entries:
            response.error_msg = 'Dialogue interaction history is empty'
            return response

        turn_role, user_id, text = _last_turn_descriptor(request)
        if not text:
            response.error_msg = 'Dialogue interaction input is empty'
            return response

        if turn_role == 'user':
            engine_history = list(history_entries[:-1])
        else:
            engine_history = list(history_entries)

        with self._session_lock:
            session.history = list(engine_history)
            session.last_user_id = user_id
            self._session = session

        turn_id = '%s:%d' % (session.role_name, session.request_count + 1)
        self.get_logger().info(
            '[CHATBOT] dialogue=%s user=%s turn=%s input=%s'
            % (
                _short_uuid(dialogue_id),
                user_id,
                turn_id,
                _preview_text(text),
            )
        )

        current_snapshot = self._knowledge_snapshot_client.fetch_snapshot(
            session.knowledge_settings,
            user_text=text,
            turn_id=turn_id,
            trace=self._trace,
        )
        current_snapshot_rows = tuple(
            getattr(self._knowledge_snapshot_client, 'last_rows', ())
        )
        grounded_context = {}
        if self._planner_handoff is not None:
            grounded_context = self._planner_handoff.grounded_context(
                current_snapshot,
                knowledge_rows=current_snapshot_rows,
            )
        scene_digest = build_scene_digest(grounded_context)
        grounded_context_block = build_grounded_context_block(
            grounded_context,
            max_entities=_GROUNDED_BLOCK_MAX_ENTITIES,
        )
        grounded_context_text = '\n\n'.join(
            part for part in (scene_digest, grounded_context_block) if part
        )
        if grounded_context_text:
            self._trace(
                turn_id,
                'GROUNDED_CONTEXT',
                grounded_context_text,
            )

        result = self._turn_engine.execute_turn(
            user_text=text,
            history=list(session.history),
            user_id=user_id,
            knowledge_snapshot=grounded_context_text,
            progress_callback=lambda status, progress: self._trace(
                turn_id,
                'PROGRESS',
                '%s %.2f' % (status, progress),
                level='debug',
            ),
            turn_id=turn_id,
            trace=self._trace,
        )

        with self._session_lock:
            tracked = self._dialogue_sessions.get(dialogue_id)
            if tracked is not None:
                tracked.history = list(result.updated_history)
                tracked.recent_scene_memory = self._remember_scene_memory(
                    tracked.recent_scene_memory,
                    current_snapshot,
                )
                tracked.last_user_id = user_id
                tracked.request_count += 1
                self._session = tracked

        response.response = _sanitize_spoken_response(
            result.verbal_ack,
            fallback_response=self._config.fallback_response,
        )
        direct_intents = []
        if result.route != 'execution':
            direct_intents = build_response_intents(
                resolved_intent=result.intent,
                user_intent=result.user_intent,
                source_user_id=user_id,
                verbal_ack=result.verbal_ack,
                raw_input=text,
                confidence=result.intent_confidence,
            )
        planner_handoff_allowed = user_id != SYSTEM_USER_ID
        planner_handoff_published = False
        if (
            planner_handoff_allowed
            and self._planner_handoff is not None
            and self._planner_handoff.publish_execution_turn_if_needed(
                session=session,
                user_id=user_id,
                turn_id=turn_id,
                user_text=text,
                knowledge_context=grounded_context_text,
                result=result,
                direct_intents=direct_intents,
                grounded_context=grounded_context,
            )
        ):
            planner_handoff_published = True
            response.intents = []
        else:
            response.intents = direct_intents
        if user_id == SYSTEM_USER_ID:
            response.intents = []
        response.dialogue_terminal = False
        response.results = ''
        self._publish_turn_trace_event(
            {
                'event_type': 'chatbot_turn_result',
                'turn_id': turn_id,
                'route': result.route,
                'intent': result.intent,
                'intent_source': result.intent_source,
                'intent_confidence': result.intent_confidence,
                'user_intent': dict(result.user_intent or {}),
                'verbal_ack': response.response,
                'planner_mode_enabled': bool(self._config.planner_mode_enabled),
                'planner_handoff_allowed': planner_handoff_allowed,
                'planner_handoff_published': planner_handoff_published,
                'direct_intent_count': len(direct_intents),
                'grounded_context': grounded_context,
            }
        )
        response.error_msg = ''
        return response

    def on_get_supported_locales(self, _request, response):
        """Return an empty locale list to mean 'implementation-dependent'."""
        response.locales = []
        return response

    def on_set_default_locale_goal(self, _goal_request):
        """Accept locale-setting requests when the dependency exists."""
        return GoalResponse.ACCEPT

    def on_set_default_locale_exec(self, goal_handle):
        """Persist the default locale in-process and report success."""
        self._default_locale = str(getattr(goal_handle.request, 'locale', '')).strip()
        result = SetLocale.Result() if SetLocale is not None else None
        goal_handle.succeed()
        return result

    # -----------------------------------------------------------------------
    # ROS lifecycle transitions
    # -----------------------------------------------------------------------

    def on_configure(self, _state: State) -> TransitionCallbackReturn:
        """Configure transport, prompts, diagnostics, and locale interfaces."""
        self._config = load_backend_config(self)
        self._transport = OllamaTransport(
            server_url=self._config.server_url,
            context_window_tokens=self._config.context_window_tokens,
            logger=self.get_logger(),
        )

        self._skill_catalog_text = ''
        self._skill_catalog_size = 0
        if self._config.use_skill_catalog and self._config.skill_catalog_packages:
            self._skill_catalog_text, descriptors = build_skill_catalog_text_from_shared_registry(
                max_entries=self._config.skill_catalog_max_entries,
                max_chars=self._config.skill_catalog_max_chars,
            )
            if not descriptors:
                self._skill_catalog_text, descriptors = build_skill_catalog_text(
                    package_names=self._config.skill_catalog_packages,
                    max_entries=self._config.skill_catalog_max_entries,
                    max_chars=self._config.skill_catalog_max_chars,
                    logger=self.get_logger(),
                )
            self._skill_catalog_size = len(descriptors)

        self._knowledge_snapshot_client = KnowledgeSnapshotClient(
            node=self,
            callback_group=self._callback_group,
            service_name=self._config.knowledge_query_service_name,
            timeout_sec=self._config.knowledge_query_timeout_sec,
        )

        self._turn_engine = DialogueTurnEngine(
            config=self._config,
            transport=self._transport,
            logger=self.get_logger(),
            skill_catalog_text=self._skill_catalog_text,
        )

        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self._diag_timer = self.create_timer(1.0, self.publish_diagnostics)
        self._planner_handoff = None
        if self._config.planner_mode_enabled:
            self._planner_handoff = PlannerHandoff(
                self,
                self._config,
                trace=self._trace,
                on_planner_goal_id=self._on_planner_goal_committed,
            )
        if self._config.turn_trace_enabled:
            self._turn_trace_pub = self.create_publisher(
                String,
                self._config.turn_trace_topic,
                10,
            )

        if not self._run_llm_preflight():
            return TransitionCallbackReturn.FAILURE
        self._start_llm_keepalive()

        if GetLocales is not None and SetLocale is not None:
            self._get_supported_locales_server = self.create_service(
                GetLocales,
                '~/get_supported_locales',
                self.on_get_supported_locales,
            )
            self._set_default_locale_server = ActionServer(
                self,
                SetLocale,
                '~/set_default_locale',
                goal_callback=self.on_set_default_locale_goal,
                execute_callback=self.on_set_default_locale_exec,
                callback_group=self._callback_group,
            )
        else:
            self.get_logger().warn(
                'i18n_msgs is unavailable; locale action/service will not be created'
            )

        self.get_logger().info(
            '[STACK READY] chatbot_llm configured | server_url=%s model=%s intent_model=%s '
            'intent_mode=%s skill_catalog=%s planner_mode=%s planner_topic=%s'
            % (
                self._config.server_url,
                self._config.model,
                self._config.intent_model,
                self._config.intent_detection_mode,
                self._skill_catalog_size,
                self._config.planner_mode_enabled,
                self._config.planner_request_topic,
            )
        )
        self._transport.log_model_inventory()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the stateless dialogue services."""
        self._prepare_dialogue_srv = self.create_service(
            PrepareDialogue,
            '~/prepare_dialogue',
            self.on_prepare_dialogue,
            callback_group=self._callback_group,
        )
        self._dialogue_interaction_srv = self.create_service(
            DialogueInteraction,
            '~/dialogue_interaction',
            self.on_dialogue_interaction,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            'chatbot_llm is active and serving ~/prepare_dialogue and '
            '~/dialogue_interaction'
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop serving dialogue requests and terminate any active session."""
        if self._prepare_dialogue_srv is not None:
            self.destroy_service(self._prepare_dialogue_srv)
            self._prepare_dialogue_srv = None
        if self._dialogue_interaction_srv is not None:
            self.destroy_service(self._dialogue_interaction_srv)
            self._dialogue_interaction_srv = None
        with self._session_lock:
            self._dialogue_sessions.clear()
            self._session = None
        self.get_logger().info('chatbot_llm is inactive')
        return super().on_deactivate(state)

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        """Tear down timers, publishers, and optional locale endpoints."""
        if self._diag_timer is not None:
            self.destroy_timer(self._diag_timer)
            self._diag_timer = None
        if self._llm_keepalive_timer is not None:
            self.destroy_timer(self._llm_keepalive_timer)
            self._llm_keepalive_timer = None
        if self._diag_pub is not None:
            self.destroy_publisher(self._diag_pub)
            self._diag_pub = None
        if self._turn_trace_pub is not None:
            self.destroy_publisher(self._turn_trace_pub)
            self._turn_trace_pub = None
        if self._planner_handoff is not None:
            self._planner_handoff.destroy()
            self._planner_handoff = None

        if self._prepare_dialogue_srv is not None:
            self.destroy_service(self._prepare_dialogue_srv)
            self._prepare_dialogue_srv = None
        if self._dialogue_interaction_srv is not None:
            self.destroy_service(self._dialogue_interaction_srv)
            self._dialogue_interaction_srv = None
        with self._session_lock:
            self._dialogue_sessions.clear()
            self._session = None

        if self._get_supported_locales_server is not None:
            self.destroy_service(self._get_supported_locales_server)
            self._get_supported_locales_server = None
        if self._set_default_locale_server is not None:
            self._set_default_locale_server.destroy()
            self._set_default_locale_server = None

        self.get_logger().info('chatbot_llm finalized')
        return TransitionCallbackReturn.SUCCESS

    # -----------------------------------------------------------------------
    # Diagnostics and session bookkeeping
    # -----------------------------------------------------------------------

    def publish_diagnostics(self) -> None:
        """Publish compact runtime diagnostics."""
        arr = DiagnosticArray()
        session = self._session
        status = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name='/chatbot_llm',
            message='chatbot_llm is running',
            values=[
                KeyValue(key='active_dialogue', value=str(session is not None)),
                KeyValue(
                    key='active_role',
                    value=session.role_name if session is not None else '',
                ),
                KeyValue(
                    key='request_count',
                    value=str(session.request_count if session is not None else 0),
                ),
                KeyValue(
                    key='recent_scene_memory_count',
                    value=str(len(session.recent_scene_memory) if session is not None else 0),
                ),
                KeyValue(key='model', value=self._config.model if self._config else ''),
                KeyValue(
                    key='intent_mode',
                    value=self._config.intent_detection_mode if self._config else '',
                ),
                KeyValue(key='skill_catalog_entries', value=str(self._skill_catalog_size)),
                KeyValue(
                    key='knowledge_enabled',
                    value=str(self._config.knowledge_enabled if self._config else False),
                ),
                KeyValue(
                    key='planner_mode_enabled',
                    value=str(self._config.planner_mode_enabled if self._config else False),
                ),
            ],
        )
        arr.status = [status]
        arr.header.stamp = self.get_clock().now().to_msg()
        self._diag_pub.publish(arr)

    def _on_planner_goal_committed(
        self,
        session: DialogueSession,
        goal_id: str,
    ) -> None:
        """Persist planner goal metadata for this dialogue id."""
        with self._session_lock:
            tracked = self._dialogue_sessions.get(session.dialogue_id)
            if tracked is None:
                return
            tracked.active_planner_goal_id = goal_id
            if self._session is not None and self._session.dialogue_id == session.dialogue_id:
                self._session.active_planner_goal_id = goal_id

    def _remember_scene_memory(
        self,
        existing_entries: list[str],
        current_snapshot: str,
    ) -> list[str]:
        """Retain a bounded sequence of compact scene summaries across turns."""
        limit = max(0, self._config.scene_memory_turns if self._config else 0)
        if limit <= 0:
            return []

        current_entry = extract_scene_memory_entry(current_snapshot)
        retained = [str(entry).strip() for entry in existing_entries if str(entry).strip()]
        if not current_entry:
            return retained[-limit:]

        if retained and retained[-1].lower() == current_entry.lower():
            return retained[-limit:]

        retained.append(current_entry)
        return retained[-limit:]

    def _trace(self, turn_id: str, stage: str, message: str, level: str = 'info') -> None:
        """Compact structured logging helper used by the turn engine."""
        line = '[turn:%s] %s | %s' % (turn_id or 'unknown', stage, message)
        if level == 'debug':
            self.get_logger().debug(line)
            return
        if level == 'warn':
            self.get_logger().warn(line)
            return
        if level == 'error':
            self.get_logger().error(line)
            return
        self.get_logger().info(line)

    def _publish_turn_trace_event(self, payload: dict) -> None:
        if self._turn_trace_pub is None:
            return
        msg = String()
        msg.data = json.dumps(dict(payload or {}), separators=(',', ':'), ensure_ascii=True)
        self._turn_trace_pub.publish(msg)

    def _run_llm_preflight(self) -> bool:
        """Warm response and intent models before lifecycle activation."""
        config = self._config
        if config is None or self._transport is None or not config.preflight_enabled:
            return True

        models = _unique_models(config.model, config.intent_model)
        self.get_logger().info(
            '[LLM PREFLIGHT] chatbot starting | models=%s timeout=%.1fs required=%s '
            'attempts=%d realistic=%s'
            % (
                ','.join(models),
                config.preflight_timeout_sec,
                config.preflight_required,
                config.preflight_attempts,
                config.preflight_realistic_enabled,
            )
        )
        failed_models = []
        for model in models:
            if self._run_model_readiness_probes(model):
                continue
            failed_models.append(model)
            self.get_logger().error('[LLM PREFLIGHT] chatbot model failed | model=%s' % model)

        if failed_models and config.preflight_required:
            self.get_logger().error(
                '[LLM PREFLIGHT] chatbot required preflight failed | models=%s'
                % ','.join(failed_models)
            )
            return False
        return True

    def _run_model_readiness_probes(self, model: str) -> bool:
        config = self._config
        if config is None or self._transport is None:
            return True

        for attempt in range(1, config.preflight_attempts + 1):
            if not self._transport.preflight(
                model=model,
                timeout_sec=config.preflight_timeout_sec,
                temperature=config.temperature,
                top_p=config.top_p,
                think=config.think,
            ):
                self.get_logger().warn(
                    '[LLM PREFLIGHT] chatbot tiny probe failed | model=%s attempt=%d/%d'
                    % (model, attempt, config.preflight_attempts)
                )
                continue

            if config.preflight_realistic_enabled and not self._transport.readiness_probe(
                model=model,
                timeout_sec=config.preflight_timeout_sec,
                temperature=config.temperature,
                top_p=config.top_p,
                think=config.think,
                max_tokens=config.response_max_tokens,
            ):
                self.get_logger().warn(
                    '[LLM PREFLIGHT] chatbot realistic probe failed | model=%s attempt=%d/%d'
                    % (model, attempt, config.preflight_attempts)
                )
                continue

            self.get_logger().info(
                '[LLM PREFLIGHT] chatbot model ready | model=%s attempt=%d/%d'
                % (model, attempt, config.preflight_attempts)
            )
            return True
        return False

    def _start_llm_keepalive(self) -> None:
        config = self._config
        if (
            config is None
            or self._transport is None
            or not config.preflight_enabled
            or config.preflight_keepalive_interval_sec <= 0.0
        ):
            return
        self._llm_keepalive_timer = self.create_timer(
            config.preflight_keepalive_interval_sec,
            self._llm_keepalive,
        )

    def _llm_keepalive(self) -> None:
        config = self._config
        if config is None or self._transport is None:
            return
        for model in _unique_models(config.model, config.intent_model):
            if not self._transport.preflight(
                model=model,
                timeout_sec=min(config.preflight_timeout_sec, 15.0),
                temperature=config.temperature,
                top_p=config.top_p,
                think=config.think,
            ):
                self.get_logger().warn('[LLM PREFLIGHT] chatbot keepalive failed | model=%s' % model)


# ---------------------------------------------------------------------------
# Module-local helpers
# ---------------------------------------------------------------------------

def _seed_history(role_name: str, role_configuration: str) -> list[str]:
    entries = []
    clean_role = _normalize_role_name(role_name)
    clean_config = _normalize_role_configuration(role_configuration)

    if clean_role != DEFAULT_ROLE:
        system_message = 'Dialogue role: %s.' % clean_role
        if clean_config and clean_config != '{}':
            system_message += ' Role configuration: %s' % clean_config
        entries.append('system:%s' % system_message)
    elif clean_config and clean_config != '{}':
        entries.append('system:Dialogue configuration: %s' % clean_config)

    return entries


def _history_entries_from_stateless_request(request: DialogueInteraction.Request) -> list[str]:
    """Convert stateless DialogueInteraction history into role-prefixed entries."""
    entries: list[str] = []
    summary = str(getattr(request, 'summary', '') or '').strip()
    if summary:
        entries.append('system:Prior conversation summary:\n%s' % summary)

    for utterance in list(getattr(request, 'history', []) or []):
        speaker = str(getattr(utterance, 'speaker', '') or '').strip()
        text = str(getattr(utterance, 'text', '') or '').strip()
        if not text:
            continue
        if speaker == SYSTEM_USER_ID:
            role = 'system'
        elif speaker == ASSISTANT_USER_ID:
            role = 'assistant'
        else:
            role = 'user'
        entries.append('%s:%s' % (role, text))
    return entries


def _last_turn_descriptor(request: DialogueInteraction.Request) -> tuple[str, str, str]:
    """Return (role, user_id, text) for the latest utterance in one request."""
    history = list(getattr(request, 'history', []) or [])
    if not history:
        return 'user', 'anonymous_user', ''

    last = history[-1]
    speaker = str(getattr(last, 'speaker', '') or '').strip()
    text = str(getattr(last, 'text', '') or '').strip()
    if speaker == SYSTEM_USER_ID:
        return 'system', SYSTEM_USER_ID, text
    if speaker == ASSISTANT_USER_ID:
        return 'assistant', ASSISTANT_USER_ID, text
    user_id = speaker or 'anonymous_user'
    return 'user', user_id, text


def _unique_models(*models: str) -> list[str]:
    unique = []
    seen = set()
    for model in models:
        clean_model = str(model or '').strip()
        if not clean_model or clean_model in seen:
            continue
        seen.add(clean_model)
        unique.append(clean_model)
    return unique


def _short_uuid(dialogue_id: tuple[int, ...] | None) -> str:
    if not dialogue_id:
        return 'unknown'
    return ''.join('%02x' % value for value in dialogue_id[:4])


def _preview_text(text: str, max_len: int = 72) -> str:
    clean = ' '.join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + '...'


def _sanitize_spoken_response(text: str, *, fallback_response: str) -> str:
    clean_text = str(text or '').strip()
    if not clean_text:
        return ''
    extracted_ack = _extract_ack_text(clean_text)
    if extracted_ack:
        return extracted_ack
    if _looks_like_json_payload(clean_text):
        return str(fallback_response or '').strip()
    return clean_text


def _normalize_role_name(value: str) -> str:
    return str(value or DEFAULT_ROLE).strip() or DEFAULT_ROLE


def _normalize_role_configuration(value: str) -> str:
    return str(value or '{}').strip() or '{}'
