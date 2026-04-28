"""Upstream ``chatbot_llm`` backend adapted to the local Ollama policy."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from chatbot_msgs.action import Dialogue
from chatbot_msgs.srv import DialogueInteraction
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from std_msgs.msg import String

from chatbot_llm.backend_config import ChatbotConfig
from chatbot_llm.backend_config import declare_backend_parameters
from chatbot_llm.backend_config import load_backend_config
from chatbot_llm.chat_history import history_to_messages
from chatbot_llm.chat_history import messages_to_history
from chatbot_llm.intent_adapter import build_response_intents
from chatbot_llm.knowledge_snapshot import KnowledgeSnapshotSettings
from chatbot_llm.knowledge_snapshot import build_scene_context
from chatbot_llm.knowledge_snapshot import extract_scene_memory_entry
from chatbot_llm.knowledge_snapshot import resolve_knowledge_snapshot_settings
from chatbot_llm.knowledge_snapshot_client import KnowledgeSnapshotClient
from chatbot_llm.ollama_transport import OllamaTransport
from chatbot_llm.planner_request_adapter import build_planner_request_intent
from chatbot_llm.planner_request_adapter import should_route_intents_through_planner
from chatbot_llm.skill_catalog import build_skill_catalog_text
from chatbot_llm.turn_engine import DialogueTurnEngine
from chatbot_llm.turn_engine import _extract_ack_text
from chatbot_llm.turn_engine import _looks_like_json_payload
from hri_actions_msgs.msg import Intent

try:  # pragma: no cover - optional dependency
    from i18n_msgs.action import SetLocale
    from i18n_msgs.srv import GetLocales
except ImportError:  # pragma: no cover - optional dependency
    SetLocale = None
    GetLocales = None


SYSTEM_USER_ID = '__system__'
ASSISTANT_USER_ID = '__assistant__'
DEFAULT_ROLE = '__default__'


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
    """Lifecycle chatbot backend exposing the upstream Dialogue contract."""

    def __init__(self) -> None:
        """Declare parameters and initialize backend state containers."""
        super().__init__('chatbot_llm')

        declare_backend_parameters(self)

        self._callback_group = ReentrantCallbackGroup()
        self._session_lock = threading.Lock()

        self._dialogue_start_action = None
        self._dialogue_interaction_srv = None
        self._get_supported_locales_server = None
        self._set_default_locale_server = None

        self._diag_pub = None
        self._diag_timer = None
        self._planner_request_pub = None
        self._planner_scene_summary_sub = None
        self._planner_world_model_snapshot_sub = None
        self._planner_world_model_text_sub = None

        self._config: ChatbotConfig | None = None
        self._transport = None
        self._turn_engine = None
        self._knowledge_snapshot_client = None
        self._skill_catalog_text = ''
        self._skill_catalog_size = 0
        self._default_locale = ''

        self._dialogue_id: tuple[int, ...] | None = None
        self._dialogue_result: Dialogue.Result | None = None
        self._session: DialogueSession | None = None
        self._planner_scene_summary_payload: dict = {}
        self._planner_world_model_snapshot: dict = {}
        self._planner_world_model_text = ''

        self.get_logger().info('Chatbot backend created, awaiting lifecycle configuration.')

    # -----------------------------------------------------------------------
    # Dialogue action/service handlers
    # -----------------------------------------------------------------------

    def on_dialog_goal(self, goal_request: Dialogue.Goal):
        """Accept one dialogue at a time."""
        with self._session_lock:
            if self._dialogue_id is not None:
                self.get_logger().warn('Rejected dialogue goal because another dialogue is active')
                return GoalResponse.REJECT

        role_name = _normalize_role_name(goal_request.role.name)
        if not role_name:
            self.get_logger().warn('Rejected dialogue goal with empty role')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_dialog_accept(self, handle: ServerGoalHandle):
        """Create dialogue session state and execute the long-lived goal."""
        goal = handle.request
        dialogue_id = tuple(handle.goal_id.uuid)
        session = self._build_dialogue_session(goal, dialogue_id)
        with self._session_lock:
            self._dialogue_id = dialogue_id
            self._dialogue_result = None
            self._session = session

        self.get_logger().info(
            'Started dialogue role=%s id=%s'
            % (session.role_name, _short_uuid(dialogue_id))
        )
        handle.execute()

    def on_dialog_cancel(self, _handle: ServerGoalHandle):
        """Allow dialogue_manager to cancel long-lived dialogues."""
        return CancelResponse.ACCEPT

    def _build_dialogue_session(
        self,
        goal: Dialogue.Goal,
        dialogue_id: tuple[int, ...],
    ) -> DialogueSession:
        role_name = _normalize_role_name(goal.role.name)
        role_configuration = _normalize_role_configuration(goal.role.configuration)
        return DialogueSession(
            dialogue_id=dialogue_id,
            role_name=role_name,
            role_configuration=role_configuration,
            knowledge_settings=resolve_knowledge_snapshot_settings(
                role_configuration,
                self._config,
                logger=self.get_logger(),
            ),
            locale=str(goal.locale or self._default_locale).strip(),
            history=_seed_history(role_name, role_configuration),
        )

    def on_dialog_execute(self, handle: ServerGoalHandle):
        """Keep the dialogue action alive until cancelled or explicitly terminated."""
        dialogue_id = tuple(handle.goal_id.uuid)
        try:
            while handle.is_active:
                if handle.is_cancel_requested:
                    handle.canceled()
                    return Dialogue.Result(error_msg='Dialogue cancelled')

                with self._session_lock:
                    result = self._dialogue_result
                if result is not None:
                    if result.error_msg:
                        handle.abort()
                    else:
                        handle.succeed()
                    return result

                time.sleep(1e-2)

            return Dialogue.Result(error_msg='Dialogue execution interrupted')
        finally:
            with self._session_lock:
                if self._dialogue_id == dialogue_id:
                    self._dialogue_id = None
                    self._dialogue_result = None
                    self._session = None
            self.get_logger().info('Dialogue %s finished' % _short_uuid(dialogue_id))

    def on_dialogue_interaction(
        self,
        request: DialogueInteraction.Request,
        response: DialogueInteraction.Response,
    ):
        """Advance the active dialogue and optionally generate a reply."""
        dialogue_id = tuple(request.dialogue_id.uuid)
        with self._session_lock:
            session = self._session if self._dialogue_id == dialogue_id else None
        if session is None:
            response.error_msg = 'Received dialogue interaction for unknown dialogue id'
            self.get_logger().error(
                '[CHATBOT] Unknown dialogue interaction id=%s' % _short_uuid(dialogue_id)
            )
            return response

        text = str(request.input or '').strip()
        user_id = str(request.user_id or '').strip()
        if not user_id:
            user_id = session.last_user_id or 'anonymous_user'

        if request.locale:
            with self._session_lock:
                if self._session is not None:
                    self._session.locale = str(request.locale).strip()

        if not text:
            if request.response_expected:
                response.error_msg = 'Dialogue interaction input is empty'
            return response

        if user_id == SYSTEM_USER_ID:
            self._append_history_entry(session, 'system', text)
            if not request.response_expected:
                return response
        elif user_id == ASSISTANT_USER_ID:
            self._append_history_entry(session, 'assistant', text)
            if not request.response_expected:
                return response
        elif not request.response_expected:
            self._append_history_entry(session, 'user', text)
            with self._session_lock:
                if self._session is not None:
                    self._session.last_user_id = user_id
            return response

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
            turn_id=turn_id,
            trace=self._trace,
        )
        knowledge_context = build_scene_context(
            current_snapshot,
            recent_scene_memory=session.recent_scene_memory,
        )

        result = self._turn_engine.execute_turn(
            user_text=text,
            history=list(session.history),
            user_id=user_id,
            knowledge_snapshot=knowledge_context,
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
            if self._session is not None:
                self._session.history = list(result.updated_history)
                self._session.recent_scene_memory = self._remember_scene_memory(
                    self._session.recent_scene_memory,
                    current_snapshot,
                )
                self._session.last_user_id = user_id
                self._session.request_count += 1

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
        if self._publish_planner_request(
            session=session,
            user_id=user_id,
            turn_id=turn_id,
            user_text=text,
            knowledge_context=knowledge_context,
            result=result,
            direct_intents=direct_intents,
        ):
            response.intents = []
        else:
            response.intents = direct_intents
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
        self._planner_request_pub = None
        if self._config.planner_mode_enabled:
            self._planner_request_pub = self.create_publisher(
                Intent,
                self._config.planner_request_topic,
                10,
            )
            self._planner_scene_summary_sub = self.create_subscription(
                String,
                self._config.planner_scene_summary_topic,
                self._on_planner_scene_summary,
                10,
            )
            self._planner_world_model_snapshot_sub = self.create_subscription(
                String,
                self._config.planner_world_model_snapshot_topic,
                self._on_planner_world_model_snapshot,
                10,
            )
            self._planner_world_model_text_sub = self.create_subscription(
                String,
                self._config.planner_world_model_text_topic,
                self._on_planner_world_model_text,
                10,
            )

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
            'Configured chatbot_llm | server_url=%s model=%s intent_model=%s '
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
        """Activate the upstream action/service contract."""
        self._dialogue_start_action = ActionServer(
            self,
            Dialogue,
            '~/start_dialogue',
            execute_callback=self.on_dialog_execute,
            goal_callback=self.on_dialog_goal,
            handle_accepted_callback=self.on_dialog_accept,
            cancel_callback=self.on_dialog_cancel,
            callback_group=self._callback_group,
        )
        self._dialogue_interaction_srv = self.create_service(
            DialogueInteraction,
            '~/dialogue_interaction',
            self.on_dialogue_interaction,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            'chatbot_llm is active and serving ~/start_dialogue and '
            '~/dialogue_interaction'
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop serving dialogue requests and terminate any active session."""
        self._terminate_active_dialogue('Dialogue backend deactivated')
        if self._dialogue_start_action is not None:
            self._dialogue_start_action.destroy()
            self._dialogue_start_action = None
        if self._dialogue_interaction_srv is not None:
            self.destroy_service(self._dialogue_interaction_srv)
            self._dialogue_interaction_srv = None
        self.get_logger().info('chatbot_llm is inactive')
        return super().on_deactivate(state)

    def on_shutdown(self, _state: State) -> TransitionCallbackReturn:
        """Tear down timers, publishers, and optional locale endpoints."""
        self._terminate_active_dialogue('Dialogue backend shutdown')

        if self._diag_timer is not None:
            self.destroy_timer(self._diag_timer)
            self._diag_timer = None
        if self._diag_pub is not None:
            self.destroy_publisher(self._diag_pub)
            self._diag_pub = None
        if self._planner_request_pub is not None:
            self.destroy_publisher(self._planner_request_pub)
            self._planner_request_pub = None
        if self._planner_scene_summary_sub is not None:
            self.destroy_subscription(self._planner_scene_summary_sub)
            self._planner_scene_summary_sub = None
        if self._planner_world_model_snapshot_sub is not None:
            self.destroy_subscription(self._planner_world_model_snapshot_sub)
            self._planner_world_model_snapshot_sub = None
        if self._planner_world_model_text_sub is not None:
            self.destroy_subscription(self._planner_world_model_text_sub)
            self._planner_world_model_text_sub = None

        if self._dialogue_start_action is not None:
            self._dialogue_start_action.destroy()
            self._dialogue_start_action = None
        if self._dialogue_interaction_srv is not None:
            self.destroy_service(self._dialogue_interaction_srv)
            self._dialogue_interaction_srv = None

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

    def _append_history_entry(self, session: DialogueSession, role: str, text: str) -> None:
        """Append one system/user/assistant entry to the active session history."""
        clean_text = str(text).strip()
        if not clean_text:
            return
        messages = history_to_messages(
            session.history,
            max_history_messages=self._config.max_history_messages,
        )
        messages.append({'role': role, 'content': clean_text})
        new_history = messages_to_history(
            messages,
            max_history_messages=self._config.max_history_messages,
        )
        with self._session_lock:
            if self._session is not None and self._session.dialogue_id == session.dialogue_id:
                self._session.history = new_history

    def _publish_planner_request(
        self,
        *,
        session: DialogueSession,
        user_id: str,
        turn_id: str,
        user_text: str,
        knowledge_context: str,
        result,
        direct_intents: list[Intent],
    ) -> bool:
        """Publish the current turn to planner_llm when planner mode is enabled."""
        if self._config is None or not self._config.planner_mode_enabled:
            return False
        if self._planner_request_pub is None:
            self.get_logger().warn('planner mode is enabled but planner request publisher is unavailable')
            return False
        if not should_route_intents_through_planner(
            direct_intents,
            turn_result=result,
            user_text=user_text,
        ):
            return False

        try:
            planner_msg = build_planner_request_intent(
                turn_id=turn_id,
                user_text=user_text,
                source_user_id=user_id,
                turn_result=result,
                knowledge_context=knowledge_context,
                grounded_context=self._planner_grounded_context(knowledge_context),
                planner_request_intent=self._config.planner_request_intent,
                active_goal_id=session.active_planner_goal_id,
            )
            self._planner_request_pub.publish(planner_msg)
        except Exception as err:  # pragma: no cover - ROS publish failures are runtime-only
            self.get_logger().warn('failed to publish planner request: %s' % err)
            self._trace(turn_id, 'PLANNER_REQUEST', 'publish failed: %s' % err, level='warn')
            return False

        planner_payload = {}
        try:
            planner_payload = json.loads(planner_msg.data)
        except Exception:
            planner_payload = {}
        planner_goal_id = str(planner_payload.get('goal_id', '')).strip()
        if planner_goal_id:
            with self._session_lock:
                if self._session is not None and self._session.dialogue_id == session.dialogue_id:
                    self._session.active_planner_goal_id = planner_goal_id

        self._trace(
            turn_id,
            'PLANNER_REQUEST',
            'published planner request on %s goal_id=%s kind=%s'
            % (
                self._config.planner_request_topic,
                planner_payload.get('goal_id', ''),
                planner_payload.get('request_kind', ''),
            ),
        )
        return True

    def _on_planner_scene_summary(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data or '').strip() or '{}')
        except json.JSONDecodeError:
            payload = {}
        self._planner_scene_summary_payload = payload if isinstance(payload, dict) else {}

    def _on_planner_world_model_snapshot(self, msg: String) -> None:
        try:
            payload = json.loads(str(msg.data or '').strip() or '{}')
        except json.JSONDecodeError:
            payload = {}
        self._planner_world_model_snapshot = payload if isinstance(payload, dict) else {}

    def _on_planner_world_model_text(self, msg: String) -> None:
        self._planner_world_model_text = str(msg.data or '').strip()

    def _planner_grounded_context(self, knowledge_context: str) -> dict:
        knowledge_snapshot = {}
        clean_knowledge_context = str(knowledge_context or '').strip()
        if clean_knowledge_context:
            knowledge_snapshot['summary_text'] = clean_knowledge_context
        return {
            'knowledge_snapshot': knowledge_snapshot,
            'scene_summary': dict(self._planner_scene_summary_payload),
            'world_model_snapshot': dict(self._planner_world_model_snapshot),
            'world_model_text': self._planner_world_model_text,
        }

    def _terminate_active_dialogue(self, error_msg: str) -> None:
        """Unblock the action execution loop if a dialogue is still active."""
        with self._session_lock:
            if self._dialogue_id is None or self._dialogue_result is not None:
                return
            result = Dialogue.Result()
            result.results = '{}'
            result.error_msg = str(error_msg)
            self._dialogue_result = result

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
