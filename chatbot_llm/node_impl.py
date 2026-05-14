# Copyright (c) 2026 TODO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
from string import Template
from uuid import UUID

from chatbot_msgs.msg import DialogueRole
from chatbot_msgs.srv import DialogueInteraction, PrepareDialogue
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_actions_msgs.msg import Intent
from i18n_msgs.action import SetLocale
from i18n_msgs.srv import GetLocales
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.logging import LoggingSeverity

from .llm_client import LLMClient
from .messages import (
    build_llm_messages,
    last_user_speaker,
    render_system_prompt,
    trim_messages,
)
from .response_parser import (
    chatbot_response_schema,
    extract_json_object,
    intent_to_dict,
    parse_chatbot_response,
)
from .role_handlers import handler_for_role


class LLMChatbot(Node):
    """
    Implementation of chatbot_llm.

    Lifecycle ROS 2 node that exposes a stateless DialogueInteraction
    service backed by an OpenAI-compatible LLM. Each interaction call
    carries the full dialogue history; no per-dialogue state is kept
    between calls. An optional PrepareDialogue service signals that a
    new dialogue is about to start, letting the backend warm up
    role-specific resources.

    Use the parameters `server_url` and `model` to configure the LLM
    server. `system_prompt` is a string.Template substituted with
    `$user_id`, `$robot_name`, `$role`, `$action_list`, `$environment`
    on every call.
    """

    def __init__(self) -> None:
        """Construct the node."""
        super().__init__('intent_extractor_chatbot_llm')

        # Declare ROS parameters. Should mimick the one listed in config/00-defaults.yaml
        self.declare_parameter(
            'server_url', "http://localhost:11434",
            ParameterDescriptor(
                description='URL of the OpenAI-compatible LLM server'
            )
        )
        self.declare_parameter(
            'model', "llama3.2",
            ParameterDescriptor(description='LLM model to use')
        )
        self.declare_parameter(
            'api_key', "",
            ParameterDescriptor(
                description='API key to use for the LLM server, if any.'
            )
        )
        self.declare_parameter(
            'system_prompt', "You are a helpful interactive robot.",
            ParameterDescriptor(description='System prompt template to use for the LLM')
        )
        self.declare_parameter(
            'robot_name', "robot",
            ParameterDescriptor(
                description='Name of the robot, substituted into '
                '$robot_name in the system prompt'
            )
        )
        self.declare_parameter(
            'request_timeout', 30.0,
            ParameterDescriptor(
                description='Timeout (in seconds) for HTTP requests to the LLM server'
            )
        )
        self.declare_parameter(
            'max_history_turns', 10,
            ParameterDescriptor(
                description='Maximum number of user/assistant turn pairs kept in '
                'the LLM message list (the system prompt is always preserved)'
            )
        )

        self.get_logger().info("Initialising...")

        self._prepare_dialogue_srv = None
        self._dialogue_interaction_srv = None
        self._get_supported_locales_server = None
        self._set_default_locale_server = None

        self._diag_pub = None
        self._diag_timer = None

        self._llm_client = None
        self._system_prompt_tpl = None
        self._robot_name = None
        self._max_history_turns = None

        # Diagnostics counters. The chatbot keeps no per-dialogue state;
        # we track only request counts and the set of dialogue_ids we've
        # seen recently (for observability).
        self._counters_lock = threading.Lock()
        self._nb_requests = 0
        self._nb_prepared = 0

        self.get_logger().info('Chatbot chatbot_llm started, but not yet configured.')

    # ------------------------------------------------------------------
    # PrepareDialogue service (optional warm-up).
    # ------------------------------------------------------------------

    def on_prepare_dialogue(
        self,
        request: PrepareDialogue.Request,
        response: PrepareDialogue.Response,
    ) -> PrepareDialogue.Response:
        """
        Acknowledge an upcoming dialogue.

        The default implementation is effectively a no-op: this
        backend has no per-dialogue resources to warm up beyond what
        each DialogueInteraction call already carries. The service
        exists so dialogue_manager can opt into a future-proof
        handshake without breaking either side.
        """
        try:
            dialogue_id = UUID(bytes=bytes(request.dialogue_id.uuid))
        except Exception as exc:  # malformed UUID — log & accept silently
            self.get_logger().warn(
                f"prepare_dialogue: malformed dialogue_id ({exc}); accepting anyway"
            )
            dialogue_id = None
        with self._counters_lock:
            self._nb_prepared += 1
        self.get_logger().info(
            f"prepare_dialogue: dialogue_id={dialogue_id} "
            f"role={request.role.name!r} locale={request.locale!r}"
        )
        return response

    # ------------------------------------------------------------------
    # DialogueInteraction service (stateless main turn handler).
    # ------------------------------------------------------------------

    def on_dialogue_interaction(
        self,
        request: DialogueInteraction.Request,
        response: DialogueInteraction.Response,
    ) -> DialogueInteraction.Response:
        """
        Compute one turn of a dialogue from (role, summary, history).

        Stateless: the chatbot retains no information between calls.
        The full conversation context — including any pre-session
        summary, prior turns, robot speech and system updates — is
        contained in `request.history`. The response carries the next
        utterance (if any), detected intents, and an optional
        terminal-flag + role-specific `results` payload (used by
        role handlers like __ask__ to signal natural completion).
        """
        try:
            dialogue_id = UUID(bytes=bytes(request.dialogue_id.uuid))
        except Exception as exc:
            response.error_msg = f"malformed dialogue_id: {exc}"
            return response

        role = request.role.name or DialogueRole.DEFAULT_ROLE
        role_configuration = request.role.configuration or "{}"
        summary = request.summary or ""
        history = list(request.history)

        if not history:
            response.error_msg = (
                "DialogueInteraction called with empty history"
            )
            self.get_logger().warn(response.error_msg)
            return response

        with self._counters_lock:
            self._nb_requests += 1

        handler = handler_for_role(role, role_configuration)
        user_id = last_user_speaker(history)

        system_prompt = render_system_prompt(
            self._system_prompt_tpl,
            robot_name=self._robot_name,
            role=role,
            user_id=user_id,
            action_list=self.make_action_list(),
            environment=self.get_environment_description(),
            handler=handler,
        )
        messages = build_llm_messages(
            system_prompt=system_prompt,
            summary=summary,
            history=history,
        )
        messages = trim_messages(messages, self._max_history_turns)

        self.get_logger().info(
            f"DialogueInteraction id={dialogue_id} role={role!r} "
            f"user={user_id!r} history_len={len(history)}"
        )

        # Verbose dump of the full pre- and post-translation views, gated
        # at DEBUG so a 50-turn conversation doesn't bloat normal runs.
        # The `if` short-circuits the string formatting when DEBUG is off
        # (rclpy's log call itself would no-op, but the f-strings here
        # are non-trivial). Enable with e.g.
        # `--ros-args --log-level chatbot_llm:=debug`.
        if self.get_logger().get_effective_level() <= LoggingSeverity.DEBUG:
            inbound_dump = "\n".join(
                f"  [{u.speaker}] {u.text!r}" for u in history
            )
            self.get_logger().debug(
                f"Inbound history ({len(history)} entries):\n{inbound_dump}"
            )
            messages_dump = "\n".join(
                f"  [{m['role']}] {m['content']!r}" for m in messages
            )
            self.get_logger().debug(
                f"Messages to LLM ({len(messages)} entries):\n{messages_dump}"
            )

        llm_response = self._llm_client.chat(messages)
        if not llm_response:
            response.error_msg = "LLM request failed"
            return response

        raw_response = extract_json_object(llm_response['message']['content'])
        self.get_logger().info(f"Raw LLM response: {raw_response}")

        json_res = parse_chatbot_response(raw_response, logger=self.get_logger())

        if json_res:
            self.get_logger().info(f"Parsed LLM response: {json_res}")
            outcome = handler.on_llm_response(json_res)
            response.response = outcome.response_text
            response.dialogue_terminal = outcome.dialogue_terminal
            response.results = outcome.results

            if json_res.user_intent is not None:
                user_intent = json_res.user_intent
                response.intents = [Intent(
                    intent=user_intent.type,
                    data=json.dumps(intent_to_dict(user_intent)),
                )]
        else:
            self.get_logger().warn(
                "Unable to process user input. Forwarding a 'RAW_USER_INPUT'"
            )
            last_text = history[-1].text if history else ""
            response.intents = [Intent(
                intent=Intent.RAW_USER_INPUT,
                source=user_id,
                modality=Intent.MODALITY_SPEECH,
                confidence=1.0,
                data=json.dumps({
                    "input": last_text,
                    "suggested_response": raw_response,
                }),
            )]

        return response

    # ------------------------------------------------------------------
    # Hard-coded environment / action stubs (unchanged).
    # ------------------------------------------------------------------

    def make_action_list(self) -> str:
        """Return the hard-coded list of robot actions for the system prompt."""
        return """
        - SAY: generic action to tell something to the user.
        - GREET: say hello to the user. Parameters: user_id.
        - GO_TO: move to a specific location. Parameters: location.
        - PICK_OBJECT: pick an object. Parameters: object_id.
        - PLACE_OBJECT: place an object. Parameters: object_id, location.
        """

    def get_environment_description(self) -> str:
        """Return the hard-coded environment description for the system prompt."""
        return """
        - locations: desk_1, kitchen, kitchen_table.
        - facts:
            - apple1 isOn kitchen_table
            - book1 isOn desk_1
        """

    def on_get_supported_locales(self, request, response):
        """Return the empty list; the LLM is locale-agnostic."""
        response.locales = []
        return response

    def on_set_default_locale_goal(self, goal_request):
        """Accept any locale; the LLM is locale-agnostic."""
        return GoalResponse.ACCEPT

    def on_set_default_locale_exec(self, goal_handle):
        """Nothing to do here. Always mark as succeeded."""
        result = SetLocale.Result()
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load parameters and build the LLM client. No services yet."""
        self._robot_name = self.get_parameter('robot_name').value
        self._system_prompt_tpl = Template(
            self.get_parameter('system_prompt').value
        )
        self._max_history_turns = self.get_parameter('max_history_turns').value

        self._llm_client = LLMClient(
            server=self.get_parameter('server_url').value,
            model=self.get_parameter('model').value,
            api_key=self.get_parameter_or('api_key', None).value,
            timeout=self.get_parameter('request_timeout').value,
            response_schema=chatbot_response_schema(),
            logger=self.get_logger(),
        )

        self.get_logger().info(
            f"I will connect to the LLM server on {self._llm_client.server}."
        )

        # configure and start diagnostics publishing
        with self._counters_lock:
            self._nb_requests = 0
            self._nb_prepared = 0
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self._diag_timer = self.create_timer(1., self.publish_diagnostics)

        # start advertising supported locales
        self._get_supported_locales_server = self.create_service(
            GetLocales, "~/get_supported_locales", self.on_get_supported_locales
        )

        self._set_default_locale_server = ActionServer(
            self, SetLocale, "~/set_default_locale",
            goal_callback=self.on_set_default_locale_goal,
            execute_callback=self.on_set_default_locale_exec,
        )

        self.get_logger().info("Chatbot chatbot_llm is configured, but not yet active")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Bring up the two chatbot-facing services."""
        cbg = ReentrantCallbackGroup()
        self._prepare_dialogue_srv = self.create_service(
            PrepareDialogue,
            '/chatbot/prepare_dialogue',
            self.on_prepare_dialogue,
            callback_group=cbg,
        )
        self._dialogue_interaction_srv = self.create_service(
            DialogueInteraction,
            '/chatbot/dialogue_interaction',
            self.on_dialogue_interaction,
            callback_group=cbg,
        )

        self.get_logger().info("Chatbot chatbot_llm is active and running")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Tear down the services brought up by on_activate."""
        self.get_logger().info("Stopping chatbot...")

        self.destroy_service(self._prepare_dialogue_srv)
        self.destroy_service(self._dialogue_interaction_srv)
        self._prepare_dialogue_srv = None
        self._dialogue_interaction_srv = None

        self.get_logger().info("Chatbot chatbot_llm is stopped (inactive)")
        return super().on_deactivate(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Tear down configuration-time resources."""
        self.get_logger().info('Shutting down chatbot_llm node.')
        self.destroy_timer(self._diag_timer)
        self.destroy_publisher(self._diag_pub)

        self.destroy_service(self._get_supported_locales_server)
        self._set_default_locale_server.destroy()

        self.get_logger().info("Chatbot chatbot_llm finalized.")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def publish_diagnostics(self):
        """Publish a single OK status with request counters."""
        with self._counters_lock:
            nb_requests = self._nb_requests
            nb_prepared = self._nb_prepared

        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name="/intent_extractor_chatbot_llm",
            message="chatbot chatbot_llm is running",
            values=[
                KeyValue(key="Module name", value="chatbot_llm"),
                KeyValue(
                    key="Current lifecycle state",
                    value=self._state_machine.current_state[1],
                ),
                KeyValue(key="llm server", value=self._llm_client.server),
                KeyValue(key="llm model", value=self._llm_client.model),
                KeyValue(
                    key="# prepare_dialogue calls since start",
                    value=str(nb_prepared),
                ),
                KeyValue(
                    key="# dialogue_interaction calls since start",
                    value=str(nb_requests),
                ),
            ],
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self._diag_pub.publish(arr)
