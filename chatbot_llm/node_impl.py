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
from string import Template
from uuid import UUID

from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, GoalResponse
from hri_actions_msgs.msg import Intent
from i18n_msgs.action import SetLocale
from i18n_msgs.srv import GetLocales
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from chatbot_msgs.action import Dialogue
from chatbot_msgs.srv import DialogueInteraction
from rclpy.action import CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup

from .dialogue_state import Dialogue as DialogueState
from .dialogue_state import DialoguesRegistry
from .llm_client import LLMClient
from .response_parser import (
    chatbot_response_schema,
    extract_json_object,
    intent_to_dict,
    parse_chatbot_response,
)
from .role_handlers import handler_for_role


# Sentinel values defined in chatbot_msgs/DialogueInteraction.srv. The
# srv module exposes them as class attributes on the Request type.
_SYSTEM_USER_ID = DialogueInteraction.Request.SYSTEM_USER_ID
_ASSISTANT_USER_ID = DialogueInteraction.Request.ASSISTANT_USER_ID


class LLMChatbot(Node):
    """
    Implementation of chatbot_llm.

    This is the main class for the node. It is a ROS2  node that uses the
    lifecycle feature of ROS 2 to manage its states.

    The purpose of this node is to recognise *user intents* using a LLM.
    It relies on the OpenAI REST API to interface with a LLM: you can use
    a local server like `ollama` for edge computation, or a cloud-based service
    like ChatGPT (ie OpenAI API).

    Use the parameters `server_url` and `model` to configure the LLM server.
    """

    def __init__(self) -> None:
        """Construct the node."""
        super().__init__('intent_extractor_chatbot_llm')

        # Declare ROS parameters. Should mimick the one listed in config/00-defaults.yaml
        self.declare_parameter(
            'server_url', "http://localhost:11434",
            ParameterDescriptor(description='URL of the OpenAI-compatible LLM server')
        )
        self.declare_parameter(
            'model', "llama3.2",
            ParameterDescriptor(description='LLM model to use')
        )
        self.declare_parameter(
            'api_key', "",
            ParameterDescriptor(description='API key to use for the LLM server, if any.')
        )
        self.declare_parameter(
            'system_prompt', "You are a helpful interactive robot.",
            ParameterDescriptor(description='System prompt to use for the LLM')
        )
        self.declare_parameter(
            'robot_name', "robot",
            ParameterDescriptor(description='Name of the robot, substituted into $robot_name in the system prompt')
        )
        self.declare_parameter(
            'request_timeout', 30.0,
            ParameterDescriptor(description='Timeout (in seconds) for HTTP requests to the LLM server')
        )
        self.declare_parameter(
            'max_history_turns', 10,
            ParameterDescriptor(description='Maximum number of user/assistant turn pairs kept in the conversation history (the system prompt is always preserved)')
        )

        self.get_logger().info("Initialising...")

        self._dialogue_start_action = None
        self._dialogue_interaction_srv = None
        self._get_supported_locales_server = None
        self._set_default_locale_server = None

        self._diag_pub = None
        self._diag_timer = None

        self._llm_client = None
        self._system_prompt_tpl = None
        self._robot_name = None
        self._max_history_turns = None

        # Multi-dialogue state. The registry owns the single lock that
        # guards the dialogue dict and all per-dialogue fields. It is
        # never held across LLM HTTP round-trips: we snapshot history
        # under the lock, drop it for the request, re-acquire it to
        # commit any history extension that the call produced.
        self._dialogues = DialoguesRegistry()
        self._nb_requests = 0

        self.get_logger().info('Chatbot chatbot_llm started, but not yet configured.')

    def _trim_history(self, dialogue: DialogueState) -> None:
        """Bound a dialogue's `msgs_history` while keeping the system prompt at index 0."""
        max_msgs = 1 + 2 * self._max_history_turns
        if len(dialogue.msgs_history) > max_msgs:
            dialogue.msgs_history = (
                [dialogue.msgs_history[0]] + dialogue.msgs_history[-(max_msgs - 1):]
            )

    def _render_system_prompt(self, dialogue: DialogueState, user_id: str) -> dict:
        """Render the system prompt template for `dialogue`, with the role handler's extension."""
        rendered = self._system_prompt_tpl.safe_substitute(
            user_id=user_id,
            robot_name=self._robot_name,
            role=dialogue.role,
            action_list=self.make_action_list(),
            environment=self.get_environment_description(),
        )
        extension = handler_for_role(dialogue.role, dialogue).system_prompt_extension()
        if extension:
            rendered = rendered + "\n\n" + extension
        return {'role': 'system', 'content': rendered}

    def on_dialog_goal(self, goal: Dialogue.Goal):
        """Accept any non-empty role name. dialogue_manager owns role semantics."""
        if not goal.role.name:
            self.get_logger().warn(
                "Rejecting start_dialogue goal: empty role name."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_dialog_accept(self, handle: ServerGoalHandle):
        """Create a new Dialogue, register it, and start its execute callback."""
        dialogue_id = UUID(bytes=bytes(handle.goal_id.uuid))
        dialogue = DialogueState(
            id=dialogue_id,
            role=handle.request.role.name,
            role_configuration=handle.request.role.configuration,
        )
        if not self._dialogues.add(dialogue):
            self.get_logger().error(
                f"Refusing to register dialogue {dialogue_id}: id already in use."
            )
            handle.abort()
            return
        handle.execute()

    def on_dialog_cancel(self, handle: ServerGoalHandle):
        """Accept the cancel request iff we know about the dialogue."""
        dialogue_id = UUID(bytes=bytes(handle.goal_id.uuid))
        return CancelResponse.ACCEPT if dialogue_id in self._dialogues else CancelResponse.REJECT

    def on_dialog_execute(self, handle: ServerGoalHandle):
        """Block until the dialogue ends (external cancel or role-driven conclusion)."""
        dialogue_id = UUID(bytes=bytes(handle.goal_id.uuid))
        dialogue = self._dialogues.get(dialogue_id)
        if dialogue is None:
            handle.abort()
            return Dialogue.Result(error_msg='Dialogue not registered')

        self.get_logger().info(
            f"Starting '{handle.request.role.name}' dialogue with id {dialogue_id}"
        )

        try:
            while handle.is_active:
                # Wait for either a terminal result (set by on_dialogue_interaction
                # via the role handler) or a cancel request. The Event lets us
                # avoid a tight polling loop while still checking
                # is_cancel_requested at a reasonable cadence.
                dialogue.done.wait(timeout=0.1)
                if handle.is_cancel_requested:
                    handle.canceled()
                    return Dialogue.Result(error_msg='Dialogue cancelled')
                with self._dialogues.lock:
                    result = dialogue.result
                if result:
                    if result.error_msg:
                        handle.abort()
                    else:
                        handle.succeed()
                    return result
            return Dialogue.Result(error_msg='Dialogue execution interrupted')
        finally:
            self.get_logger().info(f"Dialogue {dialogue_id} is finished")
            self._dialogues.remove(dialogue_id)

    def on_dialogue_interaction(self,
                                chatbot_request: DialogueInteraction.Request,
                                chatbot_response: DialogueInteraction.Response):
        """Route one event into the appropriate dialogue's history."""
        user_id = chatbot_request.user_id
        input = chatbot_request.input
        response_expected = chatbot_request.response_expected
        dialogue_id = UUID(bytes=bytes(chatbot_request.dialogue_id.uuid))

        dialogue = self._dialogues.get(dialogue_id)
        if dialogue is None:
            error_msg = (
                f"Received a dialogue interaction for an unknown dialogue id: {dialogue_id}"
            )
            self.get_logger().error(error_msg)
            chatbot_response.error_msg = error_msg
            return chatbot_response

        # Route by user_id. dialogue_manager is authoritative for everything
        # that ends up in the LLM history: real user turns, the robot's
        # already-spoken assistant turns (echoed back with __assistant__)
        # and system/world updates (__system__).
        if user_id == _ASSISTANT_USER_ID:
            self._append_history(dialogue, 'assistant', input)
            return chatbot_response
        if user_id == _SYSTEM_USER_ID:
            self._append_history(dialogue, 'system', input)
            return chatbot_response

        if not user_id:
            user_id = "anonymous_user"

        self.get_logger().info(
            f"input for dialogue {dialogue_id} from {user_id}: {input}"
        )

        # Real user turn: append to the dialogue's history and snapshot the
        # message list for the LLM call. The system prompt is (re)rendered
        # under the lock so each turn sees the current user_id.
        with self._dialogues.lock:
            self._nb_requests += 1
            system_prompt_msg = self._render_system_prompt(dialogue, user_id)
            if not dialogue.msgs_history:
                dialogue.msgs_history = [system_prompt_msg]
            else:
                dialogue.msgs_history[0] = system_prompt_msg
            dialogue.msgs_history.append({
                'role': 'user',
                'content': f'{user_id} "{input}"',
            })
            self._trim_history(dialogue)
            messages_snapshot = list(dialogue.msgs_history)

        if not response_expected:
            return chatbot_response

        llm_response = self._llm_client.chat(messages_snapshot)
        if not llm_response:
            chatbot_response.error_msg = "LLM request failed"
            return chatbot_response

        raw_response = extract_json_object(llm_response['message']['content'])
        self.get_logger().info(f"Raw LLM response: {raw_response}")

        json_res = parse_chatbot_response(raw_response, logger=self.get_logger())

        if json_res:
            self.get_logger().info(f"Parsed LLM response: {json_res}")

            # NOTE: we deliberately do NOT append the LLM's proposed verbal_ack
            # to the dialogue's history here. dialogue_manager is the
            # authority on what the robot actually said: once the Say
            # sub-skill plays the utterance, dialogue_manager will call
            # back into this service with user_id=__assistant__ and the
            # actually-spoken text — and that round-trip is what extends
            # the LLM history. If Say is preempted or the text gets
            # markup-stripped, the LLM context reflects reality rather
            # than the LLM's proposal.
            handler = handler_for_role(dialogue.role, dialogue)
            outcome = handler.on_llm_response(json_res)
            if outcome.response_text:
                chatbot_response.response = outcome.response_text
            if outcome.terminal_result is not None:
                with self._dialogues.lock:
                    dialogue.result = outcome.terminal_result
                dialogue.done.set()

            if json_res.user_intent is not None:
                user_intent = json_res.user_intent
                chatbot_response.intents = [Intent(
                    intent=user_intent.type,
                    data=json.dumps(intent_to_dict(user_intent))
                )]

        else:
            self.get_logger().warn("Unable to process user input. Forwarding a 'RAW_USER_INPUT'")
            chatbot_response.intents = [Intent(
                intent=Intent.RAW_USER_INPUT,
                source=user_id,
                modality=Intent.MODALITY_SPEECH,
                confidence=1.0,
                data=json.dumps({
                    "input": input,
                    "suggested_response": raw_response,
                }),
            )]

        return chatbot_response

    def _append_history(self, dialogue: DialogueState, role: str, content: str) -> None:
        """Append one entry to `dialogue`'s history under the registry lock, then trim."""
        with self._dialogues.lock:
            # Ensure the system prompt slot exists; we don't re-render it here
            # because there's no user_id context. It will get refreshed on the
            # next real user turn via _render_system_prompt.
            if not dialogue.msgs_history:
                dialogue.msgs_history.append(
                    self._render_system_prompt(dialogue, user_id="anonymous_user")
                )
            dialogue.msgs_history.append({'role': role, 'content': content})
            self._trim_history(dialogue)

    def make_action_list(self) -> str:
        # List all the actions available to the robot, with their corresponding
        # intents.
        #
        # Here, we just hard-code a few examples.

        return """
        - SAY: generic action to tell something to the user.
        - GREET: say hello to the user. Parameters: user_id.
        - GO_TO: move to a specific location. Parameters: location.
        - PICK_OBJECT: pick an object. Parameters: object_id.
        - PLACE_OBJECT: place an object. Parameters: object_id, location.
        """

    def get_environment_description(self) -> str:
        # Describe the environment in which the robot is operating.
        #
        # Here, we just hard-code a few examples.

        return """
        - locations: desk_1, kitchen, kitchen_table.
        - facts:
            - apple1 isOn kitchen_table
            - book1 isOn desk_1
        """

    def on_get_supported_locales(self, request, response):
        response.locales = []  # the LLM can handle any
        return response

    def on_set_default_locale_goal(self, goal_request):
        return GoalResponse.ACCEPT

    def on_set_default_locale_exec(self, goal_handle):
        """Nothing to do here. Always mark as succeeded."""
        result = SetLocale.Result()
        goal_handle.succeed()
        return result

    #################################
    #
    # Lifecycle transitions callbacks
    #
    def on_configure(self, state: State) -> TransitionCallbackReturn:

        self._robot_name = self.get_parameter('robot_name').value
        self._system_prompt_tpl = Template(self.get_parameter('system_prompt').value)
        self._max_history_turns = self.get_parameter('max_history_turns').value

        self._llm_client = LLMClient(
            server=self.get_parameter('server_url').value,
            model=self.get_parameter('model').value,
            api_key=self.get_parameter_or('api_key', None).value,
            timeout=self.get_parameter('request_timeout').value,
            response_schema=chatbot_response_schema(),
            logger=self.get_logger(),
        )

        self.get_logger().info(f"I will connect to the LLM server on {self._llm_client.server}.")

        # configure and start diagnostics publishing
        self._nb_requests = 0
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self._diag_timer = self.create_timer(1., self.publish_diagnostics)

        # start advertising supported locales
        self._get_supported_locales_server = self.create_service(
            GetLocales, "~/get_supported_locales", self.on_get_supported_locales)

        self._set_default_locale_server = ActionServer(
            self, SetLocale, "~/set_default_locale",
            goal_callback=self.on_set_default_locale_goal,
            execute_callback=self.on_set_default_locale_exec)

        self.get_logger().info("Chatbot chatbot_llm is configured, but not yet active")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Activate the node.

        You usually want to do the following in this state:
        - Create and start any timers performing periodic tasks
        - Start processing data, and accepting action goals, if any

        """
        self._dialogue_start_action = ActionServer(
            self, Dialogue, '/chatbot/start_dialogue',
            execute_callback=self.on_dialog_execute,
            goal_callback=self.on_dialog_goal,
            handle_accepted_callback=self.on_dialog_accept,
            cancel_callback=self.on_dialog_cancel,
            callback_group=ReentrantCallbackGroup())
        self._dialogue_interaction_srv = self.create_service(
            DialogueInteraction, '/chatbot/dialogue_interaction', self.on_dialogue_interaction)

        self.get_logger().info("Chatbot chatbot_llm is active and running")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Tear down the action server and service brought up by on_activate."""
        self.get_logger().info("Stopping chatbot...")

        self._dialogue_start_action.destroy()
        self.destroy_service(self._dialogue_interaction_srv)

        self.get_logger().info("Chatbot chatbot_llm is stopped (inactive)")
        return super().on_deactivate(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """
        Shutdown the node, after a shutting-down transition is requested.

        :return: The state machine either invokes a transition to the
            "finalized" state or stays in the current state depending on the
            return value.
            TransitionCallbackReturn.SUCCESS transitions to "finalized".
            TransitionCallbackReturn.FAILURE remains in current state.
            TransitionCallbackReturn.ERROR or any uncaught exceptions to
            "errorprocessing"
        """
        self.get_logger().info('Shutting down chatbot_llm node.')
        self.destroy_timer(self._diag_timer)
        self.destroy_publisher(self._diag_pub)

        self.destroy_service(self._get_supported_locales_server)
        self._set_default_locale_server.destroy()

        self.get_logger().info("Chatbot chatbot_llm finalized.")
        return TransitionCallbackReturn.SUCCESS

    #################################

    def publish_diagnostics(self):
        with self._dialogues.lock:
            nb_requests = self._nb_requests
        active_ids = self._dialogues.ids()

        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name="/intent_extractor_chatbot_llm",
            message="chatbot chatbot_llm is running",
            values=[
                KeyValue(key="Module name", value="chatbot_llm"),
                KeyValue(key="Current lifecycle state",
                         value=self._state_machine.current_state[1]),
                KeyValue(key="llm server", value=self._llm_client.server),
                KeyValue(key="llm model", value=self._llm_client.model),
                KeyValue(key="# active dialogues", value=str(len(active_ids))),
                KeyValue(key="active dialogue ids",
                         value=", ".join(str(uid) for uid in active_ids)),
                KeyValue(key="# requests since start", value=str(nb_requests)),
            ],
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self._diag_pub.publish(arr)
