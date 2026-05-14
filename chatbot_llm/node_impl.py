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

import requests
import json
import threading
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

from pydantic import BaseModel
from typing import Literal
from hri_actions_msgs.msg import Intent

# Define the data models for the chatbot response and the user intent
class IntentModel(BaseModel):
    type: Literal[Intent.BRING_OBJECT,
                  Intent.GRAB_OBJECT,
                  Intent.PLACE_OBJECT,
                  Intent.GUIDE,
                  Intent.MOVE_TO,
                  Intent.SAY,
                  Intent.GREET,
                  Intent.START_ACTIVITY,
                  ]
    object: str | None
    recipient: str | None
    input: str | None
    goal: str | None

class ChatbotResponse(BaseModel):
    verbal_ack: str | None
    user_intent: IntentModel | None


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

        self.get_logger().info("Initialising...")

        self._dialogue_start_action = None
        self._dialogue_interaction_srv = None
        self._get_supported_locales_server = None
        self._set_default_locale_server = None

        self._timer = None
        self._diag_pub = None
        self._diag_timer = None

        self._llm_server = None
        self._llm_model = None
        self._api_key = None
        self._request_timeout = None
        self._system_prompt_tpl = None
        self._robot_name = None

        self._nb_requests = 0
        self._msgs_history = []
        self._dialogue_id = None
        self._dialogue_result = None
        self._dialogue_done = threading.Event()

        self.get_logger().info('Chatbot chatbot_llm started, but not yet configured.')

    def perform_request(self, server, model, messages, api_key=None):
        headers = {
                   'Content-Type': 'application/json',
                  }

        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"

        url = f"{server}/v1/chat/completions"

        json_data = json.dumps({
            "model": model,
            "messages": messages,
            "think": False,

            "format": ChatbotResponse.model_json_schema() if hasattr(ChatbotResponse, "model_json_schema") else ChatbotResponse.schema_json()
        })

        response = None
        try:
            self.get_logger().info("Sending request to LLM and waiting for response...")
            response = requests.post(url, headers=headers, data=json_data,
                                     timeout=self._request_timeout)
            response.raise_for_status()
            return response.json()['choices'][0]
        except requests.exceptions.RequestException as e:
            error_msg = "Unable to decode error message from server"
            if response is not None:
                try:
                    error_msg = response.json()['error']['message']
                except (ValueError, KeyError):
                    pass
            self.get_logger().error(f"Error while sending request to {url}:\n{e}\n{error_msg}\n"
                                    "Is the server running, or the URL incorrect "
                                    f"(eg wrong port)? I was trying to connect to: {url}")
            return None

    def preprocess_llm_response(self, raw_text):

        # extract the substring between the first
        # opening and closing curly braces (accounting for nested braces).
        start_idx = 0
        end_idx = len(raw_text) - 1
        nested = 0
        for i, c in enumerate(raw_text):
            if c == '{':
                start_idx = i
                break
        for i in range(start_idx, len(raw_text)):
            if raw_text[i] == '{':
                nested += 1
            elif raw_text[i] == '}':
                nested -= 1
                if nested == 0:
                    end_idx = i
                    break

        text = raw_text[start_idx:end_idx+1]

        # the LLM tends to remove spaces before colons, which cause
        # invalid YAML parsing.
        text = text.replace(":", ": ")

        return text

    def _render_system_prompt(self, user_id: str) -> dict:
        """Render the system prompt template with the current user_id."""
        rendered = self._system_prompt_tpl.safe_substitute(
            user_id=user_id,
            robot_name=self._robot_name,
            action_list=self.make_action_list(),
            environment=self.get_environment_description(),
        )
        return {'role': 'system', 'content': rendered}

    def on_dialog_goal(self, goal: Dialogue.Goal):
        # Check if the goal is valid and the node is able to accept it
        #
        # For simplicity, we allow only one dialogue at a time.
        # You might want to change this to allow multiple dialogues at the same time.
        #
        # We also check if the dialogue role is supported by the chatbot.
        # In this example, we only support only the "__default__" role.

        if self._dialogue_id or goal.role.name != '__default__':
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_dialog_accept(self, handle: ServerGoalHandle):
        self._dialogue_id =  UUID(bytes=bytes(handle.goal_id.uuid))
        self._dialogue_result = None
        self._dialogue_done.clear()
        handle.execute()

    def on_dialog_cancel(self, handle: ServerGoalHandle):
        if self._dialogue_id:
            return CancelResponse.ACCEPT
        else:
            return CancelResponse.REJECT

    def on_dialog_execute(self, handle: ServerGoalHandle):
        id = UUID(bytes=bytes(handle.goal_id.uuid))
        self.get_logger().info(f"Starting '{handle.request.role.name}' dialogue with id {id}")

        try:
            while handle.is_active:
                # Wait for either a terminal result (set elsewhere) or a cancel
                # request. The Event lets us avoid a tight polling loop while
                # still checking is_cancel_requested at a reasonable cadence.
                self._dialogue_done.wait(timeout=0.1)
                if handle.is_cancel_requested:
                    handle.canceled()
                    return Dialogue.Result(error_msg='Dialogue cancelled')
                if self._dialogue_result:
                    if self._dialogue_result.error_msg:
                        handle.abort()
                    else:
                        handle.succeed()
                    return self._dialogue_result
            return Dialogue.Result(error_msg='Dialogue execution interrupted')
        finally:
            self.get_logger().info(f"Dialogue {id} is finished")
            self._dialogue_id = None
            self._dialogue_result = None
            self._dialogue_done.clear()

    def on_dialogue_interaction(self,
                                chatbot_request: DialogueInteraction.Request,
                                chatbot_response: DialogueInteraction.Response):
        user_id = chatbot_request.user_id
        input = chatbot_request.input
        response_expected = chatbot_request.response_expected
        id = UUID(bytes=bytes(chatbot_request.dialogue_id.uuid))
        if id != self._dialogue_id:
            error_msg = f"Received a dialogue interaction for an unknown dialogue id: {id}"
            self.get_logger().error(error_msg)
            chatbot_response.error_msg = error_msg
            return chatbot_response

        # Implement here the logic to process the natural text input
        #
        # You might want to:
        # - recognize and return the user intent (if any), and map it to the
        #   Intent.msg semantics.
        # - return a suggested response to the user (if any) - return a result
        #   (if the dialogue has exhausted its role), which would close the
        #   dialogue
        #
        # For now, we just try to recognize a greeting, and close the dialogue
        # if requested
        suggested_response = ""
        intent = None

        self.get_logger().info(f"input from {user_id}: {input}")
        self._nb_requests += 1

        if not user_id:
            user_id = "anonymous_user"

        system_prompt_msg = self._render_system_prompt(user_id)
        if not self._msgs_history:
            self._msgs_history = [system_prompt_msg]
        else:
            # Refresh the system prompt so user_id stays current across turns.
            self._msgs_history[0] = system_prompt_msg
        self._msgs_history.append(
            {
                'role': 'user',
                'content': f'{user_id} "{input}"',
            }
        )

        if not response_expected:
            return chatbot_response

        llm_response = self.perform_request(
            server=self._llm_server,
            model=self._llm_model,
            messages=self._msgs_history,
            api_key=self._api_key
        )
        if not llm_response:
            chatbot_response.error_msg = "LLM request failed"
            return chatbot_response

        raw_response = self.preprocess_llm_response(llm_response['message']['content'])
        self.get_logger().info(f"Raw LLM response: {raw_response}")

        json_res = None
        if hasattr(ChatbotResponse, "model_validate_json"):
            json_res = ChatbotResponse.model_validate_json(raw_response)
        else:
            try:
                json_res = json.loads(raw_response)
                # check that the json response is valid according to the ChatbotResponse model
                json_res = ChatbotResponse(**json_res)

            except json.decoder.JSONDecodeError as e:
                self.get_logger().warn(f"Malformed JSON response: {e}")
            except Exception as e:
                self.get_logger().warn(f"LLM response does not match expected format: {e}")


        if json_res:
            self.get_logger().info(f"Parsed LLM response: {json_res}")

            if json_res.verbal_ack is not None:
                verbal_ack = json_res.verbal_ack
                # if we have a verbal acknowledgement, add it to the dialogue history,
                # and send it to the user
                self._msgs_history.append({"role": "assistant", "content": verbal_ack})
                chatbot_response.response = verbal_ack

            if json_res.user_intent is not None:
                user_intent = json_res.user_intent
                intent_dict = (
                    user_intent.model_dump()
                    if hasattr(user_intent, "model_dump")
                    else user_intent.dict()
                )
                chatbot_response.intents = [Intent(
                    intent=user_intent.type,
                    data=json.dumps(intent_dict)
                )]

        else:
            self.get_logger().warn("Unable to process user input. Forwarding a 'RAW_USER_INPUT'")
            intent = Intent(intent=Intent.RAW_USER_INPUT,
                            source=user_id,
                            modality=Intent.MODALITY_SPEECH,
                            confidence=1.0,
                            data=json.dumps({
                                "input": input,
                                "suggested_response": raw_response,
                            }))

            chatbot_response.response = suggested_response
            chatbot_response.intents = [intent]

        return chatbot_response

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

        self._llm_server = self.get_parameter('server_url').value
        self._llm_model = self.get_parameter('model').value
        self._api_key = self.get_parameter_or('api_key', None).value
        self._request_timeout = self.get_parameter('request_timeout').value
        self._robot_name = self.get_parameter('robot_name').value
        self._system_prompt_tpl = Template(self.get_parameter('system_prompt').value)

        self.get_logger().info(f"I will connect to the LLM server on {self._llm_server}.")

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

        # Define a timer that fires every second to call the run function
        timer_period = 1  # in sec
        self._timer = self.create_timer(timer_period, self.run)

        self.get_logger().info("Chatbot chatbot_llm is active and running")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop the timer to stop calling the `run` function (main task of your application)."""
        self.get_logger().info("Stopping chatbot...")

        self.destroy_timer(self._timer)
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

        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name="/intent_extractor_chatbot_llm",
            message="chatbot chatbot_llm is running",
            values=[
                KeyValue(key="Module name", value="chatbot_llm"),
                KeyValue(key="Current lifecycle state",
                         value=self._state_machine.current_state[1]),
                KeyValue(key="llm server", value=self._llm_server),
                KeyValue(key="llm model", value=self._llm_model),
                KeyValue(key="# requests since start", value=str(self._nb_requests)),
            ],
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self._diag_pub.publish(arr)

    def run(self) -> None:
        """
        Background task of the chatbot.

        For now, we do not need to do anything here, as the chatbot is
        event-driven, and the `on_user_input` callback is called when a new
        user input is received.
        """
        pass
