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

from launch import LaunchDescription
from launch.actions import EmitEvent
from launch.actions import RegisterEventHandler
from launch.events import matches_action
from launch.event_handlers import OnProcessStart
from launch_pal import get_pal_configuration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue
from lifecycle_msgs.msg import Transition

STRING_PARAMETER_NAMES = {
    'api_key',
    'environment_description',
    'fallback_response',
    'intent_detection_mode',
    'intent_model',
    'intent_prompt_addendum',
    'model',
    'persona_prompt_path',
    'prompt_pack_path',
    'response_prompt_addendum',
    'robot_name',
    'server_url',
    'skill_catalog_packages',
    'system_prompt',
}


def _coerce_string_parameters(parameters):
    coerced = []
    for entry in parameters:
        if not isinstance(entry, dict):
            coerced.append(entry)
            continue

        coerced_entry = {}
        for key, value in entry.items():
            if key in STRING_PARAMETER_NAMES:
                coerced_entry[key] = ParameterValue(value, value_type=str)
            else:
                coerced_entry[key] = value
        coerced.append(coerced_entry)
    return coerced


def generate_launch_description():
    pkg = 'chatbot_llm'
    node = 'chatbot_llm'
    ld = LaunchDescription()

    # automatically fetch the start parameters for this node,
    # using defaults installed with the node, as well as possible
    # user overrides
    config = get_pal_configuration(pkg=pkg, node=node, ld=ld)

    node = LifecycleNode(
        package=pkg,
        executable='start_node',
        namespace='',
        name=node,
        parameters=_coerce_string_parameters(config['parameters']),
        remappings=config['remappings'],
        arguments=config['arguments'],
        output='both', emulate_tty=True,
        )

    ld.add_action(node)

    # automatically perform the lifecycle transitions to configure and activate
    # the node at startup
    ld.add_action(
        RegisterEventHandler(
            OnProcessStart(
                target_action=node,
                on_start=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(node),
                            transition_id=Transition.TRANSITION_CONFIGURE,
                        )
                    )
                ],
            )
        )
    )
    ld.add_action(
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=node,
                goal_state='inactive',
                entities=[
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(node),
                            transition_id=Transition.TRANSITION_ACTIVATE,
                        )
                    )
                ],
                handle_once=True,
            )
        )
    )

    return ld
