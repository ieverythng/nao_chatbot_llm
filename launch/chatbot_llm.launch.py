# Copyright (c) 2026 Juan Beck. All rights reserved.
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
from launch.actions import ExecuteProcess
from launch_pal import get_pal_configuration
from launch_ros.actions import LifecycleNode
from launch_ros.parameter_descriptions import ParameterValue

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


def _lifecycle_bootstrap_script(node_name: str, timeout_sec: int = 30) -> str:
    normalized_name = f'/{str(node_name).lstrip("/")}'
    return f"""
node_name="{normalized_name}"
deadline=$((SECONDS + {max(1, int(timeout_sec))}))
while true; do
  state="$(ros2 lifecycle get "$node_name" 2>/dev/null | awk '{{print $1}}')"
  case "$state" in
    active)
      exit 0
      ;;
    inactive)
      ros2 lifecycle set "$node_name" activate >/dev/null 2>&1 || true
      ;;
    unconfigured)
      ros2 lifecycle set "$node_name" configure >/dev/null 2>&1 || true
      ;;
    finalized|errorprocessing)
      echo "lifecycle bootstrap failed for $node_name: state=$state" >&2
      exit 1
      ;;
  esac
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "lifecycle bootstrap timed out for $node_name (last_state=${{state:-unknown}})" >&2
    exit 1
  fi
  sleep 0.2
done
""".strip()


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
    node_name = 'chatbot_llm'
    ld = LaunchDescription()

    # automatically fetch the start parameters for this node,
    # using defaults installed with the node, as well as possible
    # user overrides
    config = get_pal_configuration(pkg=pkg, node=node_name, ld=ld)

    node = LifecycleNode(
        package=pkg,
        executable='start_node',
        namespace='',
        name=node_name,
        parameters=_coerce_string_parameters(config['parameters']),
        remappings=config['remappings'],
        arguments=config['arguments'],
        output='both', emulate_tty=True,
        )

    ld.add_action(node)
    ld.add_action(
        ExecuteProcess(
            cmd=['bash', '-lc', _lifecycle_bootstrap_script(node_name)],
            output='screen',
        )
    )

    return ld
