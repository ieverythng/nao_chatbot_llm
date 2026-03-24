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

####################################################################
####################################################################
#
# AUTO-GENERATED Application
#
# you should not need to modify this file.
#
####################################################################

import rclpy

import chatbot_llm.node_impl
from rclpy.executors import MultiThreadedExecutor


def main():
    """Start the lifecycle chatbot backend with a multithreaded executor."""
    rclpy.init()

    node = chatbot_llm.node_impl.LLMChatbot()
    node_executor = MultiThreadedExecutor()
    node_executor.add_node(node)

    try:
        node_executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
