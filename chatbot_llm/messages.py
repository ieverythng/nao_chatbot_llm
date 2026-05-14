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

"""
Build the LLM message list from a DialogueInteraction request.

dialogue_manager is the source of truth for the conversation history;
each DialogueInteraction call carries a Utterance[] array spanning the
full dialogue. This module translates that ROS-side history into the
{role: system|user|assistant} dict shape consumed by the LLM client.
"""

from string import Template
from typing import Iterable

from chatbot_msgs.msg import Utterance

from .role_handlers import RoleHandler


_SYSTEM = Utterance.SYSTEM
_ASSISTANT = Utterance.ASSISTANT


def last_user_speaker(history: Iterable, default: str = "anonymous_user") -> str:
    """Return the speaker of the last non-system, non-assistant utterance."""
    for utt in reversed(list(history)):
        if utt.speaker not in (_SYSTEM, _ASSISTANT):
            return utt.speaker or default
    return default


def render_system_prompt(
    template: Template,
    *,
    robot_name: str,
    role: str,
    user_id: str,
    action_list: str,
    environment: str,
    handler: RoleHandler,
) -> str:
    """Render the system-prompt template and append the handler's extension."""
    rendered = template.safe_substitute(
        user_id=user_id,
        robot_name=robot_name,
        role=role,
        action_list=action_list,
        environment=environment,
    )
    extension = handler.system_prompt_extension()
    if extension:
        rendered = rendered + "\n\n" + extension
    return rendered


def build_llm_messages(
    *,
    system_prompt: str,
    summary: str,
    history: Iterable,
) -> list:
    """
    Build the LLM message list from a stateless DialogueInteraction.

    Layout:
      [system: rendered system prompt]
      [system: prior-session summary]   # only if `summary` is non-empty
      ... one entry per history utterance, mapped by speaker:
            SYSTEM    -> {'role': 'system',    'content': text}
            ASSISTANT -> {'role': 'assistant', 'content': text}
            <user>    -> {'role': 'user',      'content': '<user> "<text>"'}
    """
    messages = [{"role": "system", "content": system_prompt}]
    if summary:
        messages.append({
            "role": "system",
            "content": f"Prior conversation summary:\n{summary}",
        })
    for utt in history:
        if utt.speaker == _SYSTEM:
            messages.append({"role": "system", "content": utt.text})
        elif utt.speaker == _ASSISTANT:
            messages.append({"role": "assistant", "content": utt.text})
        else:
            speaker = utt.speaker or "anonymous_user"
            messages.append({
                "role": "user",
                "content": f'{speaker} "{utt.text}"',
            })
    return messages


def trim_messages(messages: list, max_turns: int) -> list:
    """
    Bound the message list at `1 (system) + 2 * max_turns` entries.

    The leading system prompt is always preserved; the most recent
    user/assistant pairs are kept and earlier turns dropped. A
    secondary leading system message (e.g. a summary) is folded into
    the dropped tail if it would push us over the bound — the summary
    is informational, not load-bearing for the active turn.
    """
    if max_turns <= 0:
        return messages
    max_msgs = 1 + 2 * max_turns
    if len(messages) <= max_msgs:
        return messages
    return [messages[0]] + messages[-(max_msgs - 1):]
