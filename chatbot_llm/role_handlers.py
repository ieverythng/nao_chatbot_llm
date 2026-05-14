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
Per-role policy for chatbot_llm dialogues.

A RoleHandler bundles the small role-specific concerns that the rest
of the node should not need to know about: any system-prompt extension
that should be injected for that role, and the decision (after each
LLM turn) about whether the dialogue has reached its natural
conclusion. See docs/ARCHITECTURE.md for the broader context.
"""

from dataclasses import dataclass
from typing import Optional

from chatbot_msgs.action import Dialogue

from .dialogue_state import Dialogue as DialogueState
from .response_parser import ChatbotResponse


@dataclass
class TurnOutcome:
    """
    Result of processing a single LLM turn for a dialogue.

    `response_text` is the text to surface in the
    DialogueInteraction.Response (typically the LLM's verbal_ack).

    `terminal_result`, if set, indicates the dialogue has reached its
    natural conclusion: the node will set this on the action goal and
    finalise the action. If None, the dialogue stays open.
    """

    response_text: str = ""
    terminal_result: Optional[Dialogue.Result] = None


class RoleHandler:
    """
    Base class for per-role policies. Subclasses configure behaviour.

    The default implementation in this base class is a sensible
    "passive" role: no prompt extension, surface the verbal_ack
    verbatim, never self-close.
    """

    role_name: str = ""

    def __init__(self, dialogue: DialogueState):
        """Bind the handler to its dialogue."""
        self.dialogue = dialogue

    def system_prompt_extension(self) -> str:
        """
        Return extra text appended to the global system prompt for this dialogue.

        Returns an empty string by default. Subclasses override to add
        role-specific instructions to the LLM.
        """
        return ""

    def on_llm_response(self, parsed: ChatbotResponse) -> TurnOutcome:
        """
        Decide what to do with the LLM's parsed response.

        The default implementation surfaces `verbal_ack` and never
        self-closes the dialogue.
        """
        return TurnOutcome(
            response_text=parsed.verbal_ack or "",
            terminal_result=None,
        )


class DefaultRoleHandler(RoleHandler):
    """
    Handler for the __default__ role.

    Generic open-ended chat: surface the verbal_ack, never self-close.
    Identical to the base class behaviour; named for clarity in the
    dispatch table.
    """

    role_name = "__default__"


def handler_for_role(role: str, dialogue: DialogueState) -> RoleHandler:
    """
    Return the RoleHandler to use for `role`.

    Falls back to DefaultRoleHandler for unknown roles.
    """
    cls = _HANDLERS.get(role, DefaultRoleHandler)
    return cls(dialogue)


# Registered role handlers. Subclasses register themselves here so that
# adding a new role doesn't require touching node_impl.
_HANDLERS: dict = {
    DefaultRoleHandler.role_name: DefaultRoleHandler,
}
