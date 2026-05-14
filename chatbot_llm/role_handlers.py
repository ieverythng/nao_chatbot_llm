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

Handlers are stateless wrt. the dialogue history: they hold only the
role configuration (parsed once at construction) and re-derive their
view of "what's been collected so far" from the parsed LLM response
each turn. dialogue_manager owns the authoritative conversation
history and ships it in full on every DialogueInteraction call.
"""

import json
from dataclasses import dataclass

from .response_parser import ChatbotResponse


@dataclass
class TurnOutcome:
    """
    Result of processing a single LLM turn for a dialogue.

    `response_text` is the text to surface in the
    DialogueInteraction.Response (typically the LLM's verbal_ack).

    `dialogue_terminal`, when True, signals that the role considers
    the dialogue naturally complete; `results` carries the role-specific
    JSON-encoded outcome (empty when not terminal). dialogue_manager
    is responsible for actually closing the dialogue.
    """

    response_text: str = ""
    dialogue_terminal: bool = False
    results: str = ""


class RoleHandler:
    """
    Base class for per-role policies.

    The default implementation in this base class is a sensible
    "passive" role: no prompt extension, surface the verbal_ack
    verbatim, never self-close.
    """

    role_name: str = ""

    def __init__(self, role_configuration: str = ""):
        """Hold the (opaque JSON) role configuration verbatim."""
        self.role_configuration = role_configuration

    def system_prompt_extension(self) -> str:
        """
        Return extra text appended to the global system prompt for this role.

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
            dialogue_terminal=False,
            results="",
        )


class DefaultRoleHandler(RoleHandler):
    """
    Handler for the __default__ role.

    Generic open-ended chat: surface the verbal_ack, never self-close.
    Identical to the base class behaviour; named for clarity in the
    dispatch table.
    """

    role_name = "__default__"


class AskRoleHandler(RoleHandler):
    """
    Handler for the __ask__ role: elicit a structured answer.

    The role configuration is a JSON object that may carry:
    - "question": the question the robot is asking (already uttered by
      the dialogue_manager before the first interaction).
    - "result_schema_properties": a JSON schema describing the fields
      the chatbot is expected to fill from the user's response.

    The handler asks the LLM (via a prompt extension) to fill those
    fields under the `extracted` key of ChatbotResponse. Once all the
    required field keys are present, the dialogue is marked terminal
    and the extracted answer is JSON-serialised into the response's
    `results`.
    """

    role_name = "__ask__"

    def __init__(self, role_configuration: str = ""):
        """Parse the role configuration JSON and cache the question/schema."""
        super().__init__(role_configuration)
        self._config = self._parse_config(role_configuration)

    @staticmethod
    def _parse_config(raw: str) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @property
    def question(self) -> str:
        """Return the question text from the role configuration, or ''."""
        value = self._config.get("question", "")
        return value if isinstance(value, str) else ""

    @property
    def result_schema_properties(self) -> dict:
        """Return the schema properties dict, or {}."""
        value = self._config.get("result_schema_properties", {})
        return value if isinstance(value, dict) else {}

    @property
    def required_keys(self) -> list:
        """Return the list of keys the chatbot must produce to close the dialogue."""
        return list(self.result_schema_properties.keys())

    def system_prompt_extension(self) -> str:
        """Instruct the LLM to extract the schema fields into `extracted`."""
        if not self.result_schema_properties:
            return ""

        parts = [
            "You are conducting an information-gathering interview with the user.",
        ]
        if self.question:
            parts.append(f"The question the user is responding to is: {self.question!r}")
        parts.append(
            "The information you must collect is described by this JSON schema:"
        )
        parts.append(json.dumps(self.result_schema_properties, indent=2))
        parts.append(
            "Until you have collected ALL the required fields, use `verbal_ack` to "
            "ask follow-up questions or to acknowledge the user. When (and only "
            "when) you have a value for every required field, set `extracted` to "
            "a JSON object containing those values; the dialogue will then end."
        )
        return "\n".join(parts)

    def on_llm_response(self, parsed: ChatbotResponse) -> TurnOutcome:
        """Mark terminal iff `extracted` covers all required keys."""
        verbal_ack = parsed.verbal_ack or ""
        extracted = parsed.extracted
        if extracted and all(k in extracted for k in self.required_keys):
            return TurnOutcome(
                response_text=verbal_ack,
                dialogue_terminal=True,
                results=json.dumps(extracted),
            )
        return TurnOutcome(
            response_text=verbal_ack,
            dialogue_terminal=False,
            results="",
        )


def handler_for_role(role: str, role_configuration: str = "") -> RoleHandler:
    """
    Return the RoleHandler to use for `role`.

    Falls back to DefaultRoleHandler for unknown roles.
    """
    cls = _HANDLERS.get(role, DefaultRoleHandler)
    return cls(role_configuration)


# Registered role handlers. Subclasses register themselves here so that
# adding a new role doesn't require touching node_impl.
_HANDLERS: dict = {
    DefaultRoleHandler.role_name: DefaultRoleHandler,
    AskRoleHandler.role_name: AskRoleHandler,
}
