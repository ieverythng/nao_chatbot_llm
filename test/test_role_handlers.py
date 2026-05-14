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

"""Unit tests for chatbot_llm.role_handlers."""

from uuid import uuid4

from chatbot_llm.dialogue_state import Dialogue as DialogueState
from chatbot_llm.response_parser import ChatbotResponse
from chatbot_llm.role_handlers import (
    DefaultRoleHandler,
    handler_for_role,
)


def _dialogue(role: str = "__default__", configuration: str = "{}") -> DialogueState:
    return DialogueState(id=uuid4(), role=role, role_configuration=configuration)


class TestDefaultRoleHandler:
    """Tests for the __default__ role handler."""

    def test_empty_system_prompt_extension(self):
        """The default handler contributes nothing extra to the system prompt."""
        h = DefaultRoleHandler(_dialogue())
        assert h.system_prompt_extension() == ""

    def test_surface_verbal_ack_verbatim(self):
        """on_llm_response returns the verbal_ack and never self-closes."""
        h = DefaultRoleHandler(_dialogue())
        outcome = h.on_llm_response(ChatbotResponse(verbal_ack="hello"))
        assert outcome.response_text == "hello"
        assert outcome.terminal_result is None

    def test_missing_verbal_ack_becomes_empty_string(self):
        """A None verbal_ack is coerced to an empty response, not propagated as None."""
        h = DefaultRoleHandler(_dialogue())
        outcome = h.on_llm_response(ChatbotResponse(verbal_ack=None))
        assert outcome.response_text == ""
        assert outcome.terminal_result is None


class TestHandlerDispatch:
    """Tests for handler_for_role."""

    def test_default_role_dispatches_to_default_handler(self):
        """A __default__ role yields a DefaultRoleHandler bound to the dialogue."""
        d = _dialogue("__default__")
        h = handler_for_role(d.role, d)
        assert isinstance(h, DefaultRoleHandler)
        assert h.dialogue is d

    def test_unknown_role_falls_back_to_default(self):
        """Unknown role names fall back to the default handler."""
        d = _dialogue("__some_custom_role__")
        h = handler_for_role(d.role, d)
        assert isinstance(h, DefaultRoleHandler)
