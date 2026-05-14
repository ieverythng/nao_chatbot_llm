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

import json
from uuid import uuid4

from chatbot_llm.dialogue_state import Dialogue as DialogueState
from chatbot_llm.response_parser import ChatbotResponse
from chatbot_llm.role_handlers import (
    AskRoleHandler,
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

    def test_ask_role_dispatches_to_ask_handler(self):
        """A __ask__ role yields an AskRoleHandler."""
        d = _dialogue("__ask__")
        h = handler_for_role(d.role, d)
        assert isinstance(h, AskRoleHandler)

    def test_unknown_role_falls_back_to_default(self):
        """Unknown role names fall back to the default handler."""
        d = _dialogue("__some_custom_role__")
        h = handler_for_role(d.role, d)
        assert isinstance(h, DefaultRoleHandler)


class TestAskRoleHandler:
    """Tests for the __ask__ role handler."""

    @staticmethod
    def _ask_dialogue(configuration: dict) -> DialogueState:
        return _dialogue("__ask__", configuration=json.dumps(configuration))

    def test_empty_configuration_yields_empty_extension(self):
        """No schema means no prompt extension and no self-close."""
        h = AskRoleHandler(self._ask_dialogue({}))
        assert h.system_prompt_extension() == ""
        outcome = h.on_llm_response(ChatbotResponse(verbal_ack="ok"))
        assert outcome.terminal_result is None
        assert outcome.response_text == "ok"

    def test_invalid_configuration_json_is_ignored(self):
        """A malformed configuration JSON falls back to empty config."""
        d = _dialogue("__ask__", configuration="this is not json")
        h = AskRoleHandler(d)
        assert h.required_keys == []
        assert h.system_prompt_extension() == ""

    def test_extension_includes_question_and_schema(self):
        """When configuration is set, the extension mentions the question and the schema keys."""
        config = {
            "question": "What is your age?",
            "result_schema_properties": {
                "age": {"type": "integer", "minimum": 0},
            },
        }
        h = AskRoleHandler(self._ask_dialogue(config))
        ext = h.system_prompt_extension()
        assert "What is your age?" in ext
        assert '"age"' in ext
        assert h.required_keys == ["age"]

    def test_stays_open_when_extracted_missing(self):
        """If the LLM hasn't filled `extracted`, the dialogue stays open."""
        config = {"result_schema_properties": {"age": {"type": "integer"}}}
        h = AskRoleHandler(self._ask_dialogue(config))
        outcome = h.on_llm_response(ChatbotResponse(verbal_ack="please tell me your age"))
        assert outcome.terminal_result is None
        assert outcome.response_text == "please tell me your age"

    def test_stays_open_when_extracted_partial(self):
        """Missing any required key keeps the dialogue open."""
        config = {
            "result_schema_properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
            },
        }
        h = AskRoleHandler(self._ask_dialogue(config))
        partial = ChatbotResponse(
            verbal_ack="and your last name?", extracted={"first_name": "Ada"}
        )
        outcome = h.on_llm_response(partial)
        assert outcome.terminal_result is None
        assert outcome.response_text == "and your last name?"

    def test_closes_when_extracted_complete(self):
        """When `extracted` covers every required key, the dialogue self-closes."""
        config = {
            "result_schema_properties": {
                "age": {"type": "integer"},
            },
        }
        h = AskRoleHandler(self._ask_dialogue(config))
        outcome = h.on_llm_response(ChatbotResponse(verbal_ack="thanks!", extracted={"age": 42}))
        assert outcome.terminal_result is not None
        assert outcome.terminal_result.error_msg == ""
        assert json.loads(outcome.terminal_result.results) == {"age": 42}
        assert outcome.response_text == "thanks!"
