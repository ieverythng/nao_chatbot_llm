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

"""Unit tests for chatbot_llm.response_parser."""

import json

from chatbot_llm.response_parser import (
    extract_json_object,
    intent_to_dict,
    parse_chatbot_response,
)

from hri_actions_msgs.msg import Intent


class TestExtractJsonObject:
    """Tests for extract_json_object."""

    def test_plain_object(self):
        """Plain JSON is returned unchanged."""
        assert extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_leading_prose(self):
        """Leading prose before the JSON object is stripped."""
        assert extract_json_object('Sure, here you go: {"a": 1}') == '{"a": 1}'

    def test_trailing_prose(self):
        """Trailing prose after the JSON object is stripped."""
        assert extract_json_object('{"a": 1}. Hope that helps!') == '{"a": 1}'

    def test_nested_objects(self):
        """Nested objects are kept intact."""
        src = '{"outer": {"inner": {"deep": 1}}}'
        assert extract_json_object(src) == src

    def test_braces_inside_string_value(self):
        """Braces inside string values do not confuse the extractor."""
        src = '{"msg": "unmatched } here"}'
        assert extract_json_object(src) == src

    def test_colon_inside_string_value_preserved(self):
        """Colons inside string values are kept verbatim."""
        src = '{"url": "http://example.com:8080/path"}'
        assert extract_json_object(src) == src
        assert json.loads(extract_json_object(src))["url"] == \
            "http://example.com:8080/path"

    def test_no_json_returns_input(self):
        """When no `{` is present, the input is returned untouched."""
        assert extract_json_object("just plain text") == "just plain text"

    def test_malformed_json_returns_tail_from_first_brace(self):
        """Malformed JSON does not raise; the tail from `{` is returned."""
        out = extract_json_object("prefix {not really json")
        assert out.startswith("{")


class TestParseChatbotResponse:
    """Tests for parse_chatbot_response and intent_to_dict."""

    def test_verbal_ack_only(self):
        """Response with only verbal_ack parses cleanly."""
        parsed = parse_chatbot_response('{"verbal_ack": "hello"}')
        assert parsed is not None
        assert parsed.verbal_ack == "hello"
        assert parsed.user_intent is None

    def test_user_intent_only(self):
        """Response with only user_intent parses cleanly."""
        raw = json.dumps({
            "user_intent": {
                "type": Intent.GREET,
                "recipient": "alice",
            }
        })
        parsed = parse_chatbot_response(raw)
        assert parsed is not None
        assert parsed.verbal_ack is None
        assert parsed.user_intent is not None
        assert parsed.user_intent.type == Intent.GREET
        assert parsed.user_intent.recipient == "alice"

    def test_both_fields(self):
        """Response with both verbal_ack and user_intent parses cleanly."""
        raw = json.dumps({
            "verbal_ack": "Sure",
            "user_intent": {"type": Intent.GRAB_OBJECT, "object": "apple1"},
        })
        parsed = parse_chatbot_response(raw)
        assert parsed.verbal_ack == "Sure"
        assert parsed.user_intent.object == "apple1"

    def test_empty_object_is_valid(self):
        """An empty JSON object is a valid response."""
        parsed = parse_chatbot_response('{}')
        assert parsed is not None
        assert parsed.verbal_ack is None
        assert parsed.user_intent is None

    def test_malformed_json_returns_none(self):
        """Malformed JSON returns None and logs a warning."""
        warnings = []

        class _Logger:
            def warn(self, msg):
                warnings.append(msg)

        assert parse_chatbot_response("not json at all", logger=_Logger()) is None
        assert warnings, "expected at least one warning"

    def test_unknown_intent_type_returns_none(self):
        """Intent types not in the Literal allowed set fail validation."""
        raw = json.dumps({
            "user_intent": {"type": "__intent_that_does_not_exist__"},
        })
        warnings = []

        class _Logger:
            def warn(self, msg):
                warnings.append(msg)

        assert parse_chatbot_response(raw, logger=_Logger()) is None
        assert warnings

    def test_intent_to_dict_round_trip(self):
        """intent_to_dict serializes back to the original field values."""
        raw = json.dumps({
            "verbal_ack": "hi",
            "user_intent": {"type": Intent.SAY, "input": "hello"},
        })
        parsed = parse_chatbot_response(raw)
        dumped = intent_to_dict(parsed.user_intent)
        assert dumped["type"] == Intent.SAY
        assert dumped["input"] == "hello"
