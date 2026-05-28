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

"""Unit tests for chatbot_llm.llm_client."""

import json
from unittest.mock import MagicMock, patch

from chatbot_llm.llm_client import LLMClient

import requests


class _CapturingLogger:
    """Minimal logger that records (level, message) tuples."""

    def __init__(self):
        self.records = []

    def info(self, msg):
        self.records.append(("info", msg))

    def warn(self, msg):
        self.records.append(("warn", msg))

    def error(self, msg):
        self.records.append(("error", msg))


def _mock_response(status_code=200, json_body=None):
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error")
    else:
        mock.raise_for_status.return_value = None
    return mock


class TestLLMClient:
    """Tests for the LLMClient HTTP wrapper."""

    def test_successful_request_returns_first_choice(self):
        """A 2xx response returns the first entry of `choices`."""
        client = LLMClient(server="http://x", model="m")
        mock = _mock_response(json_body={
            "choices": [
                {"message": {"content": "{}"}},
                {"message": {"content": "ignored"}},
            ],
        })
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            choice = client.chat([{"role": "user", "content": "hi"}])
        assert choice == {"message": {"content": "{}"}}
        called_url = (
            p.call_args.args[0] if p.call_args.args else p.call_args.kwargs["url"]
        )
        assert called_url == "http://x/v1/chat/completions"

    def test_api_key_sent_as_bearer(self):
        """An api_key is forwarded as a Bearer Authorization header."""
        client = LLMClient(server="http://x", model="m", api_key="secret")
        mock = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            client.chat([])
        headers = p.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret"

    def test_no_api_key_omits_authorization(self):
        """No api_key means no Authorization header is sent."""
        client = LLMClient(server="http://x", model="m")
        mock = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            client.chat([])
        headers = p.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_response_schema_included_when_set(self):
        """A response_schema is forwarded under the `format` key."""
        schema = {"type": "object"}
        client = LLMClient(server="http://x", model="m", response_schema=schema)
        mock = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            client.chat([])
        body = json.loads(p.call_args.kwargs["data"])
        assert body["format"] == schema

    def test_response_schema_omitted_when_none(self):
        """Without a response_schema, `format` is not sent at all."""
        client = LLMClient(server="http://x", model="m", response_schema=None)
        mock = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            client.chat([])
        body = json.loads(p.call_args.kwargs["data"])
        assert "format" not in body

    def test_timeout_forwarded(self):
        """The configured timeout is forwarded to requests.post."""
        client = LLMClient(server="http://x", model="m", timeout=5.5)
        mock = _mock_response(json_body={"choices": [{"message": {"content": "{}"}}]})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock) as p:
            client.chat([])
        assert p.call_args.kwargs["timeout"] == 5.5

    def test_connection_error_returns_none_and_logs(self):
        """Connection errors are caught, logged, and surfaced as None."""
        logger = _CapturingLogger()
        client = LLMClient(server="http://x", model="m", logger=logger)
        with patch("chatbot_llm.llm_client.requests.post",
                   side_effect=requests.exceptions.ConnectionError("refused")):
            assert client.chat([]) is None
        assert any(level == "error" for level, _ in logger.records)

    def test_timeout_returns_none_and_logs(self):
        """Request timeouts are caught, logged, and surfaced as None."""
        logger = _CapturingLogger()
        client = LLMClient(server="http://x", model="m", logger=logger)
        with patch("chatbot_llm.llm_client.requests.post",
                   side_effect=requests.exceptions.Timeout("timed out")):
            assert client.chat([]) is None
        assert any(level == "error" for level, _ in logger.records)

    def test_http_5xx_returns_none_and_uses_server_error_message(self):
        """A 5xx with a parseable error body surfaces the server message."""
        logger = _CapturingLogger()
        client = LLMClient(server="http://x", model="m", logger=logger)
        mock = _mock_response(status_code=500,
                              json_body={"error": {"message": "boom"}})
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock):
            assert client.chat([]) is None
        assert any("boom" in msg for level, msg in logger.records if level == "error")

    def test_http_5xx_without_parseable_body(self):
        """A 5xx with an unparseable body still returns None and logs."""
        logger = _CapturingLogger()
        client = LLMClient(server="http://x", model="m", logger=logger)
        mock = MagicMock(spec=requests.Response)
        mock.status_code = 500
        mock.json.side_effect = ValueError("no json")
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with patch("chatbot_llm.llm_client.requests.post", return_value=mock):
            assert client.chat([]) is None
        assert any(level == "error" for level, _ in logger.records)
