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

"""Unit tests for chatbot_llm.messages."""

from string import Template

from chatbot_msgs.msg import Utterance

from chatbot_llm.messages import (
    build_llm_messages,
    last_user_speaker,
    render_system_prompt,
    trim_messages,
)
from chatbot_llm.role_handlers import DefaultRoleHandler


def _utt(speaker: str, text: str) -> Utterance:
    return Utterance(speaker=speaker, text=text)


class TestLastUserSpeaker:
    """Tests for last_user_speaker."""

    def test_empty_history_returns_default(self):
        """Empty history -> default sentinel."""
        assert last_user_speaker([]) == "anonymous_user"

    def test_only_system_and_assistant_returns_default(self):
        """A history of only system + assistant entries yields the default."""
        history = [
            _utt(Utterance.SYSTEM, "world update"),
            _utt(Utterance.ASSISTANT, "hello"),
        ]
        assert last_user_speaker(history) == "anonymous_user"

    def test_returns_most_recent_real_user(self):
        """The most recent non-sentinel speaker wins."""
        history = [
            _utt("alice", "hi"),
            _utt(Utterance.ASSISTANT, "hello alice"),
            _utt("bob", "hey"),
            _utt(Utterance.ASSISTANT, "hello bob"),
        ]
        assert last_user_speaker(history) == "bob"

    def test_empty_speaker_falls_back_to_default(self):
        """An empty speaker string is replaced with the default."""
        history = [_utt("", "hi")]
        assert last_user_speaker(history) == "anonymous_user"


class TestBuildLlmMessages:
    """Tests for build_llm_messages."""

    def test_system_prompt_is_first(self):
        """The rendered system prompt always comes first."""
        msgs = build_llm_messages(
            system_prompt="SYS",
            summary="",
            history=[_utt("alice", "hi")],
        )
        assert msgs[0] == {"role": "system", "content": "SYS"}

    def test_summary_inserted_after_system(self):
        """A non-empty summary becomes a second system message."""
        msgs = build_llm_messages(
            system_prompt="SYS",
            summary="prior session was great",
            history=[_utt("alice", "hi")],
        )
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS"
        assert msgs[1]["role"] == "system"
        assert "prior session was great" in msgs[1]["content"]

    def test_empty_summary_not_inserted(self):
        """An empty summary produces no extra entry."""
        msgs = build_llm_messages(
            system_prompt="SYS",
            summary="",
            history=[_utt("alice", "hi")],
        )
        assert len(msgs) == 2  # system + the user utterance

    def test_speaker_routing(self):
        """SYSTEM -> system; ASSISTANT -> assistant; anything else -> user."""
        history = [
            _utt(Utterance.SYSTEM, "world"),
            _utt("alice", "hi"),
            _utt(Utterance.ASSISTANT, "hello"),
        ]
        msgs = build_llm_messages(
            system_prompt="SYS", summary="", history=history
        )
        # msgs[0] = leading system prompt
        assert msgs[1] == {"role": "system", "content": "world"}
        assert msgs[2] == {"role": "user", "content": 'alice "hi"'}
        assert msgs[3] == {"role": "assistant", "content": "hello"}


class TestTrimMessages:
    """Tests for trim_messages."""

    def test_below_bound_unchanged(self):
        """If messages fit, return them as-is."""
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(3)
        ]
        assert trim_messages(msgs, max_turns=10) == msgs

    def test_keeps_system_prompt_and_tail(self):
        """When trimming, the leading system prompt is preserved."""
        msgs = [{"role": "system", "content": "S"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        trimmed = trim_messages(msgs, max_turns=3)
        # max_msgs = 1 + 2 * 3 = 7
        assert len(trimmed) == 7
        assert trimmed[0] == {"role": "system", "content": "S"}
        assert trimmed[-1] == {"role": "assistant", "content": "a19"}

    def test_zero_max_turns_is_no_op(self):
        """A non-positive max_turns disables trimming."""
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(50)
        ]
        assert trim_messages(msgs, max_turns=0) == msgs


class TestRenderSystemPrompt:
    """Tests for render_system_prompt."""

    def test_substitutes_all_placeholders(self):
        """The template renders with the provided variables."""
        tpl = Template(
            "I am $robot_name, talking to $user_id with role $role. "
            "Actions: $action_list. World: $environment."
        )
        out = render_system_prompt(
            tpl,
            robot_name="ari",
            role="__default__",
            user_id="alice",
            action_list="...",
            environment="...",
            handler=DefaultRoleHandler(),
        )
        assert "ari" in out
        assert "alice" in out
        assert "__default__" in out

    def test_handler_extension_is_appended(self):
        """Non-empty handler extensions are concatenated after a blank line."""
        class _Handler(DefaultRoleHandler):
            def system_prompt_extension(self):
                return "EXTRA"

        out = render_system_prompt(
            Template("BASE"),
            robot_name="r",
            role="__default__",
            user_id="u",
            action_list="",
            environment="",
            handler=_Handler(),
        )
        assert out == "BASE\n\nEXTRA"
