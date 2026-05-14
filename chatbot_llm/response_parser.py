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

"""Data models and parsing for LLM responses."""

import json
from typing import Literal, Optional

from hri_actions_msgs.msg import Intent
from pydantic import BaseModel


class IntentModel(BaseModel):
    """Schema for the user_intent field of a chatbot response."""

    type: Literal[
        Intent.BRING_OBJECT,
        Intent.GRAB_OBJECT,
        Intent.PLACE_OBJECT,
        Intent.GUIDE,
        Intent.MOVE_TO,
        Intent.SAY,
        Intent.GREET,
        Intent.START_ACTIVITY,
    ]
    object: Optional[str] = None
    recipient: Optional[str] = None
    input: Optional[str] = None
    goal: Optional[str] = None


class ChatbotResponse(BaseModel):
    """Schema for the structured response returned by the LLM."""

    verbal_ack: Optional[str] = None
    user_intent: Optional[IntentModel] = None


def preprocess_llm_response(raw_text: str) -> str:
    """Extract the first balanced {...} substring from `raw_text`."""
    # extract the substring between the first
    # opening and closing curly braces (accounting for nested braces).
    start_idx = 0
    end_idx = len(raw_text) - 1
    nested = 0
    for i, c in enumerate(raw_text):
        if c == '{':
            start_idx = i
            break
    for i in range(start_idx, len(raw_text)):
        if raw_text[i] == '{':
            nested += 1
        elif raw_text[i] == '}':
            nested -= 1
            if nested == 0:
                end_idx = i
                break

    text = raw_text[start_idx:end_idx + 1]

    # the LLM tends to remove spaces before colons, which cause
    # invalid YAML parsing.
    text = text.replace(":", ": ")

    return text


def parse_chatbot_response(raw_text: str, logger=None) -> Optional[ChatbotResponse]:
    """Parse the LLM response into a ChatbotResponse, or return None on failure."""
    if hasattr(ChatbotResponse, "model_validate_json"):
        return ChatbotResponse.model_validate_json(raw_text)

    try:
        as_dict = json.loads(raw_text)
        return ChatbotResponse(**as_dict)
    except json.decoder.JSONDecodeError as e:
        if logger is not None:
            logger.warn(f"Malformed JSON response: {e}")
    except Exception as e:
        if logger is not None:
            logger.warn(f"LLM response does not match expected format: {e}")
    return None
