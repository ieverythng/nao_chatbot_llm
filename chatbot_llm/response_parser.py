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

import pydantic
from pydantic import BaseModel, ValidationError

_PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


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


def extract_json_object(raw_text: str) -> str:
    """Return the first balanced JSON object substring, or `raw_text` itself if none is found."""
    start = raw_text.find('{')
    if start < 0:
        return raw_text
    try:
        _, end = json.JSONDecoder().raw_decode(raw_text, start)
    except json.JSONDecodeError:
        return raw_text[start:]
    return raw_text[start:end]


def parse_chatbot_response(raw_text: str, logger=None) -> Optional[ChatbotResponse]:
    """Parse and validate an LLM response; return None and log a warning on failure."""
    try:
        if _PYDANTIC_V2:
            return ChatbotResponse.model_validate_json(raw_text)
        return ChatbotResponse.parse_raw(raw_text)
    except ValidationError as e:
        if logger is not None:
            logger.warn(f"Failed to parse LLM response: {e}")
    return None


def chatbot_response_schema() -> dict:
    """Return the JSON schema for ChatbotResponse, across pydantic versions."""
    if _PYDANTIC_V2:
        return ChatbotResponse.model_json_schema()
    return ChatbotResponse.schema()


def intent_to_dict(intent_model) -> dict:
    """Serialize an IntentModel to a dict, across pydantic versions."""
    if _PYDANTIC_V2:
        return intent_model.model_dump()
    return intent_model.dict()
