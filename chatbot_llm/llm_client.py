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

"""HTTP client for OpenAI-compatible chat completion endpoints."""

import json
from typing import Optional

import requests


class LLMClient:
    """Thin wrapper around the OpenAI /v1/chat/completions endpoint."""

    def __init__(
        self,
        server: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        response_schema: Optional[dict] = None,
        logger=None,
    ):
        """Configure the client. See module docstring for parameter meaning."""
        self.server = server
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.response_schema = response_schema
        self._logger = logger

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    def chat(self, messages: list) -> Optional[dict]:
        """Send a chat completion request and return the first choice, or None on failure."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        url = f"{self.server}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "think": False,
        }
        if self.response_schema is not None:
            payload["format"] = self.response_schema

        response = None
        try:
            self._log('info', "Sending request to LLM and waiting for response...")
            response = requests.post(url, headers=headers,
                                     data=json.dumps(payload),
                                     timeout=self.timeout)
            response.raise_for_status()
            return response.json()['choices'][0]
        except requests.exceptions.RequestException as e:
            error_msg = "Unable to decode error message from server"
            if response is not None:
                try:
                    error_msg = response.json()['error']['message']
                except (ValueError, KeyError):
                    pass
            self._log('error',
                      f"Error while sending request to {url}:\n{e}\n{error_msg}\n"
                      "Is the server running, or the URL incorrect "
                      f"(eg wrong port)? I was trying to connect to: {url}")
            return None
