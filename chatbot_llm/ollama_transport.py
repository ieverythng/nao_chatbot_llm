"""HTTP transport used by the migrated chatbot backend."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Ollama HTTP transport
# ---------------------------------------------------------------------------

class OllamaTransport:
    """Wrapper around Ollama REST requests used by the turn engine."""

    def __init__(self, server_url: str, context_window_tokens: int, logger) -> None:
        """Store transport configuration for later HTTP requests."""
        self._server_url = str(server_url)
        self._context_window_tokens = int(context_window_tokens)
        self._logger = logger

    def query(
        self,
        messages: list[dict],
        timeout_sec: float,
        model: str,
        temperature: float,
        top_p: float,
        think: bool = False,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Run one non-streaming chat request against Ollama."""
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'think': bool(think),
            'options': {
                'num_ctx': self._context_window_tokens,
                'temperature': float(temperature),
                'top_p': float(top_p),
            },
        }
        if response_format is not None:
            payload['format'] = response_format
        if max_tokens is not None:
            payload['options']['num_predict'] = max(1, int(max_tokens))

        request = urllib.request.Request(
            self._server_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
                parsed = json.loads(response.read().decode('utf-8'))
            text = _extract_chat_text(parsed)
            if not text:
                self._logger.warn(
                    'Ollama response did not include message content; keys=%s'
                    % ','.join(sorted(str(key) for key in parsed.keys()))
                )
            return text
        except urllib.error.HTTPError as err:
            error_body = err.read().decode('utf-8', errors='replace')
            self._logger.error(
                'Ollama HTTP error %s for model %s at %s: %s'
                % (err.code, model, self._server_url, error_body)
            )
            if err.code == 404:
                self.log_model_inventory()
            return ''
        except urllib.error.URLError as err:
            self._logger.error(f'Ollama request failed: {err}')
            return ''
        except TimeoutError:
            self._logger.error('Ollama request timed out')
            return ''
        except socket.timeout:
            self._logger.error('Ollama socket timeout')
            return ''
        except json.JSONDecodeError as err:
            self._logger.error(f'Ollama response decode failed: {err}')
            return ''
        except Exception as err:  # pragma: no cover - network dependent
            self._logger.error(f'Ollama unexpected error: {err}')
            return ''

    # -----------------------------------------------------------------------
    # Debug and diagnostics helpers
    # -----------------------------------------------------------------------

    def log_model_inventory(self) -> None:
        """Log available model names from the Ollama tags endpoint."""
        try:
            tags_url = self._server_url.replace('/api/chat', '/api/tags')
            with urllib.request.urlopen(tags_url, timeout=5.0) as response:
                payload = json.loads(response.read().decode('utf-8'))
            models = [str(item.get('name', '')).strip() for item in payload.get('models', [])]
            models = [name for name in models if name]
            if models:
                self._logger.info('Ollama available models: %s' % ', '.join(models))
            else:
                self._logger.warn('Ollama tags endpoint returned no models')
        except Exception as err:  # pragma: no cover - network dependent
            self._logger.warn(f'Could not query Ollama model inventory: {err}')


def _extract_chat_text(payload: dict) -> str:
    """Read assistant text from Ollama or OpenAI-compatible response shapes."""
    if not isinstance(payload, dict):
        return ''

    message = payload.get('message', {})
    if isinstance(message, dict):
        text = str(message.get('content', '')).strip()
        if text:
            return text

    text = str(payload.get('response', '')).strip()
    if text:
        return text

    choices = payload.get('choices', [])
    if not isinstance(choices, list):
        return ''
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get('message', {})
        if isinstance(message, dict):
            text = str(message.get('content', '')).strip()
            if text:
                return text
        text = str(choice.get('text', '')).strip()
        if text:
            return text
    return ''
