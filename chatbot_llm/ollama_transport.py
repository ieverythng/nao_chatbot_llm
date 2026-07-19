"""HTTP transport used by the migrated chatbot backend."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

_FORMAT_DISABLED_MODEL_PREFIXES = ('gemma',)
_NO_THINK_MODEL_PREFIXES = ('qwen3', 'qwen3.5')
_NO_THINK_PREFIX = (
    '/no_think\n'
    'Return only the final answer. Do not emit reasoning, analysis, or thinking text.'
)


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
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        """Run one non-streaming chat request against Ollama."""
        request_messages = _no_think_messages(model, messages)
        payload = _chat_payload(
            server_url=self._server_url,
            model=model,
            messages=request_messages,
            temperature=temperature,
            top_p=top_p,
            think=think,
            context_window_tokens=self._context_window_tokens,
            max_tokens=max_tokens,
            response_format=response_format,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )

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
                    'Ollama response did not include assistant text; keys=%s'
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

    def preflight(
        self,
        *,
        model: str,
        timeout_sec: float,
        temperature: float,
        top_p: float,
        think: bool,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> bool:
        """Return true when the configured model can answer a tiny JSON request."""
        text = self.query(
            messages=[
                {
                    'role': 'system',
                    'content': 'Reply only with JSON. No prose.',
                },
                {
                    'role': 'user',
                    'content': 'Return {"ready":true}.',
                },
            ],
            timeout_sec=timeout_sec,
            model=model,
            temperature=temperature,
            top_p=top_p,
            think=think,
            max_tokens=16,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
        return _preflight_ready(text)

    def readiness_probe(
        self,
        *,
        model: str,
        timeout_sec: float,
        temperature: float,
        top_p: float,
        think: bool,
        max_tokens: int = 32,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> bool:
        """Return true when the model can answer a demo-shaped chatbot request."""
        text = self.query(
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a concise robot dialogue router. Reply with one short '
                        'verbal acknowledgement for a live demo.'
                    ),
                },
                {
                    'role': 'user',
                    'content': 'Move your head to the right and tell me what you see.',
                },
            ],
            timeout_sec=timeout_sec,
            model=model,
            temperature=temperature,
            top_p=top_p,
            think=think,
            max_tokens=max_tokens,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )
        return bool(str(text or '').strip())

    # -----------------------------------------------------------------------
    # Debug and diagnostics helpers
    # -----------------------------------------------------------------------

    def log_model_inventory(self) -> None:
        """Log available model names from the Ollama tags endpoint."""
        try:
            tags_url = _model_inventory_url(self._server_url)
            with urllib.request.urlopen(tags_url, timeout=5.0) as response:
                payload = json.loads(response.read().decode('utf-8'))
            models = _model_names(payload)
            models = [name for name in models if name]
            if models:
                self._logger.info('Available LLM models: %s' % ', '.join(models))
            else:
                self._logger.warn('LLM model inventory endpoint returned no models')
        except Exception as err:  # pragma: no cover - network dependent
            self._logger.warn(f'Could not query LLM model inventory: {err}')


def _chat_payload(
    *,
    server_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    think: bool,
    context_window_tokens: int,
    max_tokens: int | None,
    response_format: dict | None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> dict:
    if _is_openai_chat_url(server_url):
        payload = {
            'model': model,
            'messages': messages,
            'temperature': float(temperature),
            'top_p': float(top_p),
        }
        if max_tokens is not None:
            payload['max_tokens'] = max(1, int(max_tokens))
        if top_k is not None:
            payload['top_k'] = max(0, int(top_k))
        if min_p is not None:
            payload['min_p'] = max(0.0, float(min_p))
        if presence_penalty is not None:
            payload['presence_penalty'] = float(presence_penalty)
        if repetition_penalty is not None:
            payload['repetition_penalty'] = max(0.0, float(repetition_penalty))
        template_kwargs = _openai_chat_template_kwargs(model)
        if template_kwargs:
            payload['chat_template_kwargs'] = template_kwargs
        return payload

    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'think': bool(think),
        'options': {
            'num_ctx': int(context_window_tokens),
            'temperature': float(temperature),
            'top_p': float(top_p),
        },
    }
    if response_format is not None and _uses_ollama_response_format(model):
        payload['format'] = response_format
    if max_tokens is not None:
        payload['options']['num_predict'] = max(1, int(max_tokens))
    if top_k is not None:
        payload['options']['top_k'] = max(0, int(top_k))
    if min_p is not None:
        payload['options']['min_p'] = max(0.0, float(min_p))
    if presence_penalty is not None:
        payload['options']['presence_penalty'] = float(presence_penalty)
    if repetition_penalty is not None:
        payload['options']['repeat_penalty'] = max(0.0, float(repetition_penalty))
    return payload


def _is_openai_chat_url(server_url: str) -> bool:
    return str(server_url or '').rstrip('/').endswith('/v1/chat/completions')


def _model_inventory_url(server_url: str) -> str:
    clean_url = str(server_url or '').rstrip('/')
    if _is_openai_chat_url(clean_url):
        return clean_url[: -len('/chat/completions')] + '/models'
    return clean_url.replace('/api/chat', '/api/tags')


def _model_names(payload: dict) -> list[str]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get('models'), list):
        return [str(item.get('name', '')).strip() for item in payload.get('models', [])]
    if isinstance(payload.get('data'), list):
        return [str(item.get('id', '')).strip() for item in payload.get('data', [])]
    return []


def _extract_chat_text(payload: dict) -> str:
    """Read assistant text from Ollama or OpenAI-compatible response shapes."""
    if not isinstance(payload, dict):
        return ''

    message = payload.get('message', {})
    if isinstance(message, dict):
        text = _assistant_message_text(message)
        if text:
            return text

    text = str(payload.get('response', '')).strip()
    if text:
        return text

    text = _thinking_text(payload)
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
            text = _assistant_message_text(message)
            if text:
                return text
        text = str(choice.get('text', '')).strip()
        if text:
            return text
    return ''


def _assistant_message_text(message: dict) -> str:
    if not isinstance(message, dict):
        return ''
    text = str(message.get('content', '')).strip()
    if text:
        return text
    return _thinking_text(message)


def _thinking_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ''
    for key in ('thinking', 'reasoning'):
        text = str(payload.get(key, '')).strip()
        if text:
            return text
    return ''


def _no_think_messages(model: str, messages: list[dict]) -> list[dict]:
    if not _is_qwen3_model(model):
        return list(messages)
    if not messages:
        return [{'role': 'system', 'content': _NO_THINK_PREFIX}]
    prepared = [dict(message) for message in messages]
    first = prepared[0]
    if first.get('role') == 'system':
        first['content'] = '%s\n\n%s' % (_NO_THINK_PREFIX, str(first.get('content', '')).strip())
    else:
        prepared.insert(0, {'role': 'system', 'content': _NO_THINK_PREFIX})
    return prepared


def _is_qwen3_model(model: str) -> bool:
    model_name = str(model or '').strip().lower().rsplit('/', 1)[-1]
    return any(model_name.startswith(prefix) for prefix in _NO_THINK_MODEL_PREFIXES)


def _openai_chat_template_kwargs(model: str) -> dict[str, bool]:
    if not _is_qwen3_model(model):
        return {}
    return {'enable_thinking': False}


def _uses_ollama_response_format(model: str) -> bool:
    clean_model = str(model or '').strip().lower()
    return not any(clean_model.startswith(prefix) for prefix in _FORMAT_DISABLED_MODEL_PREFIXES)


def _preflight_ready(text: str) -> bool:
    clean_text = str(text or '').strip()
    if not clean_text:
        return False
    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError:
        return '"ready"' in clean_text.lower() and 'true' in clean_text.lower()
    return isinstance(parsed, dict) and parsed.get('ready') is True
