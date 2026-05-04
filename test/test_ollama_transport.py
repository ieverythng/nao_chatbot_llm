import json

from chatbot_llm.ollama_transport import OllamaTransport


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode('utf-8')


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings = []
        self.errors = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


def test_ollama_transport_accepts_openai_compatible_choices(monkeypatch):
    def fake_urlopen(_request, timeout):
        assert timeout == 3.0
        return _FakeResponse(
            {
                'choices': [
                    {
                        'message': {
                            'content': '{"verbal_ack":"Sure.","route":"execution"}',
                        },
                    },
                ],
            }
        )

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    logger = _FakeLogger()
    transport = OllamaTransport(
        server_url='http://localhost:11434/api/chat',
        context_window_tokens=4096,
        logger=logger,
    )

    text = transport.query(
        messages=[{'role': 'user', 'content': 'stand up'}],
        timeout_sec=3.0,
        model='gpt-oss:120b-cloud',
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        response_format={'type': 'object'},
    )

    assert text == '{"verbal_ack":"Sure.","route":"execution"}'
    assert logger.warnings == []


def test_ollama_transport_disables_format_for_gemma_models(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured['timeout'] = timeout
        captured['url'] = request.full_url
        captured['payload'] = json.loads(request.data.decode('utf-8'))
        return _FakeResponse({'message': {'content': '{"verbal_ack":"Sure."}'}})

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    logger = _FakeLogger()
    transport = OllamaTransport(
        server_url='http://localhost:11434/api/chat',
        context_window_tokens=4096,
        logger=logger,
    )

    text = transport.query(
        messages=[{'role': 'user', 'content': 'stand up'}],
        timeout_sec=3.0,
        model='gemma4:31b-cloud',
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        response_format={'type': 'object'},
    )

    assert text == '{"verbal_ack":"Sure."}'
    assert captured['timeout'] == 3.0
    assert 'format' not in captured['payload']


def test_ollama_transport_falls_back_to_message_thinking(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured['payload'] = json.loads(request.data.decode('utf-8'))
        return _FakeResponse(
            {
                'message': {
                    'content': '',
                    'thinking': '{"verbal_ack":"Sure.","route":"execution"}',
                },
            }
        )

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    logger = _FakeLogger()
    transport = OllamaTransport(
        server_url='http://localhost:11434/api/chat',
        context_window_tokens=4096,
        logger=logger,
    )

    text = transport.query(
        messages=[{'role': 'system', 'content': 'Return JSON only.'}],
        timeout_sec=3.0,
        model='qwen3.5:cloud',
        temperature=0.0,
        top_p=0.9,
        max_tokens=64,
        response_format={'type': 'object'},
    )

    assert text == '{"verbal_ack":"Sure.","route":"execution"}'
    assert captured['payload']['messages'][0]['content'].startswith('/no_think')
    assert logger.warnings == []


def test_ollama_transport_preflight_accepts_ready_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured['timeout'] = timeout
        captured['payload'] = json.loads(request.data.decode('utf-8'))
        return _FakeResponse({'message': {'content': '{"ready":true}'}})

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    transport = OllamaTransport(
        server_url='http://localhost:11434/api/chat',
        context_window_tokens=4096,
        logger=_FakeLogger(),
    )

    assert transport.preflight(
        model='gemma4:31b-cloud',
        timeout_sec=4.0,
        temperature=0.2,
        top_p=0.9,
        think=False,
    )
    assert captured['timeout'] == 4.0
    assert captured['payload']['options']['num_predict'] == 16
