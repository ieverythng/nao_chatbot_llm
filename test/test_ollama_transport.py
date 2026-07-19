import json

from chatbot_llm.ollama_transport import OllamaTransport
from chatbot_llm.ollama_transport import _chat_payload
from chatbot_llm.ollama_transport import _model_inventory_url
from chatbot_llm.ollama_transport import _model_names


def test_openai_chat_url_uses_openai_payload_shape():
    payload = _chat_payload(
        server_url='http://10.7.138.215:8004/v1/chat/completions',
        model='served-model',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        think=False,
        context_window_tokens=4096,
        max_tokens=32,
        response_format={'type': 'object'},
    )

    assert payload == {
        'model': 'served-model',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 1.5,
        'repetition_penalty': 1.0,
        'max_tokens': 32,
    }


def test_openai_chat_url_disables_thinking_for_namespaced_qwen_model():
    payload = _chat_payload(
        server_url='http://10.7.138.215:8004/v1/chat/completions',
        model='cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        think=False,
        context_window_tokens=4096,
        max_tokens=32,
        response_format={'type': 'object'},
    )

    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert payload['top_k'] == 20
    assert payload['min_p'] == 0.0
    assert payload['presence_penalty'] == 1.5
    assert payload['repetition_penalty'] == 1.0


def test_ollama_chat_url_keeps_ollama_payload_shape():
    payload = _chat_payload(
        server_url='http://127.0.0.1:11434/api/chat',
        model='llama3.2',
        messages=[{'role': 'user', 'content': 'hello'}],
        temperature=0.2,
        top_p=0.9,
        think=True,
        context_window_tokens=2048,
        max_tokens=16,
        response_format={'type': 'object'},
    )

    assert payload['stream'] is False
    assert payload['think'] is True
    assert payload['options'] == {
        'num_ctx': 2048,
        'temperature': 0.2,
        'top_p': 0.9,
        'num_predict': 16,
    }
    assert payload['format'] == {'type': 'object'}


def test_model_inventory_urls_match_backend_shape():
    assert (
        _model_inventory_url('http://10.7.138.215:8004/v1/chat/completions')
        == 'http://10.7.138.215:8004/v1/models'
    )
    assert (
        _model_inventory_url('http://127.0.0.1:11434/api/chat')
        == 'http://127.0.0.1:11434/api/tags'
    )


def test_model_names_support_openai_and_ollama_inventory_payloads():
    assert _model_names({'data': [{'id': 'openai-model'}]}) == ['openai-model']
    assert _model_names({'models': [{'name': 'ollama-model'}]}) == ['ollama-model']


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
