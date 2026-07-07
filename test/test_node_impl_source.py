from pathlib import Path


def test_grounded_context_digest_switch_uses_loaded_backend_config():
    source = Path(__file__).parents[1].joinpath('chatbot_llm', 'node_impl.py').read_text()

    assert 'self._config.grounded_context_digest_enabled' in source
    assert 'self.config.grounded_context_digest_enabled' not in source
