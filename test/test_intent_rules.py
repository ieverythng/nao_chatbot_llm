from chatbot_llm.intent_rules import is_execution_intent_label
from chatbot_llm.intent_rules import normalize_intent


def test_normalize_intent_keeps_fake_skill_label() -> None:
    assert normalize_intent('navigate_to', default='fallback') == 'navigate_to'


def test_execution_label_includes_wave_greet() -> None:
    assert is_execution_intent_label('wave_greet') is True
