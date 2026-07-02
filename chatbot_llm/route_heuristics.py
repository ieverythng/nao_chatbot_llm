"""Deterministic text classifiers used to nudge chatbot turn routing.

These are pure functions over ``user_text`` (and a few over ``verbal_ack``)
that the turn engine consults when the LLM does not return a usable route or
when a known route-safety guardrail applies. They contain no transport, state,
or speaking side effects.

Ownership note: this module is the single home for the route/intent
classification heuristics that previously lived inline in ``turn_engine``. The
turn engine remains the orchestration owner; these helpers only classify text.
"""

from __future__ import annotations

import re

from chatbot_llm.intent_rules import detect_intent
from chatbot_llm.intent_rules import is_execution_intent_label
from chatbot_llm.intent_rules import normalize_intent
from kb_skills.intent_labels import KB_QUERY_INTENTS, KB_QUERY_VISIBLE_OBJECTS


_DIALOGUE_ROUTE = 'dialogue'
_KNOWLEDGE_QUERY_ROUTE = 'knowledge_query'
_EXECUTION_ROUTE = 'execution'
_SUPPORTED_ROUTES = {
    _DIALOGUE_ROUTE,
    _KNOWLEDGE_QUERY_ROUTE,
    _EXECUTION_ROUTE,
}
_DIALOGUE_INTENTS = {'greet', 'identity', 'wellbeing', 'help'}
_EXECUTION_HINT_MARKERS = (
    ' and then ',
    ' then ',
    ' after that ',
    ' before that ',
    ' while also ',
    ' also ',
    ' stand',
    ' sit',
    ' kneel',
    ' look ',
    ' move ',
    ' scan ',
    ' search ',
    ' find ',
    ' locate ',
    ' detect ',
    ' turn ',
    ' head ',
    ' bring ',
    ' grab ',
    ' pick ',
    ' place ',
    ' guide ',
    ' navigate ',
    ' walk ',
    ' wave ',
    ' add ',
    ' remember ',
    ' store ',
    ' save ',
    ' revise ',
    ' update ',
    ' change ',
    ' correct ',
    ' remove ',
    ' delete ',
    ' forget ',
    ' knowledge base',
    ' kb ',
    ' rdf:type',
    ' dbp:',
    ' oro:',
)
_REFLECTIVE_EXECUTION_QUESTION_MARKERS = (
    'what did you',
    'what have you',
    'what were you able to',
    'how many',
    'which direction',
    'which directions',
    'did you',
    'have you',
)
_SOCIAL_TURN_MARKERS = {
    'hi',
    'hello',
    'hey',
    'hey there',
    'good morning',
    'good afternoon',
    'good evening',
    'how are you',
    'thanks',
    'thank you',
}


def _normalize_route(value) -> str:
    clean_value = str(value or '').strip().lower()
    if clean_value in _SUPPORTED_ROUTES:
        return clean_value
    return ''


def _looks_like_execution_text(user_text: str) -> bool:
    if _is_information_only_action_word_question(user_text):
        return False
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(marker in lowered for marker in _EXECUTION_HINT_MARKERS)


def _rules_execution_intent_allowed(user_text: str) -> bool:
    return _looks_like_execution_text(user_text) and not _is_information_only_action_word_question(
        user_text
    )


def _execution_intent_from_text(user_text: str) -> str:
    intent = normalize_intent(detect_intent(user_text), default='')
    if intent and intent != 'fallback' and is_execution_intent_label(intent):
        return intent
    clean = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    if ' look at ' in clean:
        return 'look_at'
    return ''


def _is_repeat_action_request(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized in {
        'again',
        'do it again',
        'do that again',
        'repeat it',
        'repeat that',
        'repeat the action',
        'one more time',
    }


def _is_information_only_action_word_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    asks_information = (
        '?' in clean
        or normalized.startswith(('what ', 'how ', 'why ', 'explain ', 'define '))
        or normalized.startswith('tell me about ')
    )
    if not asks_information:
        return False
    if any(
        marker in clean
        for marker in (
            'wave at',
            'wave to',
            'please wave',
            'can you wave',
            'could you wave',
            'will you wave',
            'would you wave',
            'look at',
            'navigate to',
            'walk to',
            'move to',
            'go to',
        )
    ):
        return False
    return any(
        marker in normalized
        for marker in (
            'wave particle',
            'wave equation',
            'particle equation',
            'wave duality',
            'physics',
            'wavelength',
        )
    ) or normalized.startswith(('what is wave', 'what are waves', 'explain wave'))


def _route_is_contradictory(*, user_text: str, verbal_ack: str, route: str) -> bool:
    if route == _EXECUTION_ROUTE:
        return _is_non_immediate_action_discussion(user_text)
    if route in {_DIALOGUE_ROUTE, _KNOWLEDGE_QUERY_ROUTE}:
        return (
            _ack_implies_execution(verbal_ack)
            and not _is_reflective_execution_question(user_text)
            and not _is_capability_query(user_text)
        )
    return False


def _repair_response_route(*, user_text: str, verbal_ack: str, route: str) -> str:
    if _is_reflective_execution_question(user_text):
        return _DIALOGUE_ROUTE
    if _is_capability_query(user_text):
        return _DIALOGUE_ROUTE
    if _is_personal_preference_question(user_text):
        return _DIALOGUE_ROUTE
    if _is_advice_or_idea_request(user_text):
        return _DIALOGUE_ROUTE
    if _is_social_turn(user_text) and not _looks_like_execution_text(user_text):
        return _DIALOGUE_ROUTE
    if _is_non_immediate_action_discussion(user_text):
        return _DIALOGUE_ROUTE
    if _looks_like_execution_text(user_text) and _ack_implies_execution(verbal_ack):
        return _EXECUTION_ROUTE
    if route in _SUPPORTED_ROUTES:
        return route
    return _DIALOGUE_ROUTE


def _is_non_immediate_action_discussion(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean or not _looks_like_execution_text(clean):
        return False
    if not any(marker in clean for marker in ('could', 'would', 'can we', 'could we')):
        return False
    return any(
        marker in clean
        for marker in (
            'later',
            'some other time',
            'at some point',
            'afterwards',
            'eventually',
        )
    )


def _is_social_turn(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch for ch in clean if ch.isalnum() or ch.isspace()).strip()
    if normalized in _SOCIAL_TURN_MARKERS:
        return True
    token_count = len([token for token in normalized.split(' ') if token])
    if token_count <= 2 and normalized in {'hi', 'hello', 'hey', 'thanks'}:
        return True
    return False


def _is_capability_query(user_text: str) -> bool:
    """Return whether the user is asking what the robot can do, not requesting it."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch for ch in clean if ch.isalnum() or ch.isspace()).strip()
    capability_markers = (
        'what can you do',
        'what are you able to do',
        'what capabilities do you have',
        'what are your capabilities',
        'tell me what you can do',
        'what skills do you have',
        'which skills do you have',
        'what fake skills do you have',
        'do you have fake skills',
        'do you have any fake skills',
        'tell me about your skills',
    )
    return any(marker in normalized for marker in capability_markers)


def _is_personal_preference_question(user_text: str) -> bool:
    """Return whether the user is asking for conversation, not robot execution."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    if not (
        '?' in clean
        or normalized.startswith(('what ', 'which ', 'who ', 'do you ', 'tell me '))
    ):
        return False
    preference_markers = (
        'your favorite',
        'your favourite',
        'do you like',
        'what do you like',
        'which do you like',
        'what do you prefer',
        'which do you prefer',
        'your preference',
        'your opinion',
    )
    return any(marker in normalized for marker in preference_markers)


def _is_advice_or_idea_request(user_text: str) -> bool:
    """Return whether the user is asking for suggestions, not robot execution."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    if '?' not in clean and not normalized.startswith(
        ('any ', 'what ', 'which ', 'can you suggest', 'could you suggest')
    ):
        return False
    return any(
        marker in normalized
        for marker in (
            'any ideas',
            'ideas for',
            'plans this weekend',
            'things to do',
            'what should i do',
            'what can i do',
            'can you suggest',
            'could you suggest',
            'recommend',
        )
    )


def _is_reflective_execution_question(user_text: str) -> bool:
    """Return whether the user is asking about prior execution evidence."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    if not any(marker in clean for marker in _REFLECTIVE_EXECUTION_QUESTION_MARKERS):
        return False
    return _looks_like_execution_text(clean)


def _dialogue_user_intent(user_intent: dict) -> dict:
    if not user_intent:
        return {}
    normalized = dict(user_intent)
    normalized['type'] = 'fallback'
    for key in ('goal', 'goal_text', 'intent_sequence'):
        normalized.pop(key, None)
    return normalized


def _ack_implies_execution(verbal_ack: str) -> bool:
    clean_ack = ' %s ' % ' '.join(str(verbal_ack or '').strip().lower().split())
    if not clean_ack.strip():
        return False
    commitment_markers = (
        " i will ",
        " i'll ",
        ' let me ',
        ' i can ',
    )
    if not any(marker in clean_ack for marker in commitment_markers):
        return False
    return _looks_like_execution_text(clean_ack)


def _infer_kb_query_intent_from_text(user_text: str) -> str:
    inferred = normalize_intent(detect_intent(user_text), default='')
    if inferred in KB_QUERY_INTENTS:
        return inferred
    if _looks_like_visible_scene_question(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    if _looks_like_non_mutating_kb_question(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    # Catch specific-object attribute queries like "What is the name/color of X?"
    # These are KB lookups even when detect_intent returns 'fallback'.
    if _looks_like_object_attribute_query(user_text):
        return KB_QUERY_VISIBLE_OBJECTS
    return ''


def _looks_like_visible_scene_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized.startswith(
        (
            'what can you see',
            'what do you see',
            'what are you seeing',
            'who can you see',
            'who do you see',
            'what objects can you see',
            'what objects do you see',
            'how many objects can you see',
            'how many objects do you see',
        )
    )


def _looks_like_non_mutating_kb_question(user_text: str) -> bool:
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    normalized = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in clean)
    normalized = ' '.join(normalized.split())
    return normalized.startswith(
        (
            'what do you remember',
            'what can you remember',
            'do you remember',
            'what do you know',
            'what is',
            'what are',
            'which facts',
            'tell me what',
        )
    )


def _looks_like_object_attribute_query(user_text: str) -> bool:
    """Detect queries about object attributes: name, color, type, position."""
    clean = ' '.join(str(user_text or '').strip().lower().split())
    if not clean:
        return False
    # Must ask about an attribute
    attr_markers = ('name', 'color', 'type', 'position', 'location', 'id')
    has_attr = any(m in clean for m in attr_markers)
    # Must reference a specific object (not general "what do you see")
    obj_patterns = [
        r'of\s+\w+',           # "of the cup", "of X"
        r'the\s+\w+',          # "the probe cup"
        r'is\s+the\s+',        # "is the name"
    ]
    has_object = any(re.search(p, clean) for p in obj_patterns)
    return has_attr and has_object


def _is_greeting_intent(resolved_intent: str, user_intent: dict) -> bool:
    if str(resolved_intent or '').strip().lower() == 'greet':
        return True
    return str(user_intent.get('type', '')).strip().lower() == 'greet'


def _has_explicit_perception_action_request(user_text: str) -> bool:
    lowered = ' %s ' % ' '.join(str(user_text or '').strip().lower().split())
    return any(
        marker in lowered
        for marker in (
            ' scan ',
            ' look around ',
            ' inspect the scene ',
            ' inspect scene ',
            ' search the area ',
            ' search around ',
            ' check the area ',
            ' perform a scan ',
        )
    )
