"""Intent normalization and rules fallback for the migrated chatbot backend."""

from __future__ import annotations

import re

from kb_skills.intent_labels import KB_QUERY_INTENTS
from kb_skills.intent_labels import KB_QUERY_SCENE_CHANGE
from kb_skills.intent_labels import KB_QUERY_VISIBLE_OBJECTS
from kb_skills.intent_labels import KB_QUERY_VISIBLE_PEOPLE


SUPPORTED_INTENTS = (
    'posture_stand',
    'posture_sit',
    'posture_kneel',
    'head_center',
    'head_look_left',
    'head_look_right',
    'head_look_up',
    'head_look_down',
    'greet',
    'wellbeing',
    'identity',
    'help',
    *KB_QUERY_INTENTS,
    'fallback',
)

INTENT_ALIASES = {
    '__intent_greet__': 'greet',
    '__intent_hello__': 'greet',
    '__intent_identity__': 'identity',
    '__intent_help__': 'help',
    '__intent_kb_query_visible_people__': KB_QUERY_VISIBLE_PEOPLE,
    '__intent_kb_query_visible_objects__': KB_QUERY_VISIBLE_OBJECTS,
    '__intent_kb_query_scene_change__': KB_QUERY_SCENE_CHANGE,
    '__intent_wellbeing__': 'wellbeing',
    '__intent_stand__': 'posture_stand',
    '__intent_sit__': 'posture_sit',
    '__intent_kneel__': 'posture_kneel',
    '__intent_head_center__': 'head_center',
    '__intent_look_left__': 'head_look_left',
    '__intent_look_right__': 'head_look_right',
    '__intent_look_up__': 'head_look_up',
    '__intent_look_down__': 'head_look_down',
}


def normalize_intent(intent: str, default: str = 'fallback', hint_text: str = '') -> str:
    """Normalize incoming intent labels to one of ``SUPPORTED_INTENTS``."""
    raw = str(intent).strip().lower()
    hints = str(hint_text).strip().lower()
    if not raw:
        if hints:
            hinted = detect_intent(hints)
            if hinted != 'fallback':
                return hinted
        return default
    if raw in SUPPORTED_INTENTS:
        return raw
    if raw in INTENT_ALIASES:
        return INTENT_ALIASES[raw]

    search_space = f'{raw} {hints}'.strip().replace('_', ' ').replace('-', ' ')
    if 'stand' in search_space or search_space.endswith(' up'):
        return 'posture_stand'
    if 'sit' in search_space or 'seat' in search_space:
        return 'posture_sit'
    if 'kneel' in search_space or 'crouch' in search_space or 'seiza' in search_space:
        return 'posture_kneel'
    if 'look left' in search_space or 'turn left' in search_space:
        return 'head_look_left'
    if 'look right' in search_space or 'turn right' in search_space:
        return 'head_look_right'
    if 'look up' in search_space:
        return 'head_look_up'
    if 'look down' in search_space:
        return 'head_look_down'
    if 'center head' in search_space or 'look center' in search_space:
        return 'head_center'
    if 'greet' in search_space or 'hello' in search_space:
        return 'greet'
    if 'who are you' in search_space or 'name' in search_space or 'identity' in search_space:
        return 'identity'
    if 'wellbeing' in search_space or 'how are you' in search_space:
        return 'wellbeing'
    if 'help' in search_space:
        return 'help'
    if _contains_any_phrase(
        search_space,
        (
            'who can you see',
            'do you see anyone',
            'who is there',
            'who is here',
            'is anyone there',
            'can you see a person',
        ),
    ):
        return KB_QUERY_VISIBLE_PEOPLE
    if _contains_any_phrase(
        search_space,
        (
            'what objects do you see',
            'what object do you see',
            'which objects are visible',
            'what things do you see',
            'what items do you see',
        ),
    ):
        return KB_QUERY_VISIBLE_OBJECTS
    if _contains_any_phrase(
        search_space,
        (
            'same person as before',
            'same as before',
            'still there',
            'did the scene change',
            'what changed',
            'what was there before',
            'do you still see',
        ),
    ):
        return KB_QUERY_SCENE_CHANGE
    return default


def detect_intent(text: str) -> str:
    """Infer one of the local canonical intents from free text."""
    lowered = text.lower()
    if _contains_any_phrase(lowered, ('look left', 'turn your head left', 'head left')):
        return 'head_look_left'
    if _contains_any_phrase(lowered, ('look right', 'turn your head right', 'head right')):
        return 'head_look_right'
    if _contains_any_phrase(lowered, ('look up', 'head up', 'tilt your head up')):
        return 'head_look_up'
    if _contains_any_phrase(lowered, ('look down', 'head down', 'tilt your head down')):
        return 'head_look_down'
    if _contains_any_phrase(
        lowered,
        (
            'look straight',
            'look center',
            'center your head',
            'head center',
            'head straight',
            'face forward',
        ),
    ):
        return 'head_center'
    if _contains_any_phrase(
        lowered,
        ('stand up', 'get up', 'please stand', 'can you stand', 'stand'),
    ):
        return 'posture_stand'
    if _contains_any_phrase(
        lowered,
        ('sit down', 'take a seat', 'please sit', 'can you sit', 'sit'),
    ):
        return 'posture_sit'
    if _contains_any_phrase(
        lowered,
        ('kneel down', 'kneel', 'crouch', 'on your knees', 'seiza'),
    ):
        return 'posture_kneel'
    if _contains_any_phrase(lowered, ('hello', 'hi', 'hey')):
        return 'greet'
    if _contains_any_phrase(lowered, ('how are you',)):
        return 'wellbeing'
    if _contains_any_phrase(lowered, ('your name', 'who are you')):
        return 'identity'
    if _contains_any_phrase(lowered, ('help',)):
        return 'help'
    if _contains_any_phrase(
        lowered,
        (
            'who can you see',
            'do you see anyone',
            'who is there',
            'who is here',
            'is anyone there',
        ),
    ):
        return KB_QUERY_VISIBLE_PEOPLE
    if _contains_any_phrase(
        lowered,
        (
            'what objects do you see',
            'what object do you see',
            'which objects are visible',
            'what things do you see',
            'what items do you see',
        ),
    ):
        return KB_QUERY_VISIBLE_OBJECTS
    if _contains_any_phrase(
        lowered,
        (
            'same person as before',
            'same as before',
            'still there',
            'did the scene change',
            'what changed',
            'what was there before',
            'do you still see',
        ),
    ):
        return KB_QUERY_SCENE_CHANGE
    return 'fallback'


def build_rule_response(intent: str) -> str:
    """Build rule-based verbal acknowledgement for the fallback path."""
    if intent == 'posture_stand':
        return 'Sure. I am switching to a standing posture.'
    if intent == 'posture_sit':
        return 'Sure. I am switching to a sitting posture.'
    if intent == 'posture_kneel':
        return 'Sure. I am switching to a kneeling posture.'
    if intent == 'head_look_left':
        return 'Sure. I am turning my head to the left.'
    if intent == 'head_look_right':
        return 'Sure. I am turning my head to the right.'
    if intent == 'head_look_up':
        return 'Sure. I am tilting my head up.'
    if intent == 'head_look_down':
        return 'Sure. I am tilting my head down.'
    if intent == 'head_center':
        return 'Sure. I am centering my head.'
    if intent == 'greet':
        return 'Hello! Nice to meet you.'
    if intent == 'wellbeing':
        return 'I am doing well. Thank you for asking.'
    if intent == 'identity':
        return 'I am your NAO chatbot backend.'
    if intent == 'help':
        return (
            'You can greet me, ask my name, ask how I am, or ask for posture '
            'changes like stand, kneel, or sit, and head movements like look '
            'left, right, up, down, or center.'
        )
    if intent == KB_QUERY_VISIBLE_PEOPLE:
        return 'I will answer using the people I can currently detect.'
    if intent == KB_QUERY_VISIBLE_OBJECTS:
        return 'I will answer using the objects I can currently confirm.'
    if intent == KB_QUERY_SCENE_CHANGE:
        return 'I will compare the current scene with the recent scene memory.'
    return 'I heard you. We are testing the dialogue backend.'


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        words = [re.escape(part) for part in phrase.split()]
        pattern = '\\b' + '\\s+'.join(words) + '\\b'
        if re.search(pattern, text):
            return True
    return False
