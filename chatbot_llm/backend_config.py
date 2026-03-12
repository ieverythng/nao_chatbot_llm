"""Configuration loading for the upstream-aligned chatbot backend."""

from __future__ import annotations

from dataclasses import dataclass

from chatbot_llm.prompt_pack import default_prompt_pack
from chatbot_llm.prompt_pack import load_prompt_pack
from chatbot_llm.skill_catalog import parse_package_list


INTENT_DETECTION_MODES = {'rules', 'llm', 'llm_with_rules_fallback'}

DEFAULT_RESPONSE_PROMPT_ADDENDUM = (
    'Use concise speech suitable for text-to-speech. '
    'Posture requests should map to stand, sit, or kneel motions when relevant.'
)


@dataclass(frozen=True)
class ChatbotConfig:
    """Runtime configuration for the lifecycle chatbot backend."""

    server_url: str
    model: str
    api_key: str
    system_prompt: str
    enabled: bool
    intent_model: str
    request_timeout_sec: float
    first_request_timeout_sec: float
    intent_request_timeout_sec: float
    context_window_tokens: int
    temperature: float
    top_p: float
    fallback_response: str
    max_history_messages: int
    robot_name: str
    persona_prompt_path: str
    response_prompt_addendum: str
    intent_prompt_addendum: str
    environment_description: str
    response_schema: dict
    intent_schema: dict
    identity_reminder_every_n_turns: int
    intent_detection_mode: str
    prompt_pack_path: str
    use_skill_catalog: bool
    skill_catalog_packages: list[str]
    skill_catalog_max_entries: int
    skill_catalog_max_chars: int


def declare_backend_parameters(node) -> None:
    """Declare lifecycle parameters used by the migrated backend."""
    node.declare_parameter('server_url', 'http://localhost:11434/api/chat')
    node.declare_parameter('model', 'llama3.2:1b')
    node.declare_parameter('api_key', '')
    node.declare_parameter('system_prompt', '')

    node.declare_parameter('enabled', True)
    node.declare_parameter('intent_model', '')
    node.declare_parameter('request_timeout_sec', 20.0)
    node.declare_parameter('first_request_timeout_sec', 60.0)
    node.declare_parameter('intent_request_timeout_sec', 10.0)
    node.declare_parameter('context_window_tokens', 4096)
    node.declare_parameter('temperature', 0.2)
    node.declare_parameter('top_p', 0.9)
    node.declare_parameter(
        'fallback_response',
        'I am having trouble reaching my language model right now.',
    )
    node.declare_parameter('max_history_messages', 12)
    node.declare_parameter('robot_name', 'NAO')
    node.declare_parameter('persona_prompt_path', '')
    node.declare_parameter('response_prompt_addendum', DEFAULT_RESPONSE_PROMPT_ADDENDUM)
    node.declare_parameter('intent_prompt_addendum', '')
    node.declare_parameter('environment_description', 'No specific objects described.')
    node.declare_parameter('identity_reminder_every_n_turns', 6)
    node.declare_parameter('intent_detection_mode', 'llm_with_rules_fallback')

    node.declare_parameter('prompt_pack_path', '')
    node.declare_parameter('use_skill_catalog', True)
    node.declare_parameter('skill_catalog_packages', 'communication_skills,nao_skills')
    node.declare_parameter('skill_catalog_max_entries', 16)
    node.declare_parameter('skill_catalog_max_chars', 3000)


def load_backend_config(node) -> ChatbotConfig:
    """Load effective configuration using defaults < prompt pack < explicit params."""
    prompt_pack_path = str(node.get_parameter('prompt_pack_path').value).strip()
    loaded_pack = load_prompt_pack(prompt_pack_path, logger=node.get_logger())
    defaults = default_prompt_pack()

    raw_system_prompt = str(node.get_parameter('system_prompt').value).strip()
    raw_response_addendum = str(node.get_parameter('response_prompt_addendum').value).strip()
    raw_intent_addendum = str(node.get_parameter('intent_prompt_addendum').value).strip()
    raw_environment = str(node.get_parameter('environment_description').value).strip()

    system_prompt = _pick_prompt_value(raw_system_prompt, '', loaded_pack.system_prompt)
    response_prompt_addendum = _pick_prompt_value(
        raw_response_addendum,
        DEFAULT_RESPONSE_PROMPT_ADDENDUM,
        loaded_pack.response_prompt_addendum,
    )
    intent_prompt_addendum = _pick_prompt_value(
        raw_intent_addendum,
        '',
        loaded_pack.intent_prompt_addendum,
    )
    environment_description = _pick_prompt_value(
        raw_environment,
        defaults.environment_description,
        loaded_pack.environment_description,
    )

    intent_detection_mode = str(node.get_parameter('intent_detection_mode').value).strip().lower()
    if intent_detection_mode not in INTENT_DETECTION_MODES:
        node.get_logger().warn(
            'Unsupported intent_detection_mode=%s, defaulting to llm_with_rules_fallback'
            % intent_detection_mode
        )
        intent_detection_mode = 'llm_with_rules_fallback'

    model = str(node.get_parameter('model').value).strip()
    intent_model = str(node.get_parameter('intent_model').value).strip() or model

    return ChatbotConfig(
        server_url=str(node.get_parameter('server_url').value).strip(),
        model=model,
        api_key=str(node.get_parameter('api_key').value).strip(),
        system_prompt=system_prompt,
        enabled=as_bool(node.get_parameter('enabled').value),
        intent_model=intent_model,
        request_timeout_sec=float(node.get_parameter('request_timeout_sec').value),
        first_request_timeout_sec=max(
            0.5,
            float(node.get_parameter('first_request_timeout_sec').value),
        ),
        intent_request_timeout_sec=max(
            0.5,
            float(node.get_parameter('intent_request_timeout_sec').value),
        ),
        context_window_tokens=max(
            256,
            int(node.get_parameter('context_window_tokens').value),
        ),
        temperature=float(node.get_parameter('temperature').value),
        top_p=float(node.get_parameter('top_p').value),
        fallback_response=str(node.get_parameter('fallback_response').value),
        max_history_messages=max(
            0,
            int(node.get_parameter('max_history_messages').value),
        ),
        robot_name=str(node.get_parameter('robot_name').value),
        persona_prompt_path=str(node.get_parameter('persona_prompt_path').value),
        response_prompt_addendum=response_prompt_addendum,
        intent_prompt_addendum=intent_prompt_addendum,
        environment_description=environment_description,
        response_schema=loaded_pack.response_schema,
        intent_schema=loaded_pack.intent_schema,
        identity_reminder_every_n_turns=max(
            0,
            int(node.get_parameter('identity_reminder_every_n_turns').value),
        ),
        intent_detection_mode=intent_detection_mode,
        prompt_pack_path=prompt_pack_path,
        use_skill_catalog=as_bool(node.get_parameter('use_skill_catalog').value),
        skill_catalog_packages=parse_package_list(
            str(node.get_parameter('skill_catalog_packages').value)
        ),
        skill_catalog_max_entries=max(
            0,
            int(node.get_parameter('skill_catalog_max_entries').value),
        ),
        skill_catalog_max_chars=max(
            0,
            int(node.get_parameter('skill_catalog_max_chars').value),
        ),
    )


def as_bool(value) -> bool:
    """Coerce bool-ish parameter values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def coerce_float(value) -> float:
    """Convert value to float with a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pick_prompt_value(current: str, param_default: str, pack_value: str) -> str:
    """Apply precedence defaults < prompt pack < explicit non-default params."""
    if str(current).strip() != str(param_default).strip():
        return str(current).strip()
    return str(pack_value).strip()
