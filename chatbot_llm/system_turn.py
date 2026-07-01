"""System-turn payload extraction and prompt-addendum framing.

This module owns the parsing of internal ``__system__`` turn payloads (planner
completion, planner dialogue act, and execution report) into normalized context
dicts, plus the structural prompt-addendum helpers that frame those internal
wording turns.

Prompt policy and wording remain owned by the canonical prompt pack. The helpers
here only provide structural framing (which canonical rules still apply to a
system wording turn, and the locked-route instruction block); they do not invent
new user-facing route or response policy.
"""

from __future__ import annotations

import json

from chatbot_llm.backend_config import coerce_float
from chatbot_llm.response_fallbacks import _coerce_user_intent
from chatbot_llm.response_fallbacks import _extract_json_object
from chatbot_llm.route_heuristics import _DIALOGUE_ROUTE
from chatbot_llm.route_heuristics import _normalize_route


_SYSTEM_TASK_RESPONSE_MODE_ADDENDUM = """
System wording mode:
- This turn is a closed internal wording task, not a new user request.
- Keep the spoken answer short, natural for text-to-speech, and consistent with
  the configured robot identity.
- Do not apply route admission, planner handoff, or execution-acknowledgement
  policy here. This prompt exists only to word an already-classified system
  payload.
""".strip()
_PLANNER_COMPLETION_RESPONSE_ADDENDUM = """
Planner completion wording task:
- You are wording one spoken completion sentence after robot task execution.
- Use only facts from the planner_completion JSON payload.
- If facts are missing, briefly say no confirmed result was available.
- Return only the normal response JSON with verbal_ack as the TTS-ready sentence.
- Do not mention JSON, planner internals, routes, or policies.
""".strip()
_PLANNER_DIALOGUE_RESPONSE_ADDENDUM = """
Planner dialogue wording task:
- You are wording one spoken robot sentence for a planner dialogue act.
- Use only facts from the planner_dialogue JSON payload.
- Keep it concise and first-person as the robot.
- Return only the normal response JSON with verbal_ack as the TTS-ready sentence.
- Do not mention JSON, planner internals, routes, or policies.
""".strip()
_EXECUTION_REPORT_RESPONSE_ADDENDUM = """
Execution report wording task:
- You are wording a spoken report for robot task execution.
- Use only facts from the execution_report JSON payload.
- If report_role is "intermediate", report only the latest completed action or
  observation needed at this point. Do not summarize future or not-yet-executed
  steps, and do not claim the whole goal is finished.
- For intermediate reports, use latest_result_summary and latest_result_payload
  as the primary evidence. Older steps are context only and must not replace the
  latest completed action.
- Intermediate reports must not preview the next movement. Do not say "next
  object", "another object", "I am now walking", or equivalent continuation
  wording. Report only the arrival or observation that just completed.
- If report_role is "final" or absent, summarize the whole evidence span in
  steps.
- Synthesize related routine steps into natural language when producing a final
  report; do not recite each internal motion or execution result as a separate
  ledger sentence.
- Use goal_text and dialogue_context to produce a coherent continuation and to
  decide which outcomes matter to the user.
- Do not answer with "I finished:" followed by the user's request. Synthesize
  what was actually done from steps, latest_result_summary, and grounded_context.
- Use grounded_context to translate entity handles into meaningful user-facing
  labels and relations when the facts are present.
- When a place_object or bring_object result uses a person or Human as the
  destination, describe it as delivering or handing the object to that person,
  not placing the object on that person. Keep "on" only for support surfaces
  such as tables, shelves, counters, or named places.
- Treat requested_summary as evidence or a wording hint, not as text that must
  be repeated verbatim.
- Use step results as the authority for factual execution claims, respecting
  success or failure flags and the status of the previous steps in the chain.
- Mention relevant observations from scan/perception results.
- Use one report for this report_result call. Preserve useful dialogue context,
  avoid repeated information, and match the report_role.
- Return only the normal response JSON with verbal_ack as the TTS-ready report.
- Do not mention JSON, planner internals, report_result, routes, or policies.
""".strip()


_PLANNER_COMPLETION_KEYS = (
    'goal_text',
    'result_summary',
    'text_hint',
    'requested_intents',
    'result_payload',
)
_PLANNER_DIALOGUE_KEYS = (
    'act',
    'reason',
    'text_hint',
    'slots_needed',
    'context',
    'completion_context',
)
_EXECUTION_REPORT_KEYS = (
    'goal_text',
    'requested_intents',
    'dialogue_context',
    'scene_targets',
    'grounded_context',
    'requested_summary',
    'report_role',
    'future_steps',
    'steps',
    'latest_result_summary',
    'latest_result_payload',
)


# ---------------------------------------------------------------------------
# Prompt-addendum framing helpers
# ---------------------------------------------------------------------------

def _join_prompt_addenda(*parts: str) -> str:
    return '\n\n'.join(str(part).strip() for part in parts if str(part).strip())


def _system_task_response_addendum(base_addendum: str, task_addendum: str) -> str:
    return _join_prompt_addenda(
        _SYSTEM_TASK_RESPONSE_MODE_ADDENDUM,
        _extract_system_task_response_rules(base_addendum),
        task_addendum,
    )


def _locked_route_prompt_block(context: dict) -> str:
    route = _normalize_route(context.get('route', '')) or _DIALOGUE_ROUTE
    payload = {
        'route': route,
        'resolved_intent': str(context.get('resolved_intent', '')).strip(),
        'user_intent': _coerce_user_intent(context.get('user_intent', {})),
    }
    return _join_prompt_addenda(
        'Locked route context:',
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')),
        (
            'The route above was selected by the intent stage before this wording '
            'request. Do not change it. Return the same route in the response JSON. '
            'If route is dialogue or knowledge_query, do not promise physical action. '
            'If route is execution, acknowledge the request in future tense only and '
            'do not claim completion or observations.'
        ),
    )


def _extract_system_task_response_rules(base_addendum: str) -> str:
    sections = _response_addendum_sections(base_addendum)
    parts: list[str] = []

    intro_lines = [
        line.strip()
        for line in sections.get('__intro__', '').splitlines()
        if 'text-to-speech' in line.lower()
    ]
    if intro_lines:
        parts.append('\n'.join(intro_lines))

    perception = sections.get('Perception and knowledge policy', '').strip()
    if perception:
        parts.append('Perception and knowledge policy:\n%s' % perception)

    shared_style = _shared_response_style_rules(sections.get('Response style', ''))
    if shared_style:
        parts.append('Response style:\n%s' % shared_style)

    extracted = _join_prompt_addenda(*parts)
    if extracted:
        return extracted
    return str(base_addendum or '').strip()


def _response_addendum_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {'__intro__': []}
    current = '__intro__'
    for raw_line in str(text or '').splitlines():
        stripped = raw_line.strip()
        if stripped.endswith(':') and stripped and not stripped.startswith('- '):
            current = stripped[:-1]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line.rstrip())
    return {
        key: '\n'.join(value).strip()
        for key, value in sections.items()
        if any(line.strip() for line in value)
    }


def _shared_response_style_rules(section_text: str) -> str:
    shared_lines: list[str] = []
    skip_route_block = False
    for raw_line in str(section_text or '').splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if not skip_route_block:
                shared_lines.append('')
            continue
        if stripped.startswith('- If route="'):
            skip_route_block = True
            continue
        if skip_route_block:
            if stripped.startswith('- ') and not raw_line.startswith('  '):
                skip_route_block = False
            else:
                continue
        if not skip_route_block:
            shared_lines.append(raw_line.rstrip())
    return '\n'.join(shared_lines).strip()


# ---------------------------------------------------------------------------
# System payload context extractors
# ---------------------------------------------------------------------------

def _extract_planner_completion_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('planner_completion', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    normalized = {
        'goal_text': str(context.get('goal_text', '')).strip(),
        'result_summary': str(context.get('result_summary', '')).strip(),
        'text_hint': str(context.get('text_hint', '')).strip(),
        'requested_intents': [
            str(item).strip()
            for item in context.get('requested_intents', [])
            if str(item).strip()
        ]
        if isinstance(context.get('requested_intents', []), list)
        else [],
        'result_payload': context.get('result_payload', {})
        if isinstance(context.get('result_payload', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _PLANNER_COMPLETION_KEYS):
        return {}
    return normalized


def _extract_execution_report_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('execution_report', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    steps = context.get('steps', [])
    normalized_steps = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            normalized_steps.append(
                {
                    'id': str(step.get('id', '')).strip(),
                    'name': str(step.get('name', '')).strip(),
                    'type': str(step.get('type', '')).strip(),
                    'args': step.get('args', {})
                    if isinstance(step.get('args', {}), dict)
                    else {},
                    'status': str(step.get('status', '')).strip(),
                    'reason': str(step.get('reason', '')).strip(),
                    'result_summary': str(step.get('result_summary', '')).strip(),
                    'result_payload': step.get('result_payload', {})
                    if isinstance(step.get('result_payload', {}), dict)
                    else {},
                }
            )
    future_steps = context.get('future_steps', [])
    normalized_future_steps = []
    if isinstance(future_steps, list):
        for step in future_steps:
            if not isinstance(step, dict):
                continue
            normalized_future_steps.append(
                {
                    'id': str(step.get('id', '')).strip(),
                    'name': str(step.get('name', '')).strip(),
                    'type': str(step.get('type', '')).strip(),
                    'args': step.get('args', {})
                    if isinstance(step.get('args', {}), dict)
                    else {},
                    'requires': [
                        str(item).strip()
                        for item in step.get('requires', [])
                        if str(item).strip()
                    ]
                    if isinstance(step.get('requires', []), list)
                    else [],
                }
            )
    normalized = {
        'goal_text': str(context.get('goal_text', '')).strip(),
        'requested_intents': [
            str(item).strip()
            for item in context.get('requested_intents', [])
            if str(item).strip()
        ]
        if isinstance(context.get('requested_intents', []), list)
        else [],
        'dialogue_context': [
            str(item).strip()
            for item in context.get('dialogue_context', [])
            if str(item).strip()
        ]
        if isinstance(context.get('dialogue_context', []), list)
        else [],
        'scene_targets': [
            str(item).strip()
            for item in context.get('scene_targets', [])
            if str(item).strip()
        ]
        if isinstance(context.get('scene_targets', []), list)
        else [],
        'grounded_context': context.get('grounded_context', {})
        if isinstance(context.get('grounded_context', {}), dict)
        else {},
        'requested_summary': str(context.get('requested_summary', '')).strip(),
        'report_role': str(context.get('report_role', '')).strip(),
        'future_steps': normalized_future_steps,
        'steps': normalized_steps,
        'latest_result_summary': str(context.get('latest_result_summary', '')).strip(),
        'latest_result_payload': context.get('latest_result_payload', {})
        if isinstance(context.get('latest_result_payload', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _EXECUTION_REPORT_KEYS):
        return {}
    return normalized


def _extract_planner_dialogue_context(payload: str) -> dict:
    parsed = _extract_json_object(str(payload or '').strip())
    context = parsed.get('planner_dialogue', {}) if isinstance(parsed, dict) else {}
    if not isinstance(context, dict):
        return {}
    normalized = {
        'act': str(context.get('act', '')).strip().lower(),
        'goal_id': str(context.get('goal_id', '')).strip(),
        'plan_id': str(context.get('plan_id', '')).strip(),
        'plan_version': int(coerce_float(context.get('plan_version', 0))),
        'reason': str(context.get('reason', '')).strip(),
        'text_hint': str(context.get('text_hint', '')).strip(),
        'await_user_response': bool(context.get('await_user_response', False)),
        'slots_needed': [
            str(item).strip()
            for item in context.get('slots_needed', [])
            if str(item).strip()
        ]
        if isinstance(context.get('slots_needed', []), list)
        else [],
        'context': context.get('context', {})
        if isinstance(context.get('context', {}), dict)
        else {},
        'completion_context': context.get('completion_context', {})
        if isinstance(context.get('completion_context', {}), dict)
        else {},
    }
    if not any(normalized.get(key) for key in _PLANNER_DIALOGUE_KEYS):
        return {}
    return normalized
