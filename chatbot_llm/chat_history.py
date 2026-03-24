"""Conversation-history utilities for the migrated chatbot backend."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Public history conversions
# ---------------------------------------------------------------------------

def coerce_history(raw_history) -> list[str]:
    """Normalize inbound history payload to ``list[str]``."""
    if isinstance(raw_history, list):
        return [str(entry).strip() for entry in raw_history if str(entry).strip()]
    if isinstance(raw_history, str) and raw_history.strip():
        return [raw_history.strip()]
    return []


def trim_messages(messages: list[dict], max_history_messages: int) -> list[dict]:
    """Trim non-system messages while preserving one leading system message."""
    if max_history_messages <= 0:
        return list(messages)
    if not messages:
        return list(messages)

    system_prefix = []
    non_system_messages = []
    for message in messages:
        if (
            not system_prefix
            and message.get('role', '') == 'system'
            and str(message.get('content', '')).strip()
        ):
            system_prefix = [message]
            continue
        non_system_messages.append(message)

    if len(non_system_messages) > max_history_messages:
        non_system_messages = non_system_messages[-max_history_messages:]
    return system_prefix + non_system_messages


def history_to_messages(history_entries, max_history_messages: int) -> list[dict]:
    """Convert serialized history entries (``role:text``) to chat messages."""
    messages = []
    for index, entry in enumerate(history_entries):
        clean = str(entry).strip()
        if not clean:
            continue

        role = ''
        content = ''
        if ':' in clean:
            role_part, content_part = clean.split(':', 1)
            maybe_role = role_part.strip().lower()
            if maybe_role in {'system', 'user', 'assistant'}:
                role = maybe_role
                content = content_part.strip()

        if not role:
            role = 'user' if index % 2 == 0 else 'assistant'
            content = clean

        if content:
            messages.append({'role': role, 'content': content})

    return trim_messages(messages, max_history_messages=max_history_messages)


def messages_to_history(messages: list[dict], max_history_messages: int) -> list[str]:
    """Serialize chat messages back to compact role-prefixed history entries."""
    serialized = []
    for message in trim_messages(messages, max_history_messages=max_history_messages):
        role = str(message.get('role', '')).strip().lower()
        content = str(message.get('content', '')).strip()
        if role not in {'system', 'user', 'assistant'}:
            continue
        if not content:
            continue
        serialized.append(f'{role}:{content}')
    return serialized
