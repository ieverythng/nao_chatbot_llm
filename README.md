# chatbot_llm

`chatbot_llm` is the upstream-aligned chatbot backend used by the migrated
NAO ROS4HRI stack.

It keeps the public ROS contract expected by `dialogue_manager` while using the
local Ollama prompt, history, and intent pipeline internally.

## ROS API

- action: `<prefix>/start_dialogue`
  - type: `chatbot_msgs/action/Dialogue`
- service: `<prefix>/dialogue_interaction`
  - type: `chatbot_msgs/srv/DialogueInteraction`

The default backend prefix is `chatbot_llm`.

## Internal Modules

- `node_impl.py`: lifecycle node and ROS contract
- `turn_engine.py`: response and intent generation pipeline
- `ollama_transport.py`: HTTP backend transport
- `prompt_pack.py` and `prompt_builders.py`: prompt loading and assembly
- `skill_catalog.py`: installed skill summary
- `chat_history.py`: bounded dialogue history
- `intent_rules.py` and `intent_adapter.py`: deterministic fallback and message conversion

## Launch

Standalone:

```bash
ros2 launch chatbot_llm chatbot_llm.launch.py
```

As part of the migrated stack:

```bash
ros2 launch nao_chatbot nao_chatbot_ros4hri_migration.launch.py
```

## Important Parameters

Defaults live in `config/00-defaults.yml`.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `server_url` | `http://localhost:11434/api/chat` | Backend HTTP endpoint |
| `model` | `llama3.2:1b` | Main response model |
| `intent_model` | `""` | Optional dedicated intent model |
| `system_prompt` | built-in default | Main persona/system prompt |
| `prompt_pack_path` | `""` | Optional YAML override for prompt pack |
| `use_skill_catalog` | `true` | Include discovered skills in prompts |
| `intent_detection_mode` | `llm_with_rules_fallback` | Intent extraction strategy |
| `request_timeout_sec` | `20.0` | Response request timeout |
| `intent_request_timeout_sec` | `10.0` | Intent request timeout |
| `max_history_messages` | `12` | Conversation history bound |

## Operational Notes

- this repo is a fork-tracked upstream package; keep ROS-facing changes aligned
  with the upstream `chatbot_llm` contract
- package naming stays `chatbot_llm` even though the local implementation is
  the NAO-specific backend used by this workspace
- robot-side dispatch must stay out of this package; `/intents` consumers belong
  in `nao_orchestrator`

## Verification

```bash
colcon build --packages-select chatbot_llm
colcon test --packages-select chatbot_llm
ros2 launch chatbot_llm chatbot_llm.launch.py --show-args
```

## Development Hooks

This repo ships its own `.pre-commit-config.yaml` so it can be validated
independently from the monorepo workspace hooks.
