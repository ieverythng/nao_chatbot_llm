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

## Turn Pipeline

One interaction turn follows this path:

1. `dialogue_manager` opens a dialogue action and forwards user turns through
   `dialogue_interaction`.
2. `chatbot_llm` loads bounded chat history and optional role configuration.
3. If knowledge grounding is enabled, it queries `KnowledgeCore` before prompt
   assembly.
4. The response model generates `verbal_ack`.
5. The intent model, or rule fallback, generates structured intent data.
6. `chatbot_llm` returns the spoken answer plus translated HRI intents.

Main implementation modules:

- `node_impl.py`: lifecycle node and ROS contract
- `turn_engine.py`: response and intent generation pipeline
- `knowledge_snapshot_client.py`: read-only `/kb/query` client
- `knowledge_snapshot.py`: role-level settings merge and snapshot formatting
- `ollama_transport.py`: HTTP backend transport
- `prompt_pack.py` and `prompt_builders.py`: prompt loading and assembly
- `skill_catalog.py`: installed skill summary
- `chat_history.py`: bounded dialogue history
- `intent_rules.py` and `intent_adapter.py`: fallback and intent translation

The intent pipeline now preserves a small set of execution-oriented metadata in
`Intent.data` so downstream routing can stay flexible without changing the ROS
message type:

- `ack_text`
- `ack_mode`
- `scene_targets`
- `plan`

The package also preserves scene-query intents as explicit labels:

- `kb_query_visible_people`
- `kb_query_visible_objects`
- `kb_query_scene_change`

## KnowledgeCore Grounding

`knowledge_core` does not directly push prompt text into `chatbot_llm`.
The grounding seam lives inside this package.

Current behavior:

- `chatbot_llm` calls `/kb/query` using `kb_msgs/srv/Query`
- the default query group is:
  `myself sees ?entity && ?entity rdf:type ?type`
- the service returns a JSON-encoded list of binding dictionaries
- `chatbot_llm` formats that result into a bounded text block and appends it to
  the response and intent prompts
- the prompt builder labels it as live scene state, and the node also keeps a
  short recent-scene-memory trail across turns

Example `/kb/query` response payload shape:

```json
[
  {"entity": "face_1", "type": "Person"},
  {"entity": "mug_1", "type": "Mug"}
]
```

Example dialogue-role override:

```json
{
  "knowledge_snapshot": {
    "enabled": true,
    "query_groups": [
      "myself sees ?entity && ?entity rdf:type ?type"
    ],
    "patterns": [
      "myself sees ?entity",
      "?entity rdf:type ?type"
    ],
    "vars": ["?entity", "?type"],
    "models": [],
    "max_results": 40,
    "max_chars": 3000
  }
}
```

If no `knowledge_snapshot` block is provided in `role.configuration`, the node
falls back to its launch defaults.

The live snapshot is complemented by a short recent-scene-memory trail so the
response and intent prompts can reason about what changed across turns, not
only what is visible right now.

Historical note:

- before the local KB grounding work landed, this package had no
  `knowledge_*` parameters, no `/kb/query` client, and no knowledge snapshot
  prompt block
- the injection seam is therefore a local `chatbot_llm` extension, not an
  upstream `knowledge_core` feature

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

The lifecycle node declares conservative built-in defaults and then loads the
effective package defaults from `config/00-defaults.yml`. The table below
reflects the shipped launch defaults.

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
| `max_history_messages` | `20` | Conversation history bound |
| `scene_memory_turns` | `4` | Number of recent scene summaries retained |
| `knowledge_enabled` | `true` | Enable read-only KB grounding by default |
| `knowledge_query_service_name` | `/kb/query` | Query service used for snapshots |
| `knowledge_default_query_groups` | `myself sees ?entity && ?entity rdf:type ?type` | Default grouped KB query |
| `knowledge_default_vars` | `?entity, ?type` | Variables requested from `/kb/query` |
| `knowledge_max_results` | `40` | Snapshot row cap before formatting |
| `knowledge_max_chars` | `3000` | Prompt budget reserved for snapshot text |

## Operational Notes

- this repo is a fork-tracked upstream package; keep ROS-facing changes aligned
  with the upstream `chatbot_llm` contract
- package naming stays `chatbot_llm` even though the local implementation is
  the NAO-specific backend used by this workspace
- robot-side dispatch must stay out of this package; `/intents` consumers belong
  in `nao_orchestrator`
- `plan` generation is advisory metadata for downstream execution, not direct
  robot control from inside `chatbot_llm`

## Verification

```bash
colcon build --packages-select chatbot_llm
colcon test --packages-select chatbot_llm
ros2 launch chatbot_llm chatbot_llm.launch.py --show-args
```

## Development Hooks

This repo ships its own `.pre-commit-config.yaml` so it can be validated
independently from the monorepo workspace hooks.
