# chatbot_llm

`chatbot_llm` is the ROS4HRI chatbot backend used by the NAO stack. It keeps the
public `dialogue_manager` backend contract while adding local grounding,
intent-declaration, and planner-routing behavior.

This package is editable for the current planner work. Keep changes aligned with
the upstream backend contract.

## Owns

- `chatbot_msgs/action/Dialogue` backend action.
- `chatbot_msgs/srv/DialogueInteraction` backend service.
- prompt construction, history, and Ollama-compatible transport.
- KnowledgeCore snapshot injection through `kb_skills`.
- direct intent extraction and planner request publication.

It does not own robot execution, final skill dispatch, or planner supervision.

## Public ROS API

| Interface | Type | Purpose |
| --- | --- | --- |
| `<prefix>/start_dialogue` | `chatbot_msgs/action/Dialogue` | Open backend dialogue |
| `<prefix>/dialogue_interaction` | `chatbot_msgs/srv/DialogueInteraction` | Process one user turn |
| `/planner/request` | `hri_actions_msgs/msg/Intent` | Planner ingress when planner mode is enabled |

## Planner Role

`chatbot_llm` should declare intent and route execution turns. The planner should
not have to re-parse raw user language as its primary API.

Preferred planner request inputs:

- `goal_text`: concise task goal for the planner.
- `normalized_intents`: strict intent labels.
- `scene_targets`: grounded labels/entities.
- `grounded_context`: KB/scene/world context.
- `requested_plan`: optional hint/fallback.

The current implementation publishes `goal_text`, `normalized_intents`,
`requested_plan`, and `grounded_context`. It deliberately omits raw
`user_text` from normal planner requests.

## Knowledge Snapshot Role

`knowledge_snapshot` is local prompt context, not a native KnowledgeCore object.

Current path:

```text
knowledge_core -> /kb/query -> kb_skills -> chatbot_llm
chatbot_llm -> formatted knowledge_snapshot -> response/intent prompts
```

Default query group:

```text
myself sees ?entity && ?entity rdf:type ?type
```

## Important Parameters

Defaults live in `config/00-defaults.yml`.

- `server_url`
- `model`
- `intent_model`
- `think`: forwarded as Ollama `think=false/true`; default is `false`.
- `response_max_tokens`: forwarded to Ollama as `num_predict` for the response
  stage; default is `64`.
- `intent_max_tokens`: forwarded to Ollama as `num_predict` for the intent
  stage; default is `64`.
- `planner_mode_enabled`
- `planner_request_topic`
- `planner_request_intent`
- `planner_scene_summary_topic`
- `planner_world_model_snapshot_topic`
- `planner_world_model_text_topic`
- `knowledge_enabled`
- `knowledge_query_service_name`
- `knowledge_default_query_groups`
- `knowledge_max_results`
- `knowledge_max_chars`
- `scene_memory_turns`

The default response and intent model is currently `qwen3.5:397b-cloud` with
`think: false`. Keep the generation caps low for spoken dialogue latency; raise
`response_max_tokens` only when the turn genuinely needs a longer utterance.

## Tests

```bash
cd src/chatbot_llm
PYTHONPATH="$PWD:../planner_common:../kb_skills" \
python3 -m pytest -q test/test_planner_request_adapter.py
```

## Design Notes

- Keep `dialogue_manager` as the dialogue/speaking owner.
- Keep `planner_llm` as the planner/supervisor.
- Keep `nao_orchestrator` as the executor.
- Use `say` as a plan step when speech order matters.
