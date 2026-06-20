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
- `grounded_context`: canonical compact LLM context with `entities[]`.

The current implementation publishes `goal_text`, `normalized_intents`,
and `grounded_context`. It deliberately omits raw `user_text`, `requested_plan`,
and transport-only interaction mode fields from normal planner requests.

## Knowledge Snapshot Role

`knowledge_snapshot` is local prompt context, not a native KnowledgeCore object.
Planner source grounding keeps it JSON-first with compact references so the
projection step can build deterministic LLM context without large free-text blobs.
Grounded scene payloads should keep humans under `people` and reserve
`objects` for non-person detections to avoid contract ambiguity.
Those raw KB and scene inputs are projected into the single compact
`grounded_context.entities[]` object before entering chatbot/planner LLM prompts.

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
- `turn_pipeline_mode`: selects `response_first`, `intent_first`, or the
  isolated one-call `atomic_irr` ablation. The default remains
  `response_first` until paired live testing accepts a new default.
- `irr_max_tokens`: generation cap for the internal `irr.v1` response.
- `irr_turn_state_enabled`: supplies the model with deterministic `ts.v1`
  context assembled from current turn evidence.
- `irr_canonical_guard_enabled`: requires registry-declared skill arguments to
  resolve to canonical entities before an execution turn can reach planner
  admission.
- `irr_subject_lookup_enabled`: optionally enriches the current snapshot for
  unambiguous canonical subjects mentioned by the user. It defaults to false.
- `skill_registry_path`: optional planner registry overlay used alongside the
  shared `skill_common` registry. The integrated launch passes the same path to
  chatbot and planner.
- `planner_mode_enabled`
- `planner_request_topic`
- `planner_request_intent`
- `planner_scene_summary_topic`
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
- `atomic_irr` returns response text on every user turn, but does not decide
  whether or when the robot speaks. `dialogue_manager` remains the utterance
  authority.
- `planner_handoff.requested` is model evidence, not publication authority.
  The deterministic route guard and existing planner publisher decide whether
  a request is emitted.
- Planner-completion, planner-dialogue, execution-report, and report-result
  system turns stay on their existing wording paths and bypass `atomic_irr`.
