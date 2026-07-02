Architecture of chatbot_llm
============================

This document describes how `chatbot_llm` is organised internally and,
more importantly, how it fits into the dialogue-management stack that
drives it. The
[README](../README.md) covers the user-facing ROS API and configuration;
this document focuses on the *model* — what state lives where, who is
authoritative, and why.

The intended reader is a developer extending `chatbot_llm` (adding a
new dialogue role, swapping the LLM backend, changing the response
schema), or someone trying to understand the protocol between
`dialogue_manager` and the chatbot.


Position in the stack
---------------------

`chatbot_llm` is a *chatbot backend*. It sits behind the
[`dialogue_manager`](https://gitlab.iiia.csic.es/socialminds/ros4hri/dialogue_manager/README.md) node, which
owns the overall conversational state of the robot, and is invoked
via the two-endpoint contract defined in `chatbot_msgs`:

- `<chatbot>/prepare_dialogue` — an *optional* service
  (`chatbot_msgs/PrepareDialogue`) that signals a new dialogue is
  about to start. Backends may use it to warm up role-specific
  resources (load a model, configure a downstream semantic
  pipeline, register a session id elsewhere). A backend that has
  nothing to warm up may treat this as a no-op; a dialogue_manager
  that has nothing to gain may skip the call entirely.
- `<chatbot>/dialogue_interaction` — the main service
  (`chatbot_msgs/DialogueInteraction`) called once per user-driven
  turn. Each call carries the full role + history; the backend
  computes the next response and returns.

```
                 +----------------------------+
                 |     dialogue_manager       |
                 |  (authoritative state,     |
                 |   persistence, active-     |
                 |   dialogue tracking,       |
                 |   summarisation)           |
                 +-------------+--------------+
                               |
              prepare_dialogue | dialogue_interaction
                               v
                 +----------------------------+
                 |        chatbot_llm         |
                 |   (stateless turn handler, |
                 |    role-specific policy)   |
                 +-------------+--------------+
                               |
                  OpenAI-compatible HTTP API
                               v
                       +---------------+
                       |   LLM server  |
                       | (ollama, ...) |
                       +---------------+
```


Statelessness and the single-source-of-truth model
--------------------------------------------------

A central design principle: **`dialogue_manager` is the sole authority
on what was said in a dialogue, and the chatbot retains no per-dialogue
state between calls.** Every `dialogue_interaction` request carries:

- the `dialogue_id` (for logging / correlation only — never used to
  look up retained state),
- the `role` and its opaque JSON `configuration`,
- an optional prior-session `summary`,
- the full conversation `history` (a `chatbot_msgs/Utterance[]`)
  including every robot utterance, every system/world update, and
  the latest user turn that triggered the call.

The backend assembles the LLM message list freshly each turn, calls
the LLM, runs the role handler over the parsed response, and returns.
Nothing persists in `chatbot_llm` between calls.

| Dimension       | dialogue_manager                               | chatbot_llm                                           |
|-----------------|------------------------------------------------|-------------------------------------------------------|
| Per-entry keys  | `(timestamp, speaker_id, text)`                | `{role: system/user/assistant, content}` (per-call)   |
| Speaker model   | ROS4HRI person/group ids + `__myself__`        | Collapsed to system / user / assistant                |
| Scope           | Active dialogue per person + per-person archive | The current call only                                 |
| Lifetime        | Persisted across runs                          | None — each call is independent                       |
| Authority       | Source of truth                                | Pure function of the inbound request                  |

Two practical consequences of statelessness:

1. There is no need for an `__assistant__` round-trip: the robot's
   actually-spoken text is appended to the history by dialogue_manager
   and shows up in the *next* `dialogue_interaction` request alongside
   any new user turn. `chatbot_llm` does not need to be told "the
   robot just said this"; the next call will carry it.
2. Server-side prompt caching (Anthropic, OpenAI) benefits naturally:
   identical message prefixes hit the cache on every successive turn.
   The marginal cost of sending the full history each call is far
   smaller than the cost of the LLM round-trip itself.


Per-utterance routing (inside the message builder)
--------------------------------------------------

`chatbot_msgs/Utterance` carries a `speaker` field. The message builder
in `messages.py` maps each utterance into the LLM-side role:

| `speaker`                          | Mapped to LLM role | Content shape                |
|------------------------------------|--------------------|------------------------------|
| `Utterance.SYSTEM` (`__system__`)  | `system`           | `text` verbatim              |
| `Utterance.ASSISTANT` (`__assistant__`) | `assistant`   | `text` verbatim              |
| any other (a real user id)         | `user`             | `'<speaker> "<text>"'`       |

The leading message is always a freshly-rendered system prompt for the
current role (with the role handler's extension appended). If a
prior-session `summary` is provided, it follows as a second `system`
message.


No-dialogue lifecycle
---------------------

There is no longer a long-lived dialogue state inside `chatbot_llm`.
The two-state diagram below is the entire model:

```
   prepare_dialogue (optional)              dialogue_interaction (N times)
            |                                          |
            v                                          v
   warm up role resources                     compute one LLM turn,
   (default: no-op)                           return response + intents
                                              + dialogue_terminal flag
```

Role-driven termination — previously carried on the action's
`Dialogue.Result` — now travels on the `dialogue_interaction` response:

- `bool dialogue_terminal`: when true, the backend considers the
  dialogue naturally complete. `dialogue_manager` is responsible for
  actually closing the dialogue (cleaning up its own state, archiving,
  summarising); the chatbot has nothing more to clean up.
- `string results`: JSON-encoded structured outcome (e.g. for
  `__ask__`, the schema extraction). Empty when not terminal.

External cancellation is implicit: `dialogue_manager` simply stops
calling `dialogue_interaction`. There is no goal to cancel.


Multiple concurrent dialogues
-----------------------------

Concurrency is no longer a property of `chatbot_llm`'s state — it has
no state. Two `dialogue_interaction` calls for different
`dialogue_id`s simply run as two independent service invocations. The
`ReentrantCallbackGroup` on the service callback allows them to be
in-flight at the same time; each call's history snapshot lives in
local variables for the duration of that one call.

This is a strict simplification over the previous registry+lock model:
there is no shared mutable state to guard.


Role handlers
-------------

Per-role behaviour is parameterised through a `RoleHandler` abstraction
rather than scattered `if role == ...` branches. A handler is
constructed fresh per call with the role configuration string and
bundles three small concerns:

- a *system prompt extension* it contributes to the LLM context
  (e.g. instructions to extract a schema for `__ask__`);
- a *parser* configuration if the response model needs role-specific
  fields;
- a decision after each LLM turn: which text to surface as the
  service response, whether the dialogue is now terminally done
  (the `dialogue_terminal` flag) and the role-specific
  JSON-encoded `results` to surface alongside.

Shipped handlers:

| Role         | Handler              | Behaviour                                                                 |
|--------------|----------------------|---------------------------------------------------------------------------|
| `__default__`| `DefaultRoleHandler` | Returns `verbal_ack` verbatim; never self-closes.                         |
| `__ask__`    | `AskRoleHandler`     | Parses `result_schema_properties` from `role.configuration`; instructs the LLM to extract them; signals `dialogue_terminal=True` with JSON `results` once extraction covers every required key. |
| anything else| `DefaultRoleHandler` | Treated as default. (Custom handlers can be registered if needed.)        |

Because handlers are stateless wrt. the dialogue history, each call
re-derives their view of "what's been collected so far" from the
parsed LLM response. Adding a new role means writing a `RoleHandler`
subclass and registering it in the dispatch table; nothing in
`node_impl` needs to change.


Concurrency model
-----------------

- The node uses a `MultiThreadedExecutor` and a `ReentrantCallbackGroup`
  on both services. Multiple `prepare_dialogue` and
  `dialogue_interaction` calls may be in flight concurrently.
- There is no shared dialogue state to lock. The two short counter
  fields used for diagnostics (`_nb_requests`, `_nb_prepared`) are
  guarded by a tiny dedicated mutex.
- LLM round-trips are not serialised: two concurrent service calls
  result in two concurrent HTTP requests to the LLM server. The
  server (or the LLM client's HTTP pool) is the natural bottleneck.


Failure modes
-------------

| Failure                                  | Effect                                                                                    |
|------------------------------------------|-------------------------------------------------------------------------------------------|
| LLM server unreachable / 5xx             | Service returns with `error_msg` set; the dialogue is not advanced. dialogue_manager may retry on its terms. |
| LLM returns non-JSON / wrong schema      | Falls back to emitting a `RAW_USER_INPUT` intent containing the raw text.                 |
| `dialogue_interaction` with empty history | Service returns `error_msg`; the caller is expected to ship at least the triggering user turn. |
| Malformed `dialogue_id`                  | Service returns `error_msg`; nothing is recorded (there's nothing to record).             |
| Concurrent interactions on one UUID      | Independent calls; the LLM may interleave them — dialogue_manager (which owns history)    |
|                                          | is responsible for not issuing concurrent turns it doesn't want interleaved.              |
| Node deactivates                         | Services are torn down; in-flight calls return gracefully on the executor.                |


Non-goals
---------

The following are *intentionally* not the chatbot's responsibility,
to keep the surface narrow:

- **Per-dialogue history.** dialogue_manager keeps the canonical
  history and ships it on every call. The chatbot never persists
  anything between calls.
- **Persistence across runs.** dialogue_manager persists per-person
  and per-group conversation history to disk; the chatbot is a pure
  function.
- **Active-dialogue tracking.** Whether `alice` is currently part of
  a group dialogue or a solo dialogue is a dialogue_manager decision;
  the chatbot only sees the history of whatever dialogue it has been
  asked to advance.
- **Summarisation.** An optional `<chatbot>/summarize` service is a
  separate concern; it would consume a persisted-history JSON and
  return text. dialogue_manager ships any available summary in the
  `summary` field of `dialogue_interaction`.
- **Speech synthesis / markup.** The chatbot returns text; the
  dialogue_manager (and its Say sub-skill) handle TTS, markup,
  closed captions, and the LED/gesture choreography.
