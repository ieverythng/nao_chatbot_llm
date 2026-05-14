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
[`dialogue_manager`](../../dialogue_manager/README.md) node, which
owns the overall conversational state of the robot, and is invoked
via the two-endpoint contract defined in `chatbot_msgs`:

- `<chatbot>/start_dialogue` — an action (`chatbot_msgs/Dialogue`) that
  opens a new conversation and remains active until the dialogue ends.
- `<chatbot>/dialogue_interaction` — a service
  (`chatbot_msgs/DialogueInteraction`) called once per *event* in the
  dialogue: a user utterance, a system update, or an echo of what the
  robot has just said.

The chatbot's job is narrow: given the stream of events for an open
dialogue, decide what the robot should say next and which intents
should be emitted. Everything else — who the speaker is, which group
they belong to, when the dialogue ends, persistence across sessions —
is the dialogue_manager's concern.

```
                 +----------------------------+
                 |     dialogue_manager       |
                 |  (authoritative state,     |
                 |   persistence, group       |
                 |   fan-out, summarisation)  |
                 +-------------+--------------+
                               |
                start_dialogue | dialogue_interaction
                               v
                 +----------------------------+
                 |        chatbot_llm         |
                 |   (LLM-context projection, |
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


Authority and the projection model
-----------------------------------

A central design principle: **`dialogue_manager` is the sole authority
on what was said in a dialogue.** `chatbot_llm` never invents history
entries. Its in-memory view of a dialogue is a *projection* of the
service-call stream — a derived cache, valid for the lifetime of the
open action, dropped on dialogue close.

This is a different view of the same conversation as the one
dialogue_manager owns. The two are intentionally not the same:

| Dimension       | dialogue_manager                               | chatbot_llm                                          |
|-----------------|------------------------------------------------|------------------------------------------------------|
| Per-entry keys  | `(timestamp, speaker_id, text)`                | `{role: system/user/assistant, content}`             |
| Speaker model   | ROS4HRI person IDs + groups + `__myself__`     | Collapsed to system / user / assistant               |
| Scope           | All concurrent dialogues + per-person archive  | Just the LLM context of the currently-open dialogues |
| Lifetime        | Persisted across runs                          | In-memory, dropped on dialogue close                 |
| Authority       | Source of truth                                | Cache derived from the service stream                |

The duplication is *structural*, not redundant: the two
representations exist because they answer different questions
("what happened in this conversation across all participants and
sessions?" vs. "what is the LLM context I should send next?"). They
stay consistent because the chatbot never mutates its history except
in direct response to a service call from the dialogue_manager.

A concrete corollary: `chatbot_llm` does **not** eagerly append the
LLM's proposed `verbal_ack` to the history when it returns it to the
caller. The dialogue_manager will subsequently call
`dialogue_interaction` with `user_id=__assistant__` and the
actually-spoken text (after markup stripping, after Say has played,
possibly different from what the LLM proposed) — and *that* call is
what extends the LLM history. If the assistant turn never round-trips
back (because Say was preempted, the dialogue was cancelled, …), the
LLM context simply doesn't include it.


Per-utterance routing
---------------------

`DialogueInteraction.srv` uses the `user_id` field as a multiplexer:

| `user_id`                | Meaning                                                                 | LLM call?      |
|--------------------------|-------------------------------------------------------------------------|----------------|
| Any non-sentinel string  | A real user turn (text is the user's input)                             | Yes (if `response_expected`) |
| `__assistant__`          | The robot's actually-spoken utterance, echoed back for history          | No             |
| `__system__`             | A system/world update (e.g. prior-session summary, environment change)  | No             |

`chatbot_llm` appends one history entry per call (mapped to the
`assistant`/`user`/`system` LLM role), then either calls the LLM or
returns immediately depending on the `user_id` and `response_expected`.


Dialogue lifecycle
------------------

```
                   start_dialogue.action accepted
                                |
                                v
                       +----------------+
                       |  ACTIVE state  | <-----------+
                       +----------------+             |
                                |                     |
                                | dialogue_interaction|
                                |  (any user_id)      |
                                +---------------------+

  Termination paths:
    - dialogue_manager cancels the goal     -> ABORTED
    - chatbot_llm decides the role is done  -> SUCCEEDED with results
    - node deactivates                      -> ABORTED for all open dialogues
```

A dialogue is born when `on_dialog_goal` accepts a `start_dialogue`
action. While active, it accumulates history entries via
`dialogue_interaction` calls. It ends in one of three ways:

1. **External cancel** — the dialogue_manager cancels the action goal
   (because the user walked away, a group dispersed, a higher-priority
   dialogue arrived, …). The chatbot returns a `Dialogue.Result` with
   an `error_msg` describing the cancel.
2. **Role-driven natural conclusion** — the chatbot itself decides
   the dialogue is done. This is delegated to the dialogue's
   `RoleHandler`: for `__default__` this never happens; for `__ask__`
   it happens once the LLM has produced a complete extraction of the
   `result_schema_properties`. The chatbot returns a `Dialogue.Result`
   with a JSON-serialised `results` field.
3. **Node shutdown** — every open dialogue is aborted at
   `on_deactivate`.

In every case, the dialogue's in-memory history is discarded; nothing
of it survives in `chatbot_llm`. Recovery, if any, is the dialogue_
manager's job.


Multiple concurrent dialogues
-----------------------------

`chatbot_llm` accepts an arbitrary number of concurrent dialogues —
necessary because a single utterance from a tracked person can spawn
parallel per-person dialogues for co-members (see DIALOGUE_FLOW in
`dialogue_manager`). Each dialogue keeps its own:

- UUID (the action goal id)
- role + role configuration
- LLM message history
- termination Event and pending `Dialogue.Result`

The set of open dialogues lives in a `DialoguesRegistry` — a
UUID-keyed map protected by a single mutex. The mutex is held only
for brief bookkeeping (insert, look up, mutate history, remove); it
is **never held across an HTTP call to the LLM**. Per interaction,
the relevant history is snapshotted under the lock, the lock is
dropped for the round-trip, then re-acquired to commit any history
extension that the call produced.

Roles do not partition dialogues — two `__ask__` dialogues for two
different people can be active at once, each with its own
configuration and state.


Role handlers
-------------

Per-role behaviour is parameterised through a `RoleHandler` abstraction
rather than scattered `if role == ...` branches. A handler bundles
three small concerns:

- a *system prompt extension* it contributes to the LLM context
  (e.g. instructions to extract a schema for `__ask__`);
- a *parser* configuration if the response model needs role-specific
  fields;
- a decision after each LLM turn: which text to surface as the
  service response, and whether the dialogue is now terminally done
  (carrying a `Dialogue.Result` to return on the action).

Shipped handlers:

| Role         | Handler              | Behaviour                                                                 |
|--------------|----------------------|---------------------------------------------------------------------------|
| `__default__`| `DefaultRoleHandler` | Returns `verbal_ack` verbatim; never self-closes.                         |
| `__ask__`    | `AskRoleHandler`     | Parses `result_schema_properties` from `role.configuration`; instructs the LLM to extract them; self-closes when extraction is complete. |
| anything else| `DefaultRoleHandler` | Treated as default. (Custom handlers can be registered if needed.)        |

Adding a new role means writing a `RoleHandler` subclass and
registering it in the handler dispatch table; nothing in the registry
or in `node_impl` needs to change.


Concurrency model
-----------------

- The node uses a `MultiThreadedExecutor` and a `ReentrantCallbackGroup`
  on the action server. Two action goal callbacks and two service
  callbacks can be in flight at once.
- All access to dialogue state goes through the `DialoguesRegistry`
  lock. The lock guards the dict structure *and* the per-dialogue
  fields (history list, pending result). Per-dialogue locking would
  be a future refinement; for now, a single mutex keeps the model
  simple and is uncontended in practice (one short critical section
  per service call, an order of magnitude less than the LLM round-trip).
- The `Event` on each `Dialogue` is set to wake the action's executor
  loop on cancel or natural conclusion. The executor wakes at least
  every 100 ms anyway to poll `handle.is_cancel_requested`.


Failure modes
-------------

| Failure                                  | Effect                                                                                    |
|------------------------------------------|-------------------------------------------------------------------------------------------|
| LLM server unreachable / 5xx             | Service returns with `error_msg` set; the dialogue stays open. Caller may retry.          |
| LLM returns non-JSON / wrong schema      | Falls back to emitting a `RAW_USER_INPUT` intent containing the raw text.                 |
| `dialogue_interaction` for unknown UUID  | Service returns `error_msg`; nothing is recorded. The action server probably aborted it.  |
| Two concurrent interactions on one UUID  | Serialised at the lock; LLM calls may interleave but each dialogue's history stays valid. |
| Node deactivates with open dialogues     | All open actions are aborted with `error_msg`; in-memory state is discarded.              |


Non-goals
---------

The following are *intentionally* not the chatbot's responsibility,
to keep the surface narrow:

- **Persistence across runs.** dialogue_manager persists per-person
  and per-group conversation history to disk. The chatbot drops
  everything on dialogue close.
- **Group fan-out.** When `alice` speaks in `group_a`, dialogue_manager
  decides whether to push that utterance into alice's individual
  dialogue, the group dialogue, and each co-member's dialogue. The
  chatbot receives independent service calls for each affected
  dialogue.
- **Summarisation.** The optional `<chatbot>/summarize` service
  mentioned in dialogue_manager's docs is a separate concern, not
  required for any of the flows above. It can be added later as an
  independent service handler that consumes a persisted-history JSON
  and returns text.
- **Speech synthesis / markup.** The chatbot returns text; the
  dialogue_manager (and its Say sub-skill) handle TTS, markup,
  closed captions, and the LED/gesture choreography.
