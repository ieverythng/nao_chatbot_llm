^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package chatbot_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.0 (2026-05-14)
------------------
* user utterances now are: <person_id>: "<text>"
* node_impl: expose ~/{get,set}_logger_levels services
  Pass enable_logger_service=True to the LifecycleNode constructor so
  log levels can be flipped at runtime via the standard rclpy logger
  services. Lets callers turn the DEBUG dump in on_dialogue_interaction
  on/off without restarting the node.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* node_impl: DEBUG-level dump of inbound history + LLM message list
  Adds a verbose two-part dump in on_dialogue_interaction, right before
  the LLM call:
  - the inbound chatbot_msgs/Utterance[] received from dialogue_manager
  - the {role, content} message list after build_llm_messages +
  trim_messages have run
  Gated at DEBUG so the dump only shows up on demand; enable with
  `--ros-args --log-level chatbot_llm:=debug`. The `if`-guard
  short-circuits f-string formatting when DEBUG is off — the dump on
  a 50-turn conversation is non-trivial.
  Pairs with the dialogue_manager-side dump landed in parallel: with
  both enabled you can compare what dialogue_manager shipped against
  what the chatbot actually fed to the LLM, pinpointing where a turn
  might drop on the floor.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* Port to stateless DialogueInteraction contract
  The chatbot is now a pure function over (role, summary, history): it
  retains no per-dialogue state between calls. dialogue_manager is the
  sole authority on the conversation history and ships it in full on
  every dialogue_interaction call.
  - node_impl: drop the long-lived action handler + DialoguesRegistry.
  Expose two services:
  * /chatbot/prepare_dialogue (optional warm-up; default no-op),
  * /chatbot/dialogue_interaction (the stateless turn handler).
  on_dialogue_interaction now builds the LLM message list per call
  from the inbound (role, summary, Utterance[] history), invokes the
  role handler, and packs the result into the new response fields
  (dialogue_terminal, results) carried by the rewritten contract.
  - messages: new module that builds the LLM message list from a
  DialogueInteraction request. Maps speaker sentinels
  (Utterance.SYSTEM / Utterance.ASSISTANT) to the LLM system /
  assistant roles, everything else to user. Renders the system-prompt
  template with the last user-turn speaker and appends the role
  handler's extension.
  - role_handlers: handlers are now stateless wrt history — constructed
  with the role configuration only. TurnOutcome carries
  (response_text, dialogue_terminal, results) instead of the old
  Dialogue.Result, so role-driven termination travels on the service
  response rather than an action result.
  - Drop dialogue_state.py (and its tests): with no per-dialogue state,
  there's nothing to register.
  - ARCHITECTURE.md rewritten to describe the stateless model: the
  "projection" framing is gone; the chatbot is now described as a
  pure function of its inbound request. Concurrency, failure modes
  and non-goals sections updated accordingly.
  Pairs with the chatbot_msgs 4.0.0 contract change. The
  dialogue_manager port lands separately.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add AskRoleHandler for the __ask_\_ role
  Information-gathering interviews: the dialogue_manager opens an __ask\_\_
  dialogue with a role.configuration JSON carrying:
  - "question": the prompt the user is answering (already uttered by
  dialogue_manager via Say before the chatbot ever sees a turn);
  - "result_schema_properties": the JSON-schema-shaped fields the
  chatbot is expected to fill in from the user's reply.
  AskRoleHandler builds a system-prompt extension that names the
  question and the schema and instructs the LLM to either ask
  clarifying follow-ups via verbal_ack, or, once it has every required
  field, surface the answer under a new `extracted` field on
  ChatbotResponse. When all required keys are present in `extracted`,
  the handler returns a terminal Dialogue.Result whose `results` field
  is the JSON-serialised answer — the action server then succeeds the
  goal and the dialogue is removed from the registry.
  Implementation notes:
  - ChatbotResponse gains `extracted: Optional[Dict[str, Any]] = None`.
  Default-role dialogues simply ignore it; the field is included in
  the schema sent to the LLM as the `format` constraint so the LLM
  knows it exists.
  - Malformed / non-dict role.configuration JSON is swallowed silently
  and treated as an empty config (which yields the trivial behaviour:
  no prompt extension, no self-close).
  Tests cover the dispatch entry, empty/invalid configuration, the
  extension content, and the three on_llm_response paths
  (missing/partial/complete extraction).
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* rewire node_impl for multi-dialogue and per-utterance routing
  Replace the single-dialogue state on LLMChatbot with a
  DialoguesRegistry. The four action callbacks
  (on_dialog_goal/accept/cancel/execute) now operate on the dialogue
  identified by the action's goal_id, and the dialogue_interaction
  service routes its requests by chatbot_request.dialogue_id to the
  right entry. Multiple concurrent dialogues are supported; each owns
  its own LLM history, its own done-Event, and its own pending Result.
  on_dialog_goal drops the __default_\_-only restriction: any non-empty
  role name is accepted, and per-role policy is delegated to
  handler_for_role (see role_handlers.py). The system prompt template
  gains a $role placeholder so user templates can branch on the role if
  they want.
  DialogueInteraction routing follows the contract documented in
  chatbot_msgs/DialogueInteraction.srv:
  - user_id == __assistant_\_: record the robot's already-spoken text in
  this dialogue's history; no LLM call;
  - user_id == __system_\_: record a system/world update; no LLM call;
  - otherwise: a real user turn — append to history, snapshot, call
  the LLM, and process the response through the dialogue's
  RoleHandler.
  The LLM's verbal_ack is *not* eagerly appended to the dialogue's
  history. dialogue_manager is the authority on what the robot actually
  says; it will round-trip the actually-spoken text back to us via
  user_id=__assistant_\_ after the Say sub-skill plays it. A comment in
  the response-handling block explains this.
  Concurrency: the DialoguesRegistry's single lock guards both the dict
  and per-dialogue fields. It is held only for brief bookkeeping, never
  across the LLM HTTP round-trip; the message list is snapshotted under
  the lock and the lock is dropped for the request.
  Diagnostics expose the number of currently-open dialogues and their
  ids alongside the cumulative request count.
  See docs/ARCHITECTURE.md for the broader design.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* fix docstring style in dialogue_state and role_handlers
  ament_pep257 enforces D213 (multi-line docstring summary on the second
  line) and D401 (imperative mood on method docstrings). Reformat the
  docstrings in dialogue_state.py and role_handlers.py to comply.
  Pure formatting; no behaviour change.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add RoleHandler abstraction with DefaultRoleHandler
  A RoleHandler bundles the per-role pieces that the rest of the node
  should not need to know about:
  - a system_prompt_extension() to append role-specific instructions to
  the LLM context;
  - an on_llm_response() that decides what text to surface to the
  caller and whether the dialogue has reached its natural conclusion
  (signalled by returning a TurnOutcome with a non-None
  terminal_result).
  DefaultRoleHandler is the passive case used for __default_\_ and any
  unregistered role: surface the verbal_ack verbatim, never self-close.
  The dispatch table in handler_for_role is the single registration
  point — subclasses (e.g. AskRoleHandler in the next commit but one)
  add themselves there.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add dialogue_state module: Dialogue and DialoguesRegistry
  Introduce the data structure that will replace the single-dialogue
  state currently held inline in LLMChatbot. A Dialogue carries its
  own UUID, role, role configuration, LLM-format message history, a
  pending Result, and a done Event that the action server's executor
  loop waits on. A DialoguesRegistry holds a UUID -> Dialogue map
  guarded by a single mutex.
  The registry exposes add (rejecting duplicate ids), get, remove,
  snapshot, plus standard `in` / `len` / `ids` accessors. All access
  goes through the mutex; callers are expected to hold it only for
  brief bookkeeping and never across LLM HTTP round-trips (the
  node-level callers will follow that discipline in the next commit).
  Includes 11 unit tests covering the basic operations and two
  contention smoke tests that exercise concurrent add/remove patterns.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add docs/ARCHITECTURE.md
  Describe the internal design of chatbot_llm:
  - Position in the stack (between dialogue_manager and the LLM HTTP
  endpoint).
  - Authority model: dialogue_manager owns the conversation; chatbot_llm
  holds an in-memory LLM-format projection of the service-call stream,
  dropped on dialogue close.
  - Per-utterance routing through DialogueInteraction's user_id field
  (real user / __assistant_\_ / __system_\_).
  - Dialogue lifecycle and the three termination paths (external cancel,
  role-driven natural conclusion, node shutdown).
  - Multi-dialogue model and the DialoguesRegistry / lock discipline.
  - RoleHandler abstraction with the shipped __default_\_ and __ask\_\_
  handlers, plus the extension point.
  - Failure modes and explicit non-goals (persistence, group fan-out,
  summarisation, TTS/markup all stay with dialogue_manager).
  This documents the target of the multi-dialogue rework; the code that
  matches it lands in the following commits.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* log reason at warn level when start_dialogue goals are rejected
  on_dialog_goal silently rejected goals when another dialogue was
  already active or when the requested role was not '__default_\_',
  which made it hard to debug why clients were being turned away.
  Split the two rejection paths and warn-log the specific reason
  (active dialogue id, or unsupported role name) so the cause is
  visible in the node's output.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* fix import grouping and docstring length in new files
  Address the lint errors that the new modules and tests introduced:
  - Add blank lines between import groups (third-party vs project-local)
  in response_parser.py and the two new test modules, so
  flake8-import-order's I201 is satisfied.
  - Shorten the extract_json_object summary line so it fits under 100
  characters (E501).
  The remaining flake8 noise (Q000 double-quote preference, A003 on
  type/object/input field names) follows the existing codebase
  convention and was already present before the refactor.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add .gitignore for python build artifacts
  Avoid accidentally committing __pycache_\_/ and .pyc files when running
  the test suite locally.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add unit tests for llm_client
  Covers happy-path (first choice extraction, URL composition), auth
  header handling, response_schema forwarding, timeout forwarding, and
  error paths (connection error, timeout, 5xx with and without a
  parseable error body). Uses unittest.mock.patch on requests.post so
  the tests run without a real LLM server.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add unit tests for response_parser
  Covers extract_json_object (prose-wrapped, nested, braces and colons
  inside string values, no-JSON, malformed input) and
  parse_chatbot_response (verbal_ack/user_intent variants, empty object,
  malformed JSON, unknown intent type), plus an intent_to_dict
  round-trip.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* remove dead run() timer; declare pydantic dependency
  The run() method was a no-op invoked once per second by a dedicated
  timer. The chatbot is fully event-driven, so the timer and the empty
  method are removed.
  Also add python3-pydantic to package.xml — it has been used (via
  response models) but was never declared.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* add lock around shared dialogue state
  MultiThreadedExecutor + ReentrantCallbackGroup means the action server
  callbacks (on_dialog_goal/accept/cancel/execute) and the
  DialogueInteraction service callback can race on the same fields:
  _dialogue_id, _dialogue_result, _msgs_history, _nb_requests.
  Guard all reads and writes of that state with a single
  threading.Lock. The lock is intentionally not held across the HTTP
  call in on_dialogue_interaction — we snapshot the message list under
  the lock, drop it for the network round-trip, then re-acquire to
  append the assistant reply.
  Also reset _msgs_history when a new dialogue is accepted, so each
  dialogue starts with a clean conversation rather than inheriting the
  history of whatever dialogue ran previously.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* bound conversation history with max_history_turns parameter
  _msgs_history was appended to on every interaction and never trimmed,
  which would eventually exceed the model's context window and waste
  tokens on stale early turns. Add a `max_history_turns` parameter
  (default 10) and trim after each append, always preserving the system
  prompt at index 0.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* harden JSON extraction; centralize pydantic v1/v2 shims
  preprocess_llm_response counted braces character-by-character to find a
  balanced {...} substring. That logic was unaware of JSON string
  literals, so a `{` or `}` inside a string value would throw the matcher
  off. It also rewrote every `:` to `: ` to work around an old YAML
  parsing assumption — but that substitution corrupts any URL, time, or
  other value containing a colon.
  Replace it with json.JSONDecoder().raw_decode() starting at the first
  `{`. raw_decode understands string literals natively, so we get the end
  of the JSON object correctly regardless of its content. The function is
  renamed to extract_json_object to reflect what it does.
  Also collapse the pydantic v1/v2 branches scattered in the code into
  three small shims in response_parser (parse_chatbot_response,
  chatbot_response_schema, intent_to_dict), each detecting the pydantic
  major version once at import time. The node calls the shims rather than
  sprinkling hasattr checks at each use site.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* extract llm_client and response_parser modules
  Move the HTTP client out of node_impl.py into a new LLMClient class
  (llm_client.py), and the pydantic response models plus the JSON
  extraction helpers into response_parser.py.
  The node now constructs an LLMClient at on_configure and delegates
  chat completion + parsing to the new modules. The remaining
  customization points (dialogue handling, world model, prompt
  template) stay on LLMChatbot since downstream users are expected
  to edit them.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* fix malformed RAW_USER_INPUT fallback JSON
  The fallback path built the Intent `data` field by string-concatenation
  and wrapped the result in an extra pair of curly braces: the produced
  payload looked like \`{ {"input": "...", "suggested_response": "..."} }\`,
  which is not valid JSON and could not be parsed by downstream consumers.
  Build the payload with json.dumps instead.
  The handcrafted escape_json helper that supported the concatenation is
  no longer used, so drop it.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* render system prompt per interaction; add robot_name parameter
  The system prompt template was rendered once at on_configure with a
  hard-coded user_id of "a human" and never updated again. The default
  prompt itself advertised "The user_id of the person you are talking to
  is \$user_id", which means the LLM saw a stale (or placeholder) id
  forever. The \$robot_name placeholder used in the default prompt was
  also never substituted.
  Store the raw Template instead and render it per dialogue interaction,
  substituting the actual user_id of the current speaker. Add a
  `robot_name` parameter (default "robot") so \$robot_name resolves too.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* replace busy-wait in dialogue executor with threading.Event
  on_dialog_execute spun in a 10 ms time.sleep loop waiting for either a
  cancel or a terminal _dialogue_result. Switch to Event.wait(timeout) so
  the executor blocks until something happens (or 100 ms elapses, to keep
  cancel-request polling responsive) without burning a thread at 100 Hz.
  Also drop the now-unused `time` and `yaml` imports.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* fix request error handling and add HTTP timeout
  The except branch in perform_request referenced `response`
  unconditionally, raising UnboundLocalError when the request never
  returned (e.g. connection refused or DNS failure). Initialize
  `response = None` and guard the access.
  When the LLM call fails, the service callback used a bare `return`,
  which made rclpy crash on the missing service response. Return the
  response object with an error_msg set instead.
  Also add a `request_timeout` parameter (default 30s) so the node
  cannot hang indefinitely on a stalled LLM server.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* robustness improvements to LLM response parsing
* handle invalid JSON response from the llm
* disable 'thinking' mode by default
* use pydantic to define and check the expect LLM response format
* Contributors: Séverin Lemaignan

0.1.1 (2026-02-05)
------------------
* initial scaffold, generated with rpk
* Contributors: Séverin Lemaignan
