# ReAct Loop

Procedural layer reference — expands section H of [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).
Owner: Person 3.

The procedural layer is the only layer that talks to the LLM. It never touches
the belief graph or the world simulation directly: every read and write crosses
a layer boundary as JSON through `KnowledgeAPI` or `SensorimotorAPI`.

```
query ──► run_react ──► llm.chat_with_tools(messages, specs)
             │                      │
             │                      └── tool_calls
             ▼
          dispatch(name, args, apis) ──► KnowledgeAPI   (declarative)
             │                      └──► SensorimotorAPI (sensorimotor)
             ▼
          Step(thought, tool_call, observation) ──► Trace ──► data/runs/*.json
```

---

## 1. Module map

| Module | Responsibility |
|---|---|
| `agent.py` | `Agent(knowledge, sensors, llm, config)`; resolves config knobs; summarizes the `Trace` into an `AgentResult`. |
| `react_loop.py` | `run_react(...)` — the loop, the finalize ordering rule, the context guard, `MaxStepsExceeded`. |
| `tools.py` | `TOOL_SPECS` (the 10), `ACTUATOR_SPECS` (gated), `build_tool_specs()`, `dispatch()`. |
| `llm_client.py` | `LLMClient` (Groq) and `RecordedLLMClient` (deterministic replay). |
| `prompts.py` | `SYSTEM_PROMPT` (section H.3, verbatim), `format_query()`. |
| `trace.py` | `trace_to_log()`, `dump_trace()` — the section K log format. |
| `models.py` | `ToolCall`, `Observation`, `Step`, `Trace`, `AgentResult`. |

---

## 2. The loop, as implemented

```python
messages = [system_prompt, user_query]
trace    = Trace(query=..., scenario=..., started_at=now)

for step_idx in 1..max_steps:
    messages = truncate_context(messages, max_messages, token_budget)
    response = llm.chat_with_tools(messages, build_tool_specs(enable_actuators))

    if not response.tool_calls:                  # free text only
        trace.append(Step(idx, thought=response.content))
        messages += [assistant(thought), user(NUDGE)]
        continue

    messages.append(assistant_with_all_tool_calls(response))   # ONE message

    for tc in response.tool_calls if tc.name != "finalize_answer":
        result = dispatch(tc.name, tc.arguments, apis)
        trace.append(Step(idx, tool_call=tc, observation=result))
        messages.append(tool_result(tc.id, result))

    for position, tc in enumerate(finalize_calls):
        result = dispatch(...) if position == 0 else DUPLICATE_FINALIZE_ERROR
        trace.append(...); messages.append(tool_result(tc.id, result))
        if position == 0 and result.ok:
            trace.final_answer = result.data.answer
            return trace                                       # episode ends

trace.ended_at = now
dump partial trace if runs_dir
raise MaxStepsExceeded(trace, dump_path)
```

### Five decisions worth stating explicitly

1. **One assistant message per response, not per call.** The provider's
   contract is: one assistant turn listing every `tool_call`, then one
   `role="tool"` message per `tool_call_id`. Interleaving a tool result
   between two tool calls of the same turn produces a request the provider
   rejects.
2. **Free text is a thought, not an answer.** A response with no tool call is
   logged as `Step(thought=...)` and answered with
   *"You must call a tool to continue the episode. Use `finalize_answer` to
   end."* The agent may never assert a fact it did not read through a tool, so
   prose alone cannot end an episode.
3. **`finalize_answer` executes last.** Emission order belongs to the model;
   execution order belongs to us. Every non-terminal call in a turn runs
   first, so a model that batches `[finalize_answer, update_belief]` still
   commits the belief. A *failed* `finalize_answer` (e.g. no `answer` field)
   does not end the episode — the loop continues so the model can retry. A
   duplicate `finalize_answer` in the same turn is refused with
   `{"ok": false, "error": {"kind": "duplicate_finalize"}}`, and the first
   answer stands.
4. **Exceeding `max_steps` is not a silent failure.** The partial trace is
   dumped (when a runs directory is configured) and attached to
   `MaxStepsExceeded`, along with `dump_path`. Half an episode is evidence;
   losing it is not an option when the deliverable is a trace log.
5. **Truncation never breaks a tool group.** See §6.

---

## 3. The tool registry

Ten tools, exactly as specified in section G.4. `finalize_answer` is handled by
the loop, not by a façade.

| Tool | Layer | Arguments (required in **bold**) | Returns |
|---|---|---|---|
| `get_belief` | declarative | **subject**, **predicate**, perspective | `Belief \| null` |
| `query_perspective` | declarative | **subject**, predicate | `PerspectiveView[]` |
| `query_provenance` | declarative | **subject**, predicate, limit | `ProvenanceEntry[]` |
| `list_sources` | declarative | — | `Source[]` with `trust_prior` |
| `update_belief` | declarative | **subject**, **predicate**, **object**, **confidence**, **source**, **perspective**, **reason** | `ProvenanceEntry` |
| `downgrade_belief` | declarative | **subject**, **predicate**, **new_confidence**, **reason**, perspective | `ProvenanceEntry` |
| `sense_lidar` | sensorimotor | direction (`front`\|`left`\|`right`\|`back`) | `SensorPayload` |
| `sense_camera` | sensorimotor | direction | `SensorPayload` |
| `sense_clock` | sensorimotor | — | `SensorPayload` |
| `finalize_answer` | procedural | **answer**, cited_beliefs, cited_sensors | *ends the episode* |

### `perspective` is required on `update_belief`

This is the anti-merging guardrail (plan section J, guardrail 1). On reads the
perspective defaults to `"self"`; on a **write** it is mandatory and the
dispatcher does not supply a default. Omitting it returns

```json
{"ok": false, "error": {"kind": "missing_argument", "message": "Missing required argument: 'perspective'."}}
```

and writes nothing. A belief filed under the wrong perspective silently merges
two epistemic viewpoints — exactly the failure the architecture exists to
prevent — so the loop would rather make the model say it than guess.

### Actuators are gated

`act_move`, `act_rotate` and `act_grasp` exist in `ACTUATOR_SPECS` but are
**not offered to the LLM** and are refused by the dispatcher unless
`features.enable_actuators` is true in `configs/default.yaml` (default
`false`):

```json
{"ok": false, "error": {"kind": "tool_disabled", "message": "'act_move' is disabled. ..."}}
```

An agent that can move changes the world between a sensor read and a belief
write, which makes the 10 test logs non-reproducible. Note the distinction a
model can act on: a gated tool is `tool_disabled`, an invented one is
`unknown_tool`.

### The observation envelope

Every dispatch returns one of two shapes, and **never raises into the loop**:

```json
{"ok": true,  "data": { ... }}
{"ok": false, "error": {"kind": "...", "message": "..."}}
```

| `kind` | Means |
|---|---|
| `unknown_tool` | No such tool. |
| `tool_disabled` | Actuator called with actuators off. |
| `missing_argument` | A required argument was absent. |
| `invalid_argument` | Pydantic rejected the value (e.g. `confidence: 1.5`). |
| `knowledge_error` | `KnowledgeError` from the declarative layer. |
| `not_implemented` | A façade method that hasn't landed yet. |
| `tool_error` | Anything else, with the exception type in the message. |
| `duplicate_finalize` | Second `finalize_answer` in one turn. |

Errors are observations, not crashes: the model sees the failure and can
recover inside the same episode.

---

## 4. The system prompt

`prompts.py` holds section H.3 verbatim. Its four load-bearing rules:

1. Never assert a fact not verified via a tool.
2. On a memory-vs-sensor conflict, prefer the **higher `trust_prior`** from
   `list_sources`; call `downgrade_belief` on the loser and `update_belief`
   with the winner.
3. Keep **user / self (egocentric) / third-party** views strictly separate —
   never merge them.
4. Call `finalize_answer` **exactly once**, citing the beliefs and sensor
   readings relied on. Cap: 8 tool calls.

`format_query(query, scenario)` prefixes `[scenario=A]` when a scenario is
named, so a trace is self-describing.

---

## 5. Configuration

`configs/default.yaml`, with env vars (`.env`) overriding matching keys:

| Key | Default | Used by |
|---|---|---|
| `llm.model` | `openai/gpt-oss-20b` | `LLMClient` (`LLM_MODEL`) |
| `llm.temperature` | `0.0` | `LLMClient` (`LLM_TEMPERATURE`) — 0 for determinism |
| `llm.max_tokens` | `1024` | `LLMClient` (`LLM_MAX_TOKENS`) |
| `llm.timeout_s` / `llm.max_retries` | `30` / `3` | `LLMClient` |
| `react.max_steps` | `8` | `Agent.run` |
| `react.token_budget` | `6000` | `truncate_context` |
| `react.max_context_messages` | `8` | `truncate_context` |
| `paths.runs_dir` | `./data/runs` | where traces land |
| `logging.json_traces` | `true` | set false to stop writing traces |
| `features.enable_actuators` | `false` | `build_tool_specs` + `dispatch` |

Precedence is explicit keyword → environment → YAML → module default:

```python
llm   = LLMClient.from_config(config)          # or LLMClient() to use .env alone
agent = Agent(knowledge, sensors, llm, config)
```

`Agent(knowledge, sensors, llm)` with **no config** works too — and then
writes no trace file, since there is no runs directory to write to. Pass
`agent.run(query, runs_dir=...)` to override per call. This is why the HTTP
layer does not litter one JSON file per request.

---

## 6. Guardrails

**Step cap.** `max_steps` (default 8) → `MaxStepsExceeded(trace, dump_path)`.
The partial trace always survives.

**Context guard (H.5).** While the estimated size of the message log is within
`token_budget` the log is passed through untouched. Above it, only the most
recent `max_context_messages` are kept, under two invariants:

- the **system prompt always survives** — it carries every epistemic rule;
- an assistant turn carrying `tool_calls` is **never separated** from the
  `role="tool"` messages that answer it. Messages are grouped into atomic
  units and kept or dropped whole; the newest group is kept even if it alone
  exceeds the message budget, because half a tool group is a request the
  provider refuses.

Token estimation is deliberately crude (serialized JSON length ÷ 4) — a real
tokenizer would couple the loop to a model id.

**Malformed generations.** Small instruct-tuned models sometimes emit a tool
call whose `arguments` are not valid JSON, or a function with no name.
`LLMClient` flags that response `malformed=True` and **resamples** it, up to
`max_retries`, with exponential backoff (0.5s, 1s, 2s). If every attempt comes
back malformed the last response is returned rather than raised: the
unparseable arguments land under `__raw__`, the dispatcher reports
`missing_argument`, and the model gets a chance to fix itself. Aborting a whole
episode over one bad generation would be worse.

**Determinism.** `temperature=0.0`, and every test drives
`RecordedLLMClient` so CI never needs an API key.

---

## 7. `RecordedLLMClient`

Deterministic replay. Same interface as `LLMClient`
(`chat_with_tools(messages, tools) -> AssistantResponse`), so the loop cannot
tell them apart. **CI and all 10 epistemic tests depend on this class** — it is
importable as:

```python
from iagent.procedural.llm_client import RecordedLLMClient
```

### Constructor

```python
RecordedLLMClient(script: Iterable[dict | list[dict]])
```

One entry per **turn**, consumed in order. An entry is:

| Entry form | Produces |
|---|---|
| `{"tool": "<name>", "arguments": {...}}` | one tool call |
| `{"tool": ..., "arguments": ..., "id": "call_7"}` | one tool call with a fixed id |
| `[{...}, {...}]` | several tool calls **in one turn** (batched) |
| `{"content": "free text"}` | an assistant turn with no tool call (a thought) |

Attributes and methods:

| Member | Purpose |
|---|---|
| `.calls` | `[{"messages": [...], "tools": [...]}, ...]` — one entry per turn, for asserting what the loop actually sent |
| `.remaining()` | script entries not yet consumed |
| `ScriptExhausted` | raised when the loop asks for one more turn than scripted — a loud test-authoring error, never a hang |

Argument values are **not** validated against `TOOL_SPECS`: a script may
deliberately omit a required field to exercise the error path.

### Example — Scenario A replay

```python
from iagent.procedural.agent import Agent
from iagent.procedural.llm_client import RecordedLLMClient

script = [
    {"tool": "get_belief",
     "arguments": {"subject": "path_ahead", "predicate": "is_clear"}},
    {"tool": "sense_lidar", "arguments": {"direction": "front"}},
    {"tool": "list_sources", "arguments": {}},
    {"tool": "downgrade_belief",
     "arguments": {"subject": "path_ahead", "predicate": "is_clear",
                   "new_confidence": 0.2,
                   "reason": "live LiDAR contradicts default_map at 12cm"}},
    {"tool": "update_belief",
     "arguments": {"subject": "path_ahead", "predicate": "is_blocked",
                   "object": "obstacle@12cm", "confidence": 0.95,
                   "source": "lidar_front", "perspective": "self",
                   "reason": "live LiDAR reading"}},
    {"tool": "finalize_answer",
     "arguments": {"answer": "No — the map claimed clear; LiDAR reads 12cm ...",
                   "cited_beliefs": ["path_ahead:is_clear",
                                     "path_ahead:is_blocked"],
                   "cited_sensors": ["lidar_front"]}},
]

agent  = Agent(knowledge, sensors, RecordedLLMClient(script), config)
result = agent.run("Is your route clear?", scenario="A", test_id="test_02_stale_map_conflict")

result.answer           # the final answer string
result.tools_called     # ['get_belief', 'sense_lidar', ...]
result.beliefs_written  # [Belief(is_clear, downgraded), Belief(is_blocked, active)]
result.trace.steps      # one Step per tool call, with its observation
```

---

## 8. `AgentResult`

| Field | Meaning |
|---|---|
| `answer` | the `finalize_answer` text (empty if the episode capped out) |
| `trace` | every step, in order |
| `beliefs_written` | beliefs actually committed, **rebuilt from the provenance rows the declarative layer returned** — not from the LLM's arguments — so `provenance_id` and `status` are the real ones. Includes downgrades, with `status="downgraded"` |
| `tools_called` | tool names in execution order, failures included |

`beliefs_written` is telemetry: an observation that doesn't parse as a
provenance row is skipped rather than raised, because broken bookkeeping
should not fail an otherwise-good episode.

---

## 9. Trace / log format

`dump_trace(trace, runs_dir, *, test_id=None, assertions=None, verdict=None, filename=None)`
writes section K's shape:

```json
{
  "test_id": "test_02_stale_map_conflict",
  "query": "Is your route clear? ...",
  "scenario": "A",
  "started_at": "2026-11-01T10:15:00.234000+00:00",
  "ended_at":   "2026-11-01T10:15:03.981000+00:00",
  "steps": [
    {"idx": 1,
     "thought": null,
     "tool_call": {"name": "get_belief", "arguments": {"subject": "path_ahead", "predicate": "is_clear"}},
     "observation": {"tool": "get_belief", "ok": true, "content": {"confidence": 1.0, "source": "default_map"}},
     "final_answer": null}
  ],
  "final_answer": "No — ...",
  "assertions": [],
  "verdict": null
}
```

- **The key set is a contract.** `assertions` and `verdict` are always present
  — empty and `null` on an ordinary run. The epistemic tests fill them in,
  either by passing `assertions=[...]` to `dump_trace` or by re-writing the
  file after asserting. Renaming a field here breaks the test logs.
- `verdict` is derived when assertions are supplied and no verdict is given:
  `"pass"` if every row's `passed` is truthy, else `"fail"`.
- Filenames: `epistemic_<test_id>.json` when a test id is in play, otherwise
  `<started_at>.json` at microsecond precision, so re-runs never collide.
  `filename=` overrides both.
- `Trace.append(step)` and `Trace.dump(runs_dir)` are the convenience forms
  named in the plan; `Trace.dump` delegates to `dump_trace`.

---

## 10. Walkthroughs

### Scenario A — groundedness (plan section I)

| # | Tool | Observation |
|---|---|---|
| 1 | `get_belief(path_ahead, is_clear)` | `confidence 1.0, source default_map` |
| 2 | `sense_lidar(front)` | `value 12, status blocked, confidence 0.95` |
| 3 | `list_sources()` | `default_map 0.6`, `lidar_front 0.9` |
| 4 | `downgrade_belief(path_ahead, is_clear, 0.2, ...)` | new provenance row, `status downgraded` |
| 5 | `update_belief(path_ahead, is_blocked, obstacle@12cm, 0.95, lidar_front, self, ...)` | new provenance row, `status active` |
| 6 | `finalize_answer(...)` | episode ends |

Conflict detection is the **LLM's** job; the layers only supply the trust
priors that make one side win. Verified end-to-end against the real
`KnowledgeAPI` and the real LiDAR in
`tests/unit/test_react_loop.py::test_scenario_a_full_pipeline_through_react_loop`.

### Scenario B — perspective (plan section J)

| # | Tool | Observation |
|---|---|---|
| 1 | `query_perspective(box_01, has_color)` | grouped views: `user → red_atom`, `third_party:bot_02 → blue_atom` |
| 2 | `sense_camera(front)` | `color_perceived brown`, `ambient.lighting yellow`, unreliability warning |
| 3 | `update_belief(box_01, has_color, brown_atom, 0.82, camera_front, **self**, ...)` | egocentric belief recorded |
| 4 | `query_provenance(box_01, has_color)` | all three rows, newest first |
| 5 | `finalize_answer(...)` | three perspectives, never merged |

Step 3 is where `perspective` being required earns its keep: filed as `user`
instead of `self`, the camera reading would overwrite what the human said.

> **Status:** Scenario B is exercised as a recorded replay only.
> `SensorimotorAPI.sense_camera` and `.sense_clock` still raise
> `NotImplementedError` (P2, Phase 3 — Scenario B), which the dispatcher
> surfaces as `{"ok": false, "error": {"kind": "not_implemented"}}`. The loop
> handles that correctly, but the scenario cannot run against the real
> sensorimotor layer until P2 lands those two methods.

---

## 11. Running it

```bash
pytest -q tests/unit/test_react_loop.py tests/unit/test_tools.py   # recorded, no API key
python scripts/verify_isolation.py                                  # layer isolation
```

Live Groq runs are opt-in only (`IAGENT_LIVE_LLM=1` plus `GROQ_API_KEY` in
`.env`, which is gitignored and never committed).
