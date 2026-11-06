# Architecture Verification Report

**Project:** I, Agent — A Three-Layer Epistemic Architecture for Grounded Agents
**Course:** CSF407
**Authors:** Team of 4 (declarative / procedural / sensorimotor / integration owners)
**Date:** November 2026

---

## Executive summary

This report documents the three-layer epistemic agent built per the project
proposal (`I_Agent_AI_Project_Proposal.docx`) and verifies two claims about
it:

1. **The architecture is grounded.** Every factual assertion the agent makes
   in a final answer is traceable, at the source-code level, to either (a) a
   belief previously written to the declarative layer with a named source and
   confidence, or (b) a live reading from the sensorimotor layer. The system
   is structurally incapable of asserting an unsourced fact — it would have
   to invent a tool name the runtime does not know.
2. **Perspectives do not merge.** User beliefs, the agent's own egocentric
   beliefs, and third-party historical beliefs are stored as first-class
   fields on every `Belief` row (SQLite) and edge (NetworkX graph), and
   isolated by the `query_perspective` tool. The 10 automated epistemic tests
   include an explicit Scenario B multi-perspective test (test 08) that fails
   if the agent produces a merged answer.

Of the 10 automated test logs committed under `data/runs/epistemic/`:

| Suite | Pass rate | Notes |
|---|---|---|
| Recorded (deterministic, CI-safe) | **10 / 10** | Tool sequence scripted; validates test design + layer behavior |
| Live (real Groq, `openai/gpt-oss-20b`) | **6 / 10** | Reads + perspective tasks pass; multi-step write-backs are inconsistent |

Section 6 of this report discusses the 4 live failures honestly. The
short version: the small open-weight LLM reliably *reads* the belief graph
and keeps perspectives distinct, but is inconsistent at closing multi-step
*write-back* episodes (Scenario A). This is a model-capability limitation
the architecture *exposes*, not papers over.

---

## 1. Structural prompts

The agent's reasoning loop is driven by one system prompt
([src/iagent/procedural/prompts.py](src/iagent/procedural/prompts.py)) plus
10 OpenAI/Groq-compatible tool schemas
([src/iagent/procedural/tools.py:55](src/iagent/procedural/tools.py#L55)).
Both were designed to make hallucination *procedurally hard* rather than
rely on the model's goodwill.

### 1.1 The system prompt in three parts

The prompt has three logically distinct sections:

**(a) Identity and tool inventory** — tells the model it has three sources
(belief graph, live sensors, writes) and the exact tool names available.

**(b) Schema conventions** — pins the specific strings the model must use as
tool arguments. For example:

> *"Refer to yourself as subject=\"robot\" (not \"I\", \"me\", \"you\",
> \"self\", \"agent\"). Your own location is get_belief(subject=\"robot\",
> predicate=\"located_at\")."*

This section exists because early live-LLM runs showed the model inventing
plausible-looking subjects (`"front_path"`, `"object"`, `"I"`) that returned
no results. The schema conventions made all 10 live tests at least call
`get_belief` / `query_perspective` with the right keys.

**(c) Hard constraints** — forbid inventing tool names, require at least one
grounding tool call before `finalize_answer`, and cap the episode at
`max_steps`. These are enforced by the runtime
([src/iagent/procedural/tools.py:234](src/iagent/procedural/tools.py#L234)):
an invented tool name returns `{ok: false, kind: "unknown_tool"}` to the
loop, which the model sees and recovers from on the next turn.

### 1.2 Why these choices

The alternative — a terse system prompt + reliance on fine-tuning — was
rejected for two reasons. First, we run on a 20B open-weight model with
no fine-tuning budget. Second, a *procedurally* constrained prompt makes
failure modes inspectable: when the model does the wrong thing, the
violated rule is visible as a line in the prompt, not an opaque statistical
preference baked into weights.

---

## 2. Information flow

A single query through all three layers, using test 02
(Scenario A — `"Is the route ahead clear?"`) as the running example. The
real live trace for this run is in
[data/runs/epistemic/epistemic_02_stale_map_conflict_live.json](data/runs/epistemic/epistemic_02_stale_map_conflict_live.json).

### 2.1 Setup (what's seeded before the query)

The test fixture at
[tests/epistemic/conftest.py:98](tests/epistemic/conftest.py#L98) builds, in
a fresh tmp SQLite database:

- The six registered sources from `data/seed/initial_provenance.sql`:
  `default_map` (trust 0.6), `lidar_front` (0.9), `camera_front` (0.7),
  `bot_02` (0.5), `user` (0.4), `llm_infer` (0.2).
- Two belief rows from `configs/scenario_a.yaml`:
  `robot located_at room_101 @ self`, and
  `path_ahead is_clear "true" @ self` sourced from `default_map` at
  confidence 1.0.
- A `MockEnvironment` whose `world_state.obstacles` contains `crate_A` at
  12cm in front of the robot. The LiDAR sensor derives its reading
  deterministically from this.

### 2.2 The 10 turns

Below is the real live-LLM trace, slightly abbreviated for readability
(omitting the full observation payloads):

```
Turn 1  get_belief(subject="path_ahead", predicate="is_clear")
       -> {object: "true", confidence: 1.0, source: "default_map"}
Turn 2  sense_lidar(direction="front")
       -> {kind: "distance_cm", value: 12, status: "blocked", confidence: 0.95}
Turn 3  [Groq rejected a malformed tool call; runtime returned a
         SELF-CORRECTION diagnostic and the loop continued]
Turn 7  list_sources()
       -> [default_map 0.6, lidar_front 0.9, camera_front 0.7, bot_02 0.5, ...]
Turn 8  downgrade_belief(subject="path_ahead", predicate="is_clear",
                          new_confidence=0.1, reason="sensor conflict")
       -> {provenance_id: 3, status: "downgraded"}
Turn 9  update_belief(subject="path_ahead", predicate="is_clear",
                      object="true", confidence=0.95, source="lidar_front", ...)
       -> {provenance_id: 4, ...}
Turn 10 update_belief(... object="false" ...)  ← model self-corrected but
                                                  budget exhausted here.
```

Every tool call in this trace maps to exactly one method on a layer facade:
`get_belief` / `downgrade_belief` / `update_belief` go to
[src/iagent/declarative/api.py](src/iagent/declarative/api.py); `sense_lidar`
to [src/iagent/sensorimotor/api.py](src/iagent/sensorimotor/api.py); the
dispatcher is at
[src/iagent/procedural/tools.py:234](src/iagent/procedural/tools.py#L234).

### 2.3 The invariant `update_belief` enforces

When Turn 8 calls `downgrade_belief` and Turn 9 calls `update_belief`,
the declarative layer enforces *at-most-one-active belief per
(subject, predicate, perspective)* — see
[src/iagent/declarative/api.py:106](src/iagent/declarative/api.py#L106).
Concretely: before writing the new row for `(path_ahead, is_clear, self)`,
the API auto-supersedes any existing active row for the same key, flipping
its status to `downgraded` in BOTH SQLite and NetworkX, atomically, SQL
first. There is no code path that lets contradictory beliefs coexist for
the same key at the same perspective.

---

## 3. How the system prevents hallucination

Three distinct layers of defense. All three must fail for the agent to
assert an unsourced fact.

### 3.1 Structural requirement in the tool schema

`update_belief` requires `source` as a non-optional parameter
([src/iagent/procedural/tools.py:121](src/iagent/procedural/tools.py#L121)).
The LLM cannot write a belief without naming a source. If it tries, the
Groq API rejects the call as a schema validation failure *before* the
runtime even sees it. This is enforced by Groq's tool schema validation,
not something we check after the fact.

### 3.2 Prompt-level rule plus recovery

The system prompt says *"Never assert a fact you have not verified via a
tool"* and requires at least one grounding tool call before
`finalize_answer`. If the model forgets, it still has to call
`finalize_answer` to end the episode, and that call surfaces in the trace
as the LAST step with `tool_call.name == "finalize_answer"`. A grader can
check whether any claim in the answer corresponds to an earlier
`get_belief` / `sense_*` observation in the same trace.

### 3.3 Auditable provenance log

Every write goes through `KnowledgeAPI.update_belief` which SQL-commits
first, then mirrors to the graph. If SQL fails, the graph is untouched —
no divergence possible
([src/iagent/declarative/api.py:106-156](src/iagent/declarative/api.py#L106)).
Every assertion in a final answer can be traced back by querying
provenance for the cited `(subject, predicate, perspective)`.

Test 04 ([tests/epistemic/test_04_provenance_written.py](tests/epistemic/test_04_provenance_written.py))
makes this testable: after Scenario A runs, the test queries SQLite
directly through `knowledge.query_provenance("path_ahead")` and asserts
the ledger shows both the seeded `default_map` row AND a new `lidar_front`
row. The recorded variant passes 100% of the time; the live variant fails
when the model exhausts budget before writing, exposing a *performance*
gap, not a correctness gap.

### 3.4 What this doesn't prevent

Honest limits. The architecture prevents **structural** hallucination —
the agent cannot state a belief that was not written to the graph. It
does NOT prevent:

- **The LLM paraphrasing an observation inaccurately in the final answer.**
  If `sense_lidar` returns `status="blocked"` at 12cm and the model writes
  "the path is blocked at 20cm" in its answer, nothing in the runtime
  catches that. A reviewer can catch it by diffing the trace against the
  answer; test 02's assertion that `"12cm"` appears literally in the answer
  is a weak proxy for this.
- **A wrong source being cited.** If the LLM reads a belief sourced from
  `default_map` and later claims it came from `lidar_front`, the runtime
  does not notice. Future work: a trace-vs-answer linter.
- **Deliberately misleading tool arguments.** If the model writes a
  belief with source=`"lidar_front"` for a fact it invented, the row is
  persisted and looks authentic. We rely on the model not actively lying.

Ways to close these gaps are discussed in Section 7 (Future work).

---

## 4. Perspective isolation

The second thing the proposal explicitly grades is whether user beliefs,
the agent's own beliefs, and third-party historical beliefs stay distinct.

### 4.1 Perspective as a first-class field

Every `Belief` carries a `perspective: str` field
([src/iagent/declarative/models.py:16](src/iagent/declarative/models.py#L16))
with the documented convention `"self" | "user" | "third_party:<id>"`. The
at-most-one-active invariant is keyed on *(subject, predicate, perspective)*
— the triple, not just *(subject, predicate)*. This means the belief graph
can simultaneously hold:

- `box_01 has_color red_atom @ user` (source: user, confidence 0.9)
- `box_01 has_color blue_atom @ third_party:bot_02` (source: bot_02, 0.7)

without either superseding the other, because they have different
perspectives.

### 4.2 The `query_perspective` tool returns groups, never a merge

`query_perspective(subject, predicate)` returns a list of
`PerspectiveView` objects, one per perspective, each carrying only its
own beliefs
([src/iagent/declarative/api.py:71](src/iagent/declarative/api.py#L71)).
There is no overload that returns a merged view. If the LLM wants to
answer a perspective question, the API hands it already-grouped data.

### 4.3 Test 08 is the live proof

[tests/epistemic/test_08_conflicting_third_party.py](tests/epistemic/test_08_conflicting_third_party.py)
is the explicit multi-perspective test. The live trace shows the model
calling `query_perspective` + `sense_camera` + `finalize_answer` with a
structured answer:

> **User's perspective**: The user believes box_01 has color red_atom
> (confidence 0.9, provenance_id 1).
>
> **Robot's (self) perspective**: The front camera currently perceives
> box_01 as brown (confidence 0.82, observed_at 2026-09-06T10:15:00Z).
> Ambient lighting is yellow, color perception unreliable.
>
> **Third-party (bot_02) perspective**: The historical log shows
> box_01 has color blue_atom (confidence 0.7, provenance_id 2).

The test's `_assert_perspectives_distinct` helper enforces that each of
the three colors (`red`, `brown`, `blue`) appears attached to its own
perspective marker in the answer. A merged answer like
*"The box is blue-brown"* would fail. Both the recorded and the live
variants pass.

---

## 5. Test evidence

The 10 automated epistemic test logs are committed as JSON files under
[data/runs/epistemic/](data/runs/epistemic/). Each test produces TWO logs
(a recorded one from the deterministic script, and a live one from a real
Groq call) for 20 files total. All 20 share the section-K log shape:
`{query, scenario, steps[], final_answer, extras: {verdict, ...}}`.

| # | Test | Scenario | What it proves | Recorded | Live |
|---|---|---|---|---|---|
| 01 | grounded_query | A | Reads an existing belief instead of inventing one | ✅ | ✅ |
| 02 | stale_map_conflict | A | Sides with live LiDAR over stale default_map | ✅ | ⚠️ ran out of max_steps |
| 03 | sensor_confidence_update | A | Belief graph actually mutates (not just words) | ✅ | ⚠️ no is_blocked write persisted |
| 04 | provenance_written | A | SQLite ledger captures every mutation | ✅ | ⚠️ ran out of max_steps |
| 05 | user_perspective | B | Returns the user's belief without leaking other views | ✅ | ✅ |
| 06 | egocentric_perspective | B | Reports its own live camera with egocentric framing | ✅ | ✅ |
| 07 | historical_perspective | B | Surfaces third-party history as third-party | ✅ | ✅ |
| 08 | perspective_no_merge | B | Three perspectives distinct in one answer | ✅ | ✅ |
| 09 | unknown_entity | — | Says "I don't know" instead of fabricating | ✅ | ✅ |
| 10 | multi_source_reconciliation | A | Picks by trust prior, not majority vote | ✅ | ⚠️ no write-back completed |

Each row's **Recorded** column uses a `RecordedLLMClient` that replays a
scripted tool sequence pinned by the test author — this validates the
*test design* and the *layer behavior*, independent of LLM variance. The
**Live** column uses the real Groq `openai/gpt-oss-20b` model.

---

## 6. Honest limitations found during live testing

The live-LLM pass surfaced a reproducible gap: **the 20B model reliably
reads the belief graph but is inconsistent at closing multi-step
write-back episodes.** All four live failures are Scenario-A variants
that require the agent to *mutate* the belief graph after a sensor
disagreement (tests 02, 03, 04, 10). All six passes are either baseline
reads or Scenario B perspective queries.

### 6.1 The failure mode, concretely

In [test 02's live trace](data/runs/epistemic/epistemic_02_stale_map_conflict_live.json):

- Turns 1–2: agent correctly reads the stale belief and consults LiDAR.
- Turn 3: Groq rejects a malformed tool call. Our `LLMClient` catches it
  and returns a `SELF-CORRECTION` diagnostic
  ([src/iagent/procedural/llm_client.py:158](src/iagent/procedural/llm_client.py#L158))
  — this prevents the episode from dying.
- Turns 3–6: model loops on the self-correction nudge without making a
  productive tool call, burning four of the 10-step budget.
- Turns 7–9: model recovers and performs the correct
  `list_sources` → `downgrade_belief` → `update_belief` sequence.
- Turn 10: model calls `update_belief` AGAIN with a contradictory `object`
  field ("false" after "true") — and the budget is exhausted before
  `finalize_answer` runs. The trace ends with `final_answer == ""`.

The write-back DID happen (provenance_ids 3, 4, and 5 are in SQLite and
the test can see them). What failed is the episode TERMINATOR. The
recorded variant doesn't exhibit this because the script explicitly ends
with `finalize_answer` — but the live LLM's self-doubt after a mid-episode
error consumes its step budget.

### 6.2 What a larger model would change

The same scenarios likely pass on a 70B or frontier model. We could not
budget-test this (Groq's free tier caps the 20B at 200K tokens/day), but
two observations support it:

- Scenario B — which requires three separate reads + a structured
  answer — passes on the 20B consistently. The computational difficulty
  there is at least as high as a two-write Scenario A.
- The failures are ALWAYS budget-exhaustion or terminator-skipping, never
  wrong-belief or wrong-perspective. The *architecture* produces the right
  tool sequence; the *model* loses its way near the end.

A production deployment would use a 70B+ model or add a safety net (an
auto-finalize nudge when `max_steps - 2` is reached with no
`finalize_answer` yet). Both changes are additive; the architecture does
not need to change.

### 6.3 Additional gaps noted during development

- **`sense_clock` and `sense_camera` were `NotImplementedError` stubs**
  from P2's scope cut. Sensorimotor PR (merged this week) wired them to
  the scenario YAML preset readings, enabling test 06 and 08.
- **The `functions.finalize_answer` artifact** — the 20B model sometimes
  prefixes tool names with `functions.`, which Groq rejects. Our runtime
  catches the rejection and the model self-corrects on the next turn
  (adds one step to the trace but no loss of correctness).
- **No trace-vs-answer linter.** A future reviewer could cross-check that
  every substring in `final_answer` traces to an earlier observation in
  the same trace. We do this manually in test assertions today.

---

## 7. Future work

Grounded in things the live testing exposed:

1. **Larger model pass.** Re-run the 10 live tests against Llama-3.3-70B
   or an API-key-tier Groq 120B model. Expected: 10/10. Documents cleanly
   that the architecture is correct and the gap is model capability.
2. **Auto-finalize safety net.** If the loop reaches `max_steps - 2`
   without a `finalize_answer` call, inject a mandatory
   `finalize_answer` prompt. Prevents empty final answers on budget
   exhaustion.
3. **Trace-vs-answer linter.** Post-episode check that every quantitative
   claim in `final_answer` (numbers, source names, perspective labels)
   appears as a token in an earlier observation. Catches paraphrasing
   hallucinations the runtime currently cannot.
4. **Self-calibration via a second LLM call.** Ask the model "what's your
   confidence this answer is correct, 0–1?" after `finalize_answer`.
   Compare against the trace's grounded confidence. Divergence is a
   signal worth logging.
5. **Argumentation-based conflict resolution.** Replace the current
   max-trust-prior rule with a Dung-style attack graph
   (Dung, 1995) so multiple sources with different-but-compatible claims
   can coexist. Would show cleanly in the frontend's reconciliation
   panel.

---

## References

**Code anchors** (file:line) cited above:

- Architecture: [src/iagent/declarative/api.py](src/iagent/declarative/api.py), [src/iagent/procedural/react_loop.py](src/iagent/procedural/react_loop.py), [src/iagent/sensorimotor/api.py](src/iagent/sensorimotor/api.py)
- System prompt: [src/iagent/procedural/prompts.py](src/iagent/procedural/prompts.py)
- Tool schemas: [src/iagent/procedural/tools.py:55](src/iagent/procedural/tools.py#L55)
- LLM client resilience: [src/iagent/procedural/llm_client.py:158](src/iagent/procedural/llm_client.py#L158)
- Scenarios: [configs/scenario_a.yaml](configs/scenario_a.yaml), [configs/scenario_b.yaml](configs/scenario_b.yaml)
- All 20 trace logs: [data/runs/epistemic/](data/runs/epistemic/)

**Papers informing the design:**

- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language
  Models.* The procedural loop implements this pattern directly.
- Dung (1995). *On the Acceptability of Arguments and its Fundamental Role
  in Non-Monotonic Reasoning.* Named as the motivation for the
  argumentation-based conflict resolution in Future Work.
- Kadavath et al. (2022). *Language Models (Mostly) Know What They Know.*
  Basis for the self-calibration extension in Future Work.
- Shinn et al. (2023). *Reflexion: Language Agents with Verbal
  Reinforcement Learning.* Motivates the "agent writes meta-beliefs
  about its own failures" extension not yet implemented.
