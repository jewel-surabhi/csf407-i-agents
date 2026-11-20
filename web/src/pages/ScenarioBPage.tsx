import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWarehouse } from "../context/WarehouseContext";
import type { EpistemicTraceSummary, TraceStep } from "../api/types";
import { parseThreePerspectives } from "../util";

const LIVE_DEMO_QUERY =
  "For WIDGET-42, report three perspectives separately: what the user wants, " +
  "what you (self) currently sense via the warehouse scanner, and what the " +
  "third-party ERP and sales logs say. Do not merge them.";

export default function ScenarioBPage() {
  const { busy } = useWarehouse();
  const [tests, setTests] = useState<EpistemicTraceSummary[] | null>(null);
  const [mode, setMode] = useState<"recorded" | "live">("live");

  // This page runs its own ask flow (separate from the Operations Inspector)
  // so the demo query + its live steps stay local to this page.
  const [demoSteps, setDemoSteps] = useState<TraceStep[]>([]);
  const [demoAnswer, setDemoAnswer] = useState<string | null>(null);
  const [demoBusy, setDemoBusy] = useState(false);

  useEffect(() => {
    api.epistemicTests().then(setTests).catch(() => setTests([]));
  }, []);

  const test08 = tests?.filter((t) => t.number === 8) ?? [];
  const shown = test08.find((t) => t.mode === mode) ?? test08[0];
  const parsed = shown ? parseThreePerspectives(shown.final_answer) : null;

  const runDemo = () => {
    setDemoBusy(true);
    setDemoSteps([]);
    setDemoAnswer(null);
    api.queryStream(LIVE_DEMO_QUERY, {
      onStep: (s) => setDemoSteps((prev) => [...prev, s]),
      onDone: (r) => { setDemoAnswer(r.answer); setDemoBusy(false); },
      onError: () => setDemoBusy(false),
      maxSteps: 6,
    });
  };

  return (
    <main className="page">
      <div className="page-inner">
        <header className="page-h">
          <h2>Scenario B — Tracking Perspective</h2>
          <p>
            Straight from the project proposal: a user asks the agent to
            locate a <b>red</b> box. The agent's own camera currently sees it
            as <b>brown</b>, because the simulated lighting is yellow. A
            separate maintenance bot's log claims the box was repainted{" "}
            <b>blue</b>. The agent must report all three views distinctly —
            never merging, never re-attributing one perspective's claim to
            another.
          </p>
        </header>

        <section className="sb-setup">
          <h3 className="sb-h">The test query</h3>
          <blockquote className="sb-query">
            "What color does the user think the object is, what color do you
            register it as, and what does your data history say its true
            state is?"
          </blockquote>

          <h3 className="sb-h">Target behavior</h3>
          <div className="sb-targets">
            <div className="sb-target psp-user-bg">
              <span className="psp psp-user">User</span>
              <p>Expects <b>red</b> — what they were told before the scan.</p>
            </div>
            <div className="sb-target psp-self-bg">
              <span className="psp psp-self">Self</span>
              <p>Currently senses <b>brown</b> — live camera, unreliable yellow lighting.</p>
            </div>
            <div className="sb-target psp-third-bg">
              <span className="psp psp-third">Third-party</span>
              <p>Historical log says <b>blue</b> — another agent's provenance entry.</p>
            </div>
          </div>
        </section>

        <section className="sb-sect">
          <div className="sb-sect-h">
            <h3 className="sb-h">Verified — committed test trace</h3>
            <div className="et-mode-toggle">
              <button
                className={"et-mode-btn" + (mode === "recorded" ? " active" : "")}
                onClick={() => setMode("recorded")}
              >
                Recorded
              </button>
              <button
                className={"et-mode-btn" + (mode === "live" ? " active" : "")}
                onClick={() => setMode("live")}
              >
                Live Groq
              </button>
            </div>
          </div>

          {!shown && <div className="insp-ghost">Loading test 08 trace…</div>}

          {shown && (
            <>
              <div className="sb-query-small">"{shown.query}"</div>

              {parsed ? (
                <div className="sb-targets sb-results">
                  <div className="sb-target psp-user-bg">
                    <span className="psp psp-user">User</span>
                    <p>{parsed.user}</p>
                  </div>
                  <div className="sb-target psp-self-bg">
                    <span className="psp psp-self">Self</span>
                    <p>{parsed.self}</p>
                  </div>
                  <div className="sb-target psp-third-bg">
                    <span className="psp psp-third">Third-party</span>
                    <p>{parsed.third}</p>
                  </div>
                </div>
              ) : (
                <div className="et-answer">{shown.final_answer}</div>
              )}

              <div className="et-steps" style={{ marginTop: 12 }}>
                {shown.steps.map((s) => (
                  <span key={s.idx} className={"et-step-chip " + (s.ok === false ? "err" : "ok")}>
                    #{s.idx} {s.tool_name ?? "thought"}
                  </span>
                ))}
              </div>

              <div className={"et-badge badge-" + shown.verdict} style={{ marginTop: 10, display: "inline-block" }}>
                {shown.verdict}
              </div>
              <Link to="/epistemic" className="sb-see-more">See all 10 epistemic tests →</Link>
            </>
          )}
        </section>

        <section className="sb-sect">
          <h3 className="sb-h">Try the same pattern live</h3>
          <p className="insp-note">
            The box_01 example above is captured directly from the committed
            test suite. The button below fires the SAME perspective-isolation
            reasoning live, against the warehouse's real multi-source stock
            dispute on WIDGET-42 (ERP vs. scanner vs. sales log) instead of a
            simulated box — same epistemic pattern, live data, real agent.
          </p>
          <button className="btn-primary" onClick={runDemo} disabled={demoBusy || busy}>
            {demoBusy ? "Reasoning…" : "Ask the agent now"}
          </button>

          {demoSteps.length > 0 && (
            <ol className="np-trace" style={{ marginTop: 14 }}>
              {demoSteps.map((s, i) => (
                <li key={`${s.idx}-${i}`} className="np-step">
                  <span className="np-step-i">#{s.idx}</span>
                  {s.tool_call ? (
                    <>
                      <code className="np-step-n">{s.tool_call.name}</code>
                      <span className="np-step-a">{JSON.stringify(s.tool_call.arguments).slice(0, 70)}</span>
                    </>
                  ) : (
                    <span className="np-step-a np-step-thought">{(s.thought ?? "").slice(0, 90) || "…"}</span>
                  )}
                  {s.observation && (
                    <span className={"np-step-o " + (s.observation.ok ? "ok" : "err")}>
                      {s.observation.ok ? "✓" : "✗"}
                    </span>
                  )}
                </li>
              ))}
            </ol>
          )}

          {demoAnswer && <div className="et-answer" style={{ marginTop: 12 }}>{demoAnswer}</div>}
        </section>
      </div>
    </main>
  );
}
