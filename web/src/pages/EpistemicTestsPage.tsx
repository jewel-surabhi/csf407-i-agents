import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { EpistemicTraceSummary } from "../api/types";

interface Grouped {
  number: number;
  title: string;
  slug: string;
  recorded?: EpistemicTraceSummary;
  live?: EpistemicTraceSummary;
}

export default function EpistemicTestsPage() {
  const [tests, setTests] = useState<EpistemicTraceSummary[] | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [mode, setMode] = useState<"recorded" | "live">("recorded");

  useEffect(() => {
    api.epistemicTests().then(setTests).catch(() => setTests([]));
  }, []);

  if (tests === null) {
    return (
      <main className="page">
        <div className="page-inner"><div className="insp-ghost">Loading test results…</div></div>
      </main>
    );
  }

  const groups: Grouped[] = [];
  for (const t of tests) {
    let g = groups.find((x) => x.number === t.number);
    if (!g) {
      g = { number: t.number, title: t.title, slug: t.slug };
      groups.push(g);
    }
    if (t.mode === "recorded") g.recorded = t;
    else g.live = t;
  }
  groups.sort((a, b) => a.number - b.number);

  const recordedPass = groups.filter((g) => g.recorded?.verdict === "pass").length;
  const livePass = groups.filter((g) => g.live?.verdict === "pass").length;
  const liveCount = groups.filter((g) => g.live).length;

  return (
    <main className="page">
      <div className="page-inner">
        <header className="page-h">
          <h2>Epistemic Test Suite</h2>
          <p>
            Ten automated tests, each stressing a distinct epistemic property
            — grounding, perspective isolation, provenance, abstention,
            trust-prior reconciliation. Every test has a deterministic
            <b> recorded</b> run (scripted tool calls, always reproducible)
            and a <b> live</b> run against the real Groq model.
          </p>
        </header>

        <section className="et-summary">
          <div className="et-sum-item">
            <span className="et-sum-v et-sum-clear">{recordedPass}/{groups.length}</span>
            <span className="et-sum-l">Recorded passing</span>
          </div>
          <div className="et-sum-item">
            <span className={"et-sum-v " + (livePass === liveCount ? "et-sum-clear" : "et-sum-warn")}>
              {livePass}/{liveCount || groups.length}
            </span>
            <span className="et-sum-l">Live LLM passing</span>
          </div>
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
              Live
            </button>
          </div>
        </section>

        <div className="et-grid">
          {groups.map((g) => {
            const t = mode === "live" ? g.live : g.recorded;
            const isOpen = expanded === g.number;
            return (
              <article key={g.number} className={"et-card verdict-" + (t?.verdict ?? "unknown")}>
                <button className="et-card-h" onClick={() => setExpanded(isOpen ? null : g.number)}>
                  <span className="et-num">{String(g.number).padStart(2, "0")}</span>
                  <span className="et-title">{g.title}</span>
                  <span className={"et-badge badge-" + (t?.verdict ?? "unknown")}>
                    {t?.verdict ?? "n/a"}
                  </span>
                  <span className="et-caret">{isOpen ? "−" : "+"}</span>
                </button>

                {!t && (
                  <div className="et-body">
                    <div className="insp-ghost">No {mode} run recorded for this test.</div>
                  </div>
                )}

                {t && (
                  <div className="et-body">
                    <div className="et-query">"{t.query}"</div>

                    {isOpen && (
                      <>
                        <div className="et-steps">
                          {t.steps.map((s) => (
                            <span key={s.idx} className={"et-step-chip " + (s.ok === false ? "err" : "ok")}>
                              #{s.idx} {s.tool_name ?? "thought"}
                            </span>
                          ))}
                        </div>
                        <div className="et-answer">{t.final_answer || "(no final answer recorded)"}</div>
                        {t.verdict === "fail" && t.failure_mode_tested && (
                          <div className="et-failure">
                            <b>What this stresses:</b> {t.failure_mode_tested}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </main>
  );
}
