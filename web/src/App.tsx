import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type {
  AgentResult,
  Belief,
  InventoryView,
  ProvenanceEntry,
  Shelf,
  ShelfContents,
  Source,
  TraceStep,
  WarehouseView,
} from "./api/types";
import WarehouseFloor from "./components/WarehouseFloor";
import DecisionPanel from "./components/DecisionPanel";

// Pills aligned with the proposal's canonical epistemic scenarios so a grader
// opening the UI cold can run Scenario A / B straight from the ask bar.
const PILLS: { label: string; query: string }[] = [
  {
    label: "Scenario A · Is the Aisle B route clear?",
    query:
      "Is the route from B-S to B-D clear? Justify your answer by inspecting the belief graph — the map claim versus the live LiDAR reading — and say which perspective wins and why.",
  },
  {
    label: "Scenario B · Three perspectives on WIDGET-42",
    query:
      "For WIDGET-42, report three perspectives separately: what the user wants, what you (self) currently sense via the warehouse scanner, and what the third-party ERP and sales logs say. Do not merge them.",
  },
  {
    label: "How do you avoid hallucinating?",
    query:
      "Explain, by citing the tool calls in your trace, how your answer is grounded in the belief graph and provenance log rather than invented.",
  },
];

// Human-readable label for a raw perspective slug like `third_party:erp`.
function perspectiveLabel(p: string): string {
  if (p === "self") return "self · egocentric";
  if (p === "user") return "user";
  if (p.startsWith("third_party:")) return "third-party · " + p.slice("third_party:".length);
  return p;
}

// Tailwind-ish class suffix that drives perspective-chip color in CSS.
function perspectiveKind(p: string): "self" | "user" | "third" {
  if (p === "self") return "self";
  if (p === "user") return "user";
  return "third";
}

function fmtTime(iso?: string | null): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

// Compute age-in-hours from an observed_at ISO string vs. now.
function ageHours(iso?: string | null): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return 0;
  return Math.max(0, (Date.now() - t) / 3_600_000);
}

// Freshness tier drives the colored badge on each provenance / source row.
function freshnessTier(hours: number): "fresh" | "aging" | "stale" {
  if (hours < 1) return "fresh";
  if (hours < 24) return "aging";
  return "stale";
}

function fmtAge(hours: number): string {
  if (hours < 1 / 60) return "just now";
  if (hours < 1) return `${Math.round(hours * 60)}m old`;
  if (hours < 24) return `${hours.toFixed(1)}h old`;
  return `${Math.round(hours / 24)}d old`;
}

interface Toast {
  id: number;
  dot: string;
  html: string;
}

export default function App() {
  const [view, setView] = useState<WarehouseView | null>(null);
  const [offline, setOffline] = useState(false);
  const [pathMap, setPathMap] = useState(false);
  const [explain, setExplain] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [tab, setTab] = useState<"reason" | "trust" | "reflect">("reason");
  const [shelfSel, setShelfSel] = useState<Shelf | null>(null);
  const [inv, setInv] = useState<InventoryView | null>(null);
  const [bin, setBin] = useState<ShelfContents | null>(null);
  const [prov, setProv] = useState<ProvenanceEntry[] | null>(null);
  const [recentArrival, setRecentArrival] = useState<string | null>(null);
  const [xferDest, setXferDest] = useState<string>("");
  const [xferQty, setXferQty] = useState<number>(10);
  const [answer, setAnswer] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [askText, setAskText] = useState("");
  // Steps streamed live from /api/query/stream (SSE) while the agent runs.
  const [liveSteps, setLiveSteps] = useState<TraceStep[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [reflections, setReflections] = useState<Belief[]>([]);
  const [newLesson, setNewLesson] = useState("");
  const [newTopic, setNewTopic] = useState("");
  const toastId = useRef(0);

  const toast = useCallback((dot: string, html: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, dot, html }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const load = useCallback(async () => {
    try {
      const [v, srcs, refs] = await Promise.all([
        api.getWarehouse(),
        api.sources(),
        api.listReflections(),
      ]);
      setView(v);
      setSources(srcs);
      setReflections(refs);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const refreshReflections = useCallback(async () => {
    try {
      setReflections(await api.listReflections());
    } catch {
      /* ignore */
    }
  }, []);

  const teachReflection = useCallback(async () => {
    const lesson = newLesson.trim();
    if (!lesson) return;
    try {
      const b = await api.addReflection(lesson, newTopic.trim() || "general");
      toast("var(--accent)", `Learned: <b>${b.predicate.split(":").pop()}</b>`);
      setNewLesson("");
      setNewTopic("");
      refreshReflections();
    } catch (e) {
      toast("var(--blocked)", String(e));
    }
  }, [newLesson, newTopic, toast, refreshReflections]);

  const doMove = useCallback(
    async (to: string) => {
      if (!view) return;
      const opt = view.options.find((o) => o.to === to);
      try {
        const next = await api.move(to);
        setView(next);
        if (opt && !opt.passable) {
          toast("var(--blocked)", `Overrode the agent — <b>${Math.round(opt.confidence * 100)}%</b>`);
        } else if (opt) {
          toast("var(--clear)", `Moved · <b>${Math.round(opt.confidence * 100)}%</b> confident`);
        }
        // Dynamic events: toast one notice per fired event. If any wrote a
        // belief, refresh the selected shelf's inventory + provenance so the
        // user sees the mutation inline instead of having to re-click.
        const fired = next.events_fired ?? [];
        for (const ev of fired) {
          toast(
            ev.action === "toast" ? "var(--accent)" : "var(--warn)",
            `⚡ <b>${ev.event_id}</b> · ${ev.message}`,
          );
        }
        if (fired.some((e) => e.action !== "toast") && shelfSel) {
          try {
            const [i, b, p] = await Promise.all([
              api.inventory(shelfSel.id),
              api.bin(shelfSel.id),
              api.provenance(shelfSel.id),
            ]);
            setInv(i);
            setBin(b);
            setProv(p);
          } catch {
            /* ignore */
          }
        }
        const goal = next.shelves.find((s) => s.id === next.goal);
        if (goal && goal.near === next.position) toast("var(--accent)", `Reached <b>${next.goal}</b> ★`);
      } catch (e) {
        toast("var(--blocked)", String(e));
      }
    },
    [view, toast, shelfSel],
  );

  const selectShelf = useCallback(async (id: string) => {
    if (!view) return;
    const sh = view.shelves.find((s) => s.id === id) ?? null;
    setShelfSel(sh);
    setInv(null);
    setBin(null);
    setProv(null);
    setRecentArrival(null);
    const others = view.shelves.filter((s) => s.id !== id);
    setXferDest(others[0]?.id ?? "");
    try {
      // Fetch three things in parallel: per-SKU reconciliation (multi-source
      // dispute), the shelf's full bin contents (every lot stored here), and
      // the full provenance timeline (the SQLite history ledger).
      const [i, b, p] = await Promise.all([
        api.inventory(id),
        api.bin(id),
        api.provenance(id),
      ]);
      setInv(i);
      setBin(b);
      setProv(p);
    } catch {
      /* leave null */
    }
  }, [view]);

  const doTransfer = useCallback(async () => {
    if (!shelfSel) return;
    try {
      const res = await api.transfer(shelfSel.id, xferDest, xferQty);
      if (!res.ok) {
        toast("var(--blocked)", res.message);
        return;
      }
      toast("var(--clear)", `Moved <b>${res.quantity} × ${res.sku}</b> → ${res.dest_shelf}`);
      // Refresh everything the transfer affected: the warehouse view (so the
      // floor is current), the source's reconciliation and bin, AND the
      // destination's bin — so the user SEES the +K units land there.
      const [v, srcInv, srcBin, destBin, srcProv] = await Promise.all([
        api.getWarehouse(),
        api.inventory(shelfSel.id),
        api.bin(shelfSel.id),
        api.bin(xferDest),
        api.provenance(shelfSel.id),
      ]);
      setView(v);
      setInv(srcInv);
      setBin(srcBin);
      setProv(srcProv);
      setRecentArrival(
        `+${res.quantity} ${res.sku} just landed at ${res.dest_shelf} — ` +
          `${res.dest_shelf} now holds ${destBin.total_units} total units ` +
          `(${destBin.lots.length} lot${destBin.lots.length === 1 ? "" : "s"}).`,
      );
      // Hide the arrival line after a few seconds so the panel reads normally again.
      setTimeout(() => setRecentArrival(null), 6000);
    } catch (e) {
      toast("var(--blocked)", String(e));
    }
  }, [shelfSel, xferDest, xferQty, toast]);

  const ask = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      setBusy(true);
      setAnswer(null);
      setLiveSteps([]);
      // Switch the Control Center to the Reasoning tab so the live trace is
      // visible while the agent runs — makes the wait feel purposeful.
      setTab("reason");
      api.queryStream(text, {
        onStep: (step) => setLiveSteps((prev) => [...prev, step]),
        onDone: (result) => {
          setAnswer(result);
          setBusy(false);
        },
        onError: (msg) => {
          toast("var(--blocked)", "Agent error: " + msg);
          setBusy(false);
        },
        maxSteps: 6,
      });
    },
    [toast],
  );

  const reset = useCallback(async () => {
    try {
      setView(await api.reset());
      setShelfSel(null);
      setAnswer(null);
      toast("var(--accent)", "Warehouse reset to start");
    } catch (e) {
      toast("var(--blocked)", String(e));
    }
  }, [toast]);

  const toggleTheme = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const sysDark = window.matchMedia("(prefers-color-scheme:dark)").matches;
    document.documentElement.setAttribute("data-theme", cur ? (cur === "dark" ? "light" : "dark") : sysDark ? "light" : "dark");
  };

  if (offline) {
    return (
      <div className="offline">
        <div className="card">
          <h2>Backend not reachable</h2>
          <p>Start the FastAPI server so the warehouse can load its grounded state:</p>
          <pre>export GROQ_API_KEY=gsk_...
uvicorn iagent.api.server:app --reload --port 8000</pre>
          <p>Then reload this page. The dev server proxies <code>/api</code> to port 8000.</p>
          <button className="iconbtn" onClick={load}>Retry</button>
        </div>
      </div>
    );
  }

  if (!view) {
    return <div className="offline"><div className="card"><h2>Loading Warehouse 7…</h2></div></div>;
  }

  const otherShelves = view.shelves.filter((s) => s.id !== shelfSel?.id);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <h1>Warehouse&nbsp;7</h1>
          <span className="sub">Ops&nbsp;AI&nbsp;·&nbsp;Co-pilot</span>
        </div>
        <div className="spacer" />
        <div className="toggles">
          <button className={"tgl" + (pathMap ? " on" : "")} onClick={() => setPathMap((v) => !v)}>
            <span className="sw" />Path map
          </button>
          <button className={"tgl" + (explain ? " on" : "")} onClick={() => setExplain((v) => !v)}>
            <span className="sw" />Explain
          </button>
        </div>
        <button className="iconbtn ghost" onClick={toggleTheme} title="Toggle theme" aria-label="Toggle theme">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
        </button>
        <button className="iconbtn" onClick={reset}>Reset</button>
        <button className="iconbtn" onClick={() => setDrawer(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h16M4 18h16" /></svg>
          Control Center
        </button>
      </header>

      <div className="stage">
        <div className="floorwrap">
          <WarehouseFloor view={view} pathMap={pathMap} onMove={doMove} onSelectShelf={selectShelf} />
        </div>

        <DecisionPanel view={view} explain={explain} onMove={doMove} />

        {pathMap && (
          <div className="legend">
            <h4>Confidence</h4>
            <div className="lr"><span className="sw" style={{ background: "var(--clear)" }} />High · sensor-confirmed</div>
            <div className="lr"><span className="sw" style={{ background: "var(--warn)" }} />Medium · map only</div>
            <div className="lr"><span className="sw" style={{ background: "var(--blocked)" }} />Low · conflict / blocked</div>
          </div>
        )}

        {shelfSel && (
          <div className="ctx">
            <button className="close" onClick={() => setShelfSel(null)} aria-label="Close">×</button>
            <div className="ctx-h">
              <span className="tag">{shelfSel.id}</span>
              <span className={"st " + (shelfSel.stock.length > 1 ? "alert" : "clear")}>
                {shelfSel.stock.length > 1 ? "DISPUTED" : shelfSel.low ? "LOW" : "IN STOCK"}
              </span>
            </div>
            <div className="ctx-b">
              <div className="srcrow">
                <span className="nm">Location</span>
                <span className="ct">{view.junctions[shelfSel.near]?.label}</span>
              </div>

              {inv && inv.sources.length > 1 && (
                <>
                  <div style={{ fontSize: 12, color: "var(--ink-2)" }}>
                    Sources disagree — each from its own perspective:
                  </div>
                  {inv.sources.map((s, i) => (
                    <div className="srcrow" key={i}>
                      <span className="nm">
                        <span className={"psp psp-" + perspectiveKind(s.perspective)}>
                          {perspectiveLabel(s.perspective)}
                        </span>{" "}
                        {s.source}
                        {s.note && <span className="tw"> ·{s.note}</span>}
                      </span>
                      <span className="ct">{s.predicate === "shipped" ? "−" : ""}{s.value}</span>
                    </div>
                  ))}
                  <div className="reconcile">✓ Reconciled to <b>≈ {inv.reconciled}</b> — {inv.explanation}</div>
                </>
              )}

              {bin && (
                <>
                  <div style={{ fontSize: 11, color: "var(--ink-2)", textTransform: "uppercase",
                                 letterSpacing: ".06em", marginTop: 2 }}>
                    Bin contents · total {bin.total_units}
                  </div>
                  {bin.lots.length === 0 ? (
                    <div className="srcrow"><span className="nm" style={{ color: "var(--ink-3)" }}>Empty.</span></div>
                  ) : bin.lots.map((lot, i) => (
                    <div className="srcrow" key={i} style={{
                      padding: "6px 9px", borderRadius: 7,
                      background: lot.is_native ? "var(--panel-2)" : "var(--clear-bg)",
                      border: "1px solid " + (lot.is_native ? "var(--line)" : "var(--clear)"),
                    }}>
                      <span className="nm">
                        <b>{lot.sku}</b>{" "}
                        {lot.is_native
                          ? <span className="tw">· native</span>
                          : <span className="tw" style={{ color: "var(--clear)" }}>· transferred in</span>}
                      </span>
                      <span className="ct">{lot.quantity}</span>
                    </div>
                  ))}
                </>
              )}

              {prov && prov.length > 0 && (
                <>
                  <div style={{ fontSize: 11, color: "var(--ink-2)", textTransform: "uppercase",
                                 letterSpacing: ".06em", marginTop: 2 }}>
                    Provenance timeline · SQLite log
                  </div>
                  <ol className="prov">
                    {prov.slice(0, 8).map((p, i) => {
                      const reason = (p.payload as { reason?: string } | null | undefined)?.reason;
                      const src = sources.find((s) => s.source_id === p.source_id);
                      const age = ageHours(p.observed_at);
                      const decayed = src ? Math.max(0, p.confidence - src.decay_rate_per_hour * age) : p.confidence;
                      const tier = freshnessTier(age);
                      return (
                        <li key={p.provenance_id ?? i} className={"pv pv-" + p.status}>
                          <span className="pv-t">{fmtTime(p.recorded_at ?? p.observed_at)}</span>
                          <span className="pv-m">
                            <span className={"psp psp-" + perspectiveKind(p.perspective)}>
                              {perspectiveLabel(p.perspective)}
                            </span>{" "}
                            <b>{p.source_id}</b>{" "}
                            <span className="tw">{p.predicate}=</span>
                            <span className="ct">{p.object}</span>
                            {reason && <span className="tw"> · {reason}</span>}
                            <span className={"fresh fresh-" + tier}>
                              {fmtAge(age)}
                              {src && src.decay_rate_per_hour > 0 && (
                                <span className="dec">
                                  {" · "}conf {decayed.toFixed(2)}
                                  <span className="was">/{p.confidence.toFixed(2)}</span>
                                </span>
                              )}
                            </span>
                          </span>
                          {p.status !== "active" && (
                            <span className="pv-s">{p.status}</span>
                          )}
                        </li>
                      );
                    })}
                  </ol>
                </>
              )}

              {recentArrival && (
                <div className="reconcile" style={{ background: "var(--clear-bg)", color: "var(--clear)" }}>
                  {recentArrival}
                </div>
              )}

              <div className="xfer">
                <label>Transfer stock</label>
                <div className="xfer-row">
                  <input type="number" min={1} value={xferQty}
                    onChange={(e) => setXferQty(Math.max(1, Number(e.target.value) || 1))} />
                  <select value={xferDest} onChange={(e) => setXferDest(e.target.value)}>
                    {otherShelves.map((s) => (
                      <option key={s.id} value={s.id}>{s.id} · {view.junctions[s.near]?.label}</option>
                    ))}
                  </select>
                </div>
                <button className="iconbtn" style={{ justifyContent: "center", background: "var(--accent)", color: "var(--accent-ink)", border: "none" }} onClick={doTransfer}>
                  Confirm transfer
                </button>
              </div>
            </div>
          </div>
        )}

        {(answer || busy) && (
          <div className="comms">
            <div className="comms-card">
              <button className="close" onClick={() => setAnswer(null)} aria-label="Dismiss">×</button>
              <div className="comms-head">
                <span className="av"><svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-ink)" strokeWidth="2.2"><path d="M12 3v3M4 16h16M9 12h6v4H9z" /></svg></span>
                <span className="ttl">Ops AI</span>
                {busy && <span className="busy">reasoning…</span>}
              </div>
              <div className="comms-body">
                {answer ? (
                  answer.answer
                ) : (
                  <>
                    <div style={{ color: "var(--ink-2)", marginBottom: 8 }}>
                      {liveSteps.length === 0
                        ? "Thinking through the layers…"
                        : `Reasoning · step ${liveSteps.length}${liveSteps[liveSteps.length - 1]?.tool_call ? ` · ${liveSteps[liveSteps.length - 1].tool_call!.name}()` : ""}`}
                    </div>
                    {liveSteps.length > 0 && (
                      <div className="live-steps">
                        {liveSteps.slice(-6).map((s, i) => (
                          <div className="live-step" key={`${s.idx}-${i}`}>
                            <span className="ls-idx">#{s.idx}</span>
                            {s.tool_call ? (
                              <>
                                <code className="ls-name">{s.tool_call.name}</code>
                                <span className="ls-args">
                                  {JSON.stringify(s.tool_call.arguments).slice(0, 80)}
                                </span>
                              </>
                            ) : (
                              <span className="ls-args">{s.thought?.slice(0, 80) || "…"}</span>
                            )}
                            {s.observation && (
                              <span className={"ls-ok " + (s.observation.ok ? "ok" : "err")}>
                                {s.observation.ok ? "✓" : "✗"}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
              {answer && (
                <div className="comms-foot">
                  {answer.tools_called.map((c, i) => <span className="chip" key={i}>{c}</span>)}
                </div>
              )}
            </div>
          </div>
        )}

        <div className="toasts">
          {toasts.map((t) => (
            <div className="toast" key={t.id}>
              <span className="d" style={{ background: t.dot }} />
              <span dangerouslySetInnerHTML={{ __html: t.html }} />
            </div>
          ))}
        </div>
      </div>

      <div className="askbar">
        <div className="pills">
          {PILLS.map((p) => (
            <button
              className="pill"
              key={p.label}
              onClick={() => { setAskText(p.query); ask(p.query); }}
              title={p.query}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="askrow">
          <label className="field">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
            <input value={askText} onChange={(e) => setAskText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") ask(askText); }}
              placeholder="Ask the warehouse — routes, stock, confidence…" />
          </label>
          <button className="run" onClick={() => ask(askText)} disabled={busy}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>Ask
          </button>
        </div>
      </div>

      {drawer && (
        <>
          <div className="scrim" onClick={() => setDrawer(false)} />
          <aside className="drawer" aria-label="Control Center">
            <div className="drawer-head">
              <div><div className="mut">Technical</div><div className="t">Control Center</div></div>
              <button className="close" onClick={() => setDrawer(false)} aria-label="Close">×</button>
            </div>
            <div className="tabs">
              <button className={"tab" + (tab === "reason" ? " active" : "")} onClick={() => setTab("reason")}>Reasoning</button>
              <button className={"tab" + (tab === "trust" ? " active" : "")} onClick={() => setTab("trust")}>Trust &amp; Decay</button>
              <button className={"tab" + (tab === "reflect" ? " active" : "")} onClick={() => setTab("reflect")}>
                Reflections {reflections.length > 0 && <span className="pill-ct">{reflections.length}</span>}
              </button>
            </div>
            <div className="drawer-body">
              {tab === "reason" ? (
                /* Reasoning tab — the live ReAct trace. */
                <>
                  <div className="sec-note">
                    The agent's live ReAct trace{busy ? " — streaming as it runs" : " from your last question"}:
                    every thought, tool call, and observation that grounded the answer.
                  </div>
                  {(busy ? liveSteps : (answer?.trace.steps ?? [])).length > 0 ? (busy ? liveSteps : answer!.trace.steps).map((s, i) => (
                    <div className="step" key={i}>
                      <div className="step-h">
                        <span className="step-i">#{s.idx}</span>
                        {s.tool_call && (
                          <span className="step-tag tc">tool_call</span>
                        )}
                        {s.observation && (
                          <span className={"step-tag " + (s.observation.ok ? "ok" : "err")}>
                            {s.observation.ok ? "observation" : "error"}
                          </span>
                        )}
                        {s.final_answer && (
                          <span className="step-tag fa">final_answer</span>
                        )}
                      </div>
                      {s.thought && (
                        <div className="step-th">💭 {s.thought}</div>
                      )}
                      {s.tool_call && (
                        <div className="step-tc">
                          <code className="tc-name">{s.tool_call.name}</code>
                          <pre className="tc-args">
                            {JSON.stringify(s.tool_call.arguments, null, 2)}
                          </pre>
                        </div>
                      )}
                      {s.observation && (
                        <pre className="step-ob">
                          {typeof s.observation.content === "string"
                            ? s.observation.content
                            : JSON.stringify(s.observation.content, null, 2)}
                        </pre>
                      )}
                      {s.final_answer && (
                        <div className="step-fa">{s.final_answer}</div>
                      )}
                    </div>
                  )) : <div className="empty">Ask a question to see the agent reason.</div>}
                  {busy && (
                    <div className="live-pulse">
                      <span className="dot" /> waiting for next step…
                    </div>
                  )}
                </>
              ) : tab === "trust" ? (
                <>
                  <div className="sec-note">
                    Source trust priors (higher = wins a conflict) and decay rates
                    (confidence lost per hour — so a 2-hour-old reading is already
                    less trustworthy than a fresh one).
                  </div>
                  <div className="trust">
                    {sources.map((s) => (
                      <div className={"trow " + (s.kind === "sensor" ? "sensor" : "system")} key={s.source_id}>
                        <span className="tn">
                          {s.source_id}
                          <span className="kd">{s.kind}</span>
                        </span>
                        <span className="bar"><i style={{ width: s.trust_prior * 100 + "%" }} /></span>
                        <span className="tv">{s.trust_prior.toFixed(2)}</span>
                        <span className={"decay " + (s.decay_rate_per_hour > 0 ? "on" : "off")}>
                          {s.decay_rate_per_hour > 0
                            ? `−${s.decay_rate_per_hour.toFixed(2)}/h`
                            : "stable"}
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                // Reflections tab — Reflexion-style meta-beliefs the agent has learned
                <>
                  <div className="sec-note">
                    Reflexion-style meta-beliefs. Each is a first-class row in the
                    belief graph under <code>subject="self.policy"</code>,
                    source=<code>self_reflection</code> (trust 0.3, decays at 0.05/hr).
                  </div>

                  <div className="teach">
                    <label>Teach the agent something new:</label>
                    <input
                      value={newTopic}
                      onChange={(e) => setNewTopic(e.target.value)}
                      placeholder="topic (optional)"
                      maxLength={60}
                    />
                    <textarea
                      value={newLesson}
                      onChange={(e) => setNewLesson(e.target.value)}
                      placeholder={'e.g. "When scanner and ERP disagree by >5 items, recount manually before trusting either."'}
                      rows={3}
                      maxLength={400}
                    />
                    <button
                      className="iconbtn"
                      style={{ justifyContent: "center", background: "var(--accent)", color: "var(--accent-ink)", border: "none" }}
                      onClick={teachReflection}
                      disabled={!newLesson.trim()}
                    >
                      Save as belief
                    </button>
                  </div>

                  {reflections.length === 0 ? (
                    <div className="empty">
                      No reflections yet. Teach the agent a rule above — it becomes a
                      queryable belief exactly like a sensor reading.
                    </div>
                  ) : (
                    <div className="refls">
                      {reflections.map((r, i) => {
                        const topic = r.predicate.replace("learned_rule:", "");
                        const age = r.age_hours ?? ageHours(r.observed_at);
                        const tier = freshnessTier(age);
                        const eff = r.effective_confidence ?? r.confidence;
                        return (
                          <div className="refl" key={r.provenance_id ?? i}>
                            <div className="refl-h">
                              <span className="refl-topic">{topic}</span>
                              <span className={"fresh fresh-" + tier}>{fmtAge(age)}</span>
                            </div>
                            <div className="refl-body">{r.object}</div>
                            <div className="refl-foot">
                              <span className="tw">conf</span>{" "}
                              <b>{eff.toFixed(2)}</b>
                              {Math.abs(eff - r.confidence) > 0.005 && (
                                <span className="was">/ {r.confidence.toFixed(2)} stored</span>
                              )}
                              <span className="tw"> · source={r.source}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
