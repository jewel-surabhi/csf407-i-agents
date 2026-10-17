import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import type {
  AgentResult,
  InventoryView,
  Shelf,
  ShelfContents,
  WarehouseView,
} from "./api/types";
import WarehouseFloor from "./components/WarehouseFloor";
import DecisionPanel from "./components/DecisionPanel";

const TRUST = [
  { nm: "Front LiDAR", kd: "Sensor", trust: 0.9, cls: "sensor" },
  { nm: "Floor Scanner", kd: "Scanner", trust: 0.9, cls: "sensor" },
  { nm: "Sales Log", kd: "System", trust: 0.8, cls: "system" },
  { nm: "Dock Camera", kd: "Sensor", trust: 0.7, cls: "sensor" },
  { nm: "ERP System", kd: "System", trust: 0.6, cls: "system" },
  { nm: "Floor Map", kd: "Map", trust: 0.6, cls: "system" },
];

const PILLS = [
  "Route me to WIDGET-42",
  "How many WIDGET-42?",
  "Why avoid the mid-aisle?",
];

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
  const [tab, setTab] = useState<"reason" | "trust">("reason");
  const [shelfSel, setShelfSel] = useState<Shelf | null>(null);
  const [inv, setInv] = useState<InventoryView | null>(null);
  const [bin, setBin] = useState<ShelfContents | null>(null);
  const [recentArrival, setRecentArrival] = useState<string | null>(null);
  const [xferDest, setXferDest] = useState<string>("");
  const [xferQty, setXferQty] = useState<number>(10);
  const [answer, setAnswer] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [askText, setAskText] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);

  const toast = useCallback((dot: string, html: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, dot, html }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const load = useCallback(async () => {
    try {
      setView(await api.getWarehouse());
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

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
        const goal = next.shelves.find((s) => s.id === next.goal);
        if (goal && goal.near === next.position) toast("var(--accent)", `Reached <b>${next.goal}</b> ★`);
      } catch (e) {
        toast("var(--blocked)", String(e));
      }
    },
    [view, toast],
  );

  const selectShelf = useCallback(async (id: string) => {
    if (!view) return;
    const sh = view.shelves.find((s) => s.id === id) ?? null;
    setShelfSel(sh);
    setInv(null);
    setBin(null);
    setRecentArrival(null);
    const others = view.shelves.filter((s) => s.id !== id);
    setXferDest(others[0]?.id ?? "");
    try {
      // Fetch both: the per-SKU reconciliation (shows the multi-source dispute)
      // AND the shelf's full contents (shows every lot stored here, including
      // units transferred in).
      const [i, b] = await Promise.all([api.inventory(id), api.bin(id)]);
      setInv(i);
      setBin(b);
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
      const [v, srcInv, srcBin, destBin] = await Promise.all([
        api.getWarehouse(),
        api.inventory(shelfSel.id),
        api.bin(shelfSel.id),
        api.bin(xferDest),
      ]);
      setView(v);
      setInv(srcInv);
      setBin(srcBin);
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
    async (text: string) => {
      if (!text.trim()) return;
      setBusy(true);
      setAnswer(null);
      try {
        const res = await api.query(text);
        setAnswer(res);
      } catch (e) {
        toast("var(--blocked)", "Agent error: " + String(e));
      } finally {
        setBusy(false);
      }
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
                  <div style={{ fontSize: 12, color: "var(--ink-2)" }}>Sources disagree:</div>
                  {inv.sources.map((s, i) => (
                    <div className="srcrow" key={i}>
                      <span className="nm">{s.source} {s.note && <span className="tw">·{s.note}</span>}</span>
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
              <div className="comms-body">{busy ? "Thinking through the layers…" : answer?.answer}</div>
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
            <button className="pill" key={p} onClick={() => { setAskText(p); ask(p); }}>{p}</button>
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
              <button className={"tab" + (tab === "trust" ? " active" : "")} onClick={() => setTab("trust")}>Trust</button>
            </div>
            <div className="drawer-body">
              {tab === "reason" ? (
                <>
                  <div className="sec-note">The agent's live ReAct trace from your last question.</div>
                  {answer ? answer.trace.steps.map((s, i) => (
                    <div className="belief" key={i}>
                      <span className="bid">{s.tool_call ? `${s.tool_call.name}()` : (s.thought ?? "step")}</span>
                      {s.observation && <span className="bst">{s.observation.ok ? "ok" : "err"}</span>}
                    </div>
                  )) : <div className="empty">Ask a question to see the agent reason.</div>}
                </>
              ) : (
                <>
                  <div className="sec-note">Source trust priors — the higher prior wins a conflict.</div>
                  <div className="trust">
                    {TRUST.map((s) => (
                      <div className={"trow " + s.cls} key={s.nm}>
                        <span className="tn">{s.nm}<span className="kd">{s.kd}</span></span>
                        <span className="bar"><i style={{ width: s.trust * 100 + "%" }} /></span>
                        <span className="tv">{s.trust.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
