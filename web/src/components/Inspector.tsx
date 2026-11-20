import { useWarehouse } from "../context/WarehouseContext";
import {
  ageHours,
  dirGlyph,
  fmtAge,
  fmtTime,
  freshnessTier,
  perspectiveKind,
  perspectiveLabel,
} from "../util";

export type InspectorTab = "now" | "selection";

interface Props {
  tab: InspectorTab;
  setTab: (t: InspectorTab) => void;
}

export default function Inspector({ tab, setTab }: Props) {
  const { shelfSel } = useWarehouse();

  const TABS: { id: InspectorTab; label: string; count?: number }[] = [
    { id: "now", label: "Now" },
    { id: "selection", label: "Selection", count: shelfSel ? 1 : undefined },
  ];

  return (
    <aside className="insp" aria-label="Inspector">
      <nav className="insp-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={"insp-tab" + (tab === t.id ? " active" : "")}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.count !== undefined && <span className="insp-ct">{t.count}</span>}
          </button>
        ))}
      </nav>

      <div className="insp-body">
        {tab === "now" ? <NowPanel /> : <SelectionPanel />}
      </div>
    </aside>
  );
}

// ────────────────────────────────── NOW ───────────────────────────────────

function NowPanel() {
  const { view, busy, answer, liveSteps, dismissAnswer, doMove } = useWarehouse();
  if (!view) return null;

  const here = view.junctions[view.position];
  const goalShelf = view.shelves.find((s) => s.id === view.goal);
  const arrived = goalShelf && goalShelf.near === view.position;

  const liveNow = busy ? liveSteps : answer?.trace.steps || [];
  const lastStep = liveNow[liveNow.length - 1];

  return (
    <div className="np">
      <section className="np-pos">
        <div className="np-pos-h">
          <span className="np-pin" aria-hidden />
          <div>
            <div className="np-l">AGV-1</div>
            <div className="np-v">{here?.label ?? view.position}</div>
          </div>
          <div className="np-goal">
            <div className="np-l">Goal</div>
            <div className="np-v">{view.goal}</div>
          </div>
        </div>

        {arrived && (
          <div className="np-arrived">
            ✓ Arrived at {view.goal} — high-confidence route complete.
          </div>
        )}
      </section>

      {!arrived && view.options.length > 0 && (
        <section className="np-sect">
          <h3 className="np-h">Next move</h3>
          <div className="np-opts">
            {view.options.map((o) => {
              const to = view.junctions[o.to];
              const pct = Math.round(o.confidence * 100);
              const glyph = here && to ? dirGlyph(here, to) : "•";
              const goalHint = view.shelves.some(
                (s) => s.id === view.goal && s.near === o.to,
              );
              return (
                <button
                  key={o.to}
                  className={"np-opt tier-" + o.tier}
                  onClick={() => doMove(o.to)}
                  title={o.reason}
                >
                  <span className="np-opt-dir">{glyph}</span>
                  <span className="np-opt-info">
                    <span className="np-opt-name">
                      {to?.label ?? o.to}
                      {goalHint && <em> · goal shelf</em>}
                    </span>
                    <span className="np-opt-reason">
                      {o.passable
                        ? o.tier === "high"
                          ? "Clear — sensor-confirmed"
                          : "Map only — unverified"
                        : "Sensor conflict — avoid"}
                    </span>
                  </span>
                  <span className="np-opt-pct">{pct}%</span>
                </button>
              );
            })}
          </div>
        </section>
      )}

      <section className="np-sect">
        <div className="np-h-row">
          <h3 className="np-h">Agent</h3>
          {busy && <span className="np-pulse"><span className="dot" />reasoning · step {liveNow.length}</span>}
          {answer && !busy && (
            <button className="np-x" onClick={dismissAnswer} aria-label="Dismiss answer">clear</button>
          )}
        </div>

        {!busy && !answer && (
          <div className="np-ghost">
            Ask the agent anything about the warehouse using the command bar
            below — its reasoning will stream here.
          </div>
        )}

        {answer && !busy && (
          <div className="np-answer">{answer.answer}</div>
        )}

        {liveNow.length > 0 && (
          <ol className="np-trace">
            {liveNow.slice(-10).map((s, i) => (
              <li key={`${s.idx}-${i}`} className="np-step">
                <span className="np-step-i">#{s.idx}</span>
                {s.tool_call ? (
                  <>
                    <code className="np-step-n">{s.tool_call.name}</code>
                    <span className="np-step-a">
                      {JSON.stringify(s.tool_call.arguments).slice(0, 60)}
                    </span>
                  </>
                ) : (
                  <span className="np-step-a np-step-thought">
                    {(s.thought ?? "").slice(0, 80) || "…"}
                  </span>
                )}
                {s.observation && (
                  <span className={"np-step-o " + (s.observation.ok ? "ok" : "err")}>
                    {s.observation.ok ? "✓" : "✗"}
                  </span>
                )}
              </li>
            ))}
            {busy && lastStep && (
              <li className="np-step-wait">
                <span className="dot" /> waiting for next step…
              </li>
            )}
          </ol>
        )}
      </section>
    </div>
  );
}

// ────────────────────────────── SELECTION ────────────────────────────────

function SelectionPanel() {
  const {
    view, shelfSel, inv, bin, prov, closeSelection,
    xferDest, setXferDest, xferQty, setXferQty, doTransfer, recentArrival, sources,
  } = useWarehouse();

  if (!view || !shelfSel) {
    return (
      <div className="insp-empty">
        <p>Click a shelf on the warehouse floor to inspect it — you'll see
        its reconciled stock, every source's claim, and the full provenance
        timeline here.</p>
      </div>
    );
  }

  const otherShelves = view.shelves.filter((s) => s.id !== shelfSel.id);
  const disputed = inv && inv.sources.length > 1;

  return (
    <div className="sp">
      <header className="sp-h">
        <div>
          <div className="sp-id">{shelfSel.id}</div>
          <div className="sp-loc">{view.junctions[shelfSel.near]?.label}</div>
        </div>
        <span className={"sp-pill " + (disputed ? "alert" : "clear")}>
          {disputed ? "DISPUTED" : shelfSel.low ? "LOW" : "IN STOCK"}
        </span>
        <button className="np-x" onClick={closeSelection} aria-label="Close">×</button>
      </header>

      {inv && inv.sources.length > 1 && (
        <section className="sp-sect">
          <div className="sp-note">Sources disagree — each from its own perspective:</div>
          <div className="sp-rows">
            {inv.sources.map((s, i) => (
              <div className="sp-row" key={i}>
                <span className={"psp psp-" + perspectiveKind(s.perspective)}>
                  {perspectiveLabel(s.perspective)}
                </span>
                <span className="sp-row-name">
                  {s.source}
                  {s.note && <em> · {s.note}</em>}
                </span>
                <span className="sp-row-v">
                  {s.predicate === "shipped" ? "−" : ""}
                  {s.value}
                </span>
              </div>
            ))}
          </div>
          <div className="sp-reconcile">
            ✓ Reconciled to <b>≈ {inv.reconciled}</b> · {inv.explanation}
          </div>
        </section>
      )}

      {bin && (
        <section className="sp-sect">
          <h4 className="sp-sh">Bin contents · total {bin.total_units}</h4>
          {bin.lots.length === 0 ? (
            <div className="sp-ghost">Empty.</div>
          ) : (
            <div className="sp-rows">
              {bin.lots.map((lot, i) => (
                <div
                  className={"sp-row sp-lot" + (lot.is_native ? "" : " xferred")}
                  key={i}
                >
                  <span className="sp-row-name">
                    <b>{lot.sku}</b>
                    <em>{lot.is_native ? " · native" : " · transferred in"}</em>
                  </span>
                  <span className="sp-row-v">{lot.quantity}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {prov && prov.length > 0 && (
        <section className="sp-sect">
          <h4 className="sp-sh">Provenance timeline · SQLite log</h4>
          <ol className="prov">
            {prov.slice(0, 8).map((row, i) => {
              const reason = (row.payload as { reason?: string } | null | undefined)?.reason;
              const src = sources.find((s) => s.source_id === row.source_id);
              const age = ageHours(row.observed_at);
              const tier = freshnessTier(age);
              const decayed = src
                ? Math.max(0, row.confidence - src.decay_rate_per_hour * age)
                : row.confidence;
              return (
                <li key={row.provenance_id ?? i} className={"pv pv-" + row.status}>
                  <span className="pv-t">{fmtTime(row.recorded_at ?? row.observed_at)}</span>
                  <span className="pv-m">
                    <span className={"psp psp-" + perspectiveKind(row.perspective)}>
                      {perspectiveLabel(row.perspective)}
                    </span>
                    <b>{row.source_id}</b>
                    <span className="tw">{row.predicate}=</span>
                    <span className="ct">{row.object}</span>
                    {reason && <span className="tw"> · {reason}</span>}
                    <span className={"fresh fresh-" + tier}>
                      {fmtAge(age)}
                      {src && src.decay_rate_per_hour > 0 && (
                        <span className="dec">
                          {" · "}conf {decayed.toFixed(2)}
                          <span className="was">/{row.confidence.toFixed(2)}</span>
                        </span>
                      )}
                    </span>
                  </span>
                  {row.status !== "active" && <span className="pv-s">{row.status}</span>}
                </li>
              );
            })}
          </ol>
        </section>
      )}

      {recentArrival && (
        <div className="sp-arrival">{recentArrival}</div>
      )}

      <section className="sp-sect">
        <h4 className="sp-sh">Transfer stock</h4>
        <div className="xfer-row">
          <input
            type="number"
            min={1}
            value={xferQty}
            onChange={(e) => setXferQty(Math.max(1, Number(e.target.value) || 1))}
            aria-label="Quantity"
          />
          <select value={xferDest} onChange={(e) => setXferDest(e.target.value)}>
            {otherShelves.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} · {view.junctions[s.near]?.label}
              </option>
            ))}
          </select>
        </div>
        <button className="btn-primary" onClick={doTransfer}>Confirm transfer</button>
      </section>
    </div>
  );
}
