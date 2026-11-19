import { useMemo } from "react";
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
} from "../api/types";
import {
  ageHours,
  dirGlyph,
  fmtAge,
  fmtTime,
  freshnessTier,
  perspectiveKind,
  perspectiveLabel,
} from "../util";

export type InspectorTab = "now" | "selection" | "trust" | "memory";

interface Props {
  view: WarehouseView;
  tab: InspectorTab;
  setTab: (t: InspectorTab) => void;

  // selection state
  shelfSel: Shelf | null;
  inv: InventoryView | null;
  bin: ShelfContents | null;
  prov: ProvenanceEntry[] | null;
  closeSelection: () => void;

  // transfer
  xferDest: string;
  setXferDest: (v: string) => void;
  xferQty: number;
  setXferQty: (n: number) => void;
  doTransfer: () => void;
  recentArrival: string | null;

  // agent
  busy: boolean;
  answer: AgentResult | null;
  liveSteps: TraceStep[];
  dismissAnswer: () => void;

  // settings / reference
  sources: Source[];
  reflections: Belief[];
  newTopic: string;
  setNewTopic: (v: string) => void;
  newLesson: string;
  setNewLesson: (v: string) => void;
  teachReflection: () => void;

  // navigation
  onMove: (to: string) => void;
}

export default function Inspector(p: Props) {
  const TABS: { id: InspectorTab; label: string; count?: number }[] = [
    { id: "now", label: "Now" },
    { id: "selection", label: "Selection", count: p.shelfSel ? 1 : undefined },
    { id: "trust", label: "Trust" },
    { id: "memory", label: "Memory", count: p.reflections.length || undefined },
  ];

  return (
    <aside className="insp" aria-label="Inspector">
      <nav className="insp-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={"insp-tab" + (p.tab === t.id ? " active" : "")}
            onClick={() => p.setTab(t.id)}
          >
            {t.label}
            {t.count !== undefined && <span className="insp-ct">{t.count}</span>}
          </button>
        ))}
      </nav>

      <div className="insp-body">
        {p.tab === "now" && <NowPanel {...p} />}
        {p.tab === "selection" && <SelectionPanel {...p} />}
        {p.tab === "trust" && <TrustPanel sources={p.sources} />}
        {p.tab === "memory" && <MemoryPanel {...p} />}
      </div>
    </aside>
  );
}

// ────────────────────────────────── NOW ───────────────────────────────────

function NowPanel(p: Props) {
  const here = p.view.junctions[p.view.position];
  const goalShelf = p.view.shelves.find((s) => s.id === p.view.goal);
  const arrived = goalShelf && goalShelf.near === p.view.position;

  const liveNow = p.busy ? p.liveSteps : p.answer?.trace.steps || [];
  const lastStep = liveNow[liveNow.length - 1];

  return (
    <div className="np">
      {/* Position + goal strip */}
      <section className="np-pos">
        <div className="np-pos-h">
          <span className="np-pin" aria-hidden />
          <div>
            <div className="np-l">AGV-1</div>
            <div className="np-v">{here?.label ?? p.view.position}</div>
          </div>
          <div className="np-goal">
            <div className="np-l">Goal</div>
            <div className="np-v">{p.view.goal}</div>
          </div>
        </div>

        {arrived && (
          <div className="np-arrived">
            ✓ Arrived at {p.view.goal} — high-confidence route complete.
          </div>
        )}
      </section>

      {/* Move options — the old Decision Panel lives here now */}
      {!arrived && p.view.options.length > 0 && (
        <section className="np-sect">
          <h3 className="np-h">Next move</h3>
          <div className="np-opts">
            {p.view.options.map((o) => {
              const to = p.view.junctions[o.to];
              const pct = Math.round(o.confidence * 100);
              const glyph = here && to ? dirGlyph(here, to) : "•";
              const goalHint = p.view.shelves.some(
                (s) => s.id === p.view.goal && s.near === o.to,
              );
              return (
                <button
                  key={o.to}
                  className={"np-opt tier-" + o.tier}
                  onClick={() => p.onMove(o.to)}
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

      {/* Agent reasoning — always visible, streams while busy */}
      <section className="np-sect">
        <div className="np-h-row">
          <h3 className="np-h">Agent</h3>
          {p.busy && <span className="np-pulse"><span className="dot" />reasoning · step {liveNow.length}</span>}
          {p.answer && !p.busy && (
            <button className="np-x" onClick={p.dismissAnswer} aria-label="Dismiss answer">clear</button>
          )}
        </div>

        {!p.busy && !p.answer && (
          <div className="np-ghost">
            Ask the agent anything about the warehouse using the command bar
            below — its reasoning will stream here.
          </div>
        )}

        {p.answer && !p.busy && (
          <div className="np-answer">{p.answer.answer}</div>
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
            {p.busy && lastStep && (
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

function SelectionPanel(p: Props) {
  if (!p.shelfSel) {
    return (
      <div className="insp-empty">
        <p>Click a shelf on the warehouse floor to inspect it — you'll see
        its reconciled stock, every source's claim, and the full provenance
        timeline here.</p>
      </div>
    );
  }

  const otherShelves = p.view.shelves.filter((s) => s.id !== p.shelfSel!.id);
  const disputed = p.inv && p.inv.sources.length > 1;

  return (
    <div className="sp">
      <header className="sp-h">
        <div>
          <div className="sp-id">{p.shelfSel.id}</div>
          <div className="sp-loc">{p.view.junctions[p.shelfSel.near]?.label}</div>
        </div>
        <span className={"sp-pill " + (disputed ? "alert" : "clear")}>
          {disputed ? "DISPUTED" : p.shelfSel.low ? "LOW" : "IN STOCK"}
        </span>
        <button className="np-x" onClick={p.closeSelection} aria-label="Close">×</button>
      </header>

      {p.inv && p.inv.sources.length > 1 && (
        <section className="sp-sect">
          <div className="sp-note">Sources disagree — each from its own perspective:</div>
          <div className="sp-rows">
            {p.inv.sources.map((s, i) => (
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
            ✓ Reconciled to <b>≈ {p.inv.reconciled}</b> · {p.inv.explanation}
          </div>
        </section>
      )}

      {p.bin && (
        <section className="sp-sect">
          <h4 className="sp-sh">Bin contents · total {p.bin.total_units}</h4>
          {p.bin.lots.length === 0 ? (
            <div className="sp-ghost">Empty.</div>
          ) : (
            <div className="sp-rows">
              {p.bin.lots.map((lot, i) => (
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

      {p.prov && p.prov.length > 0 && (
        <section className="sp-sect">
          <h4 className="sp-sh">Provenance timeline · SQLite log</h4>
          <ol className="prov">
            {p.prov.slice(0, 8).map((row, i) => {
              const reason = (row.payload as { reason?: string } | null | undefined)?.reason;
              const src = p.sources.find((s) => s.source_id === row.source_id);
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

      {p.recentArrival && (
        <div className="sp-arrival">{p.recentArrival}</div>
      )}

      <section className="sp-sect">
        <h4 className="sp-sh">Transfer stock</h4>
        <div className="xfer-row">
          <input
            type="number"
            min={1}
            value={p.xferQty}
            onChange={(e) => p.setXferQty(Math.max(1, Number(e.target.value) || 1))}
            aria-label="Quantity"
          />
          <select value={p.xferDest} onChange={(e) => p.setXferDest(e.target.value)}>
            {otherShelves.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} · {p.view.junctions[s.near]?.label}
              </option>
            ))}
          </select>
        </div>
        <button className="btn-primary" onClick={p.doTransfer}>Confirm transfer</button>
      </section>
    </div>
  );
}

// ──────────────────────────────── TRUST ──────────────────────────────────

function TrustPanel({ sources }: { sources: Source[] }) {
  const ordered = useMemo(
    () => [...sources].sort((a, b) => b.trust_prior - a.trust_prior),
    [sources],
  );
  return (
    <div className="tp">
      <p className="insp-note">
        Trust priors drive who wins a conflict (higher = wins). Decay rates
        model how fast each source's beliefs lose confidence per hour.
      </p>
      <div className="trust">
        {ordered.map((s) => (
          <div
            className={"trow " + (s.kind === "sensor" ? "sensor" : "system")}
            key={s.source_id}
          >
            <span className="tn">
              {s.source_id}
              <span className="kd">{s.kind}</span>
            </span>
            <span className="bar">
              <i style={{ width: s.trust_prior * 100 + "%" }} />
            </span>
            <span className="tv">{s.trust_prior.toFixed(2)}</span>
            <span className={"decay " + (s.decay_rate_per_hour > 0 ? "on" : "off")}>
              {s.decay_rate_per_hour > 0
                ? `−${s.decay_rate_per_hour.toFixed(2)}/h`
                : "stable"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ──────────────────────────────── MEMORY ─────────────────────────────────

function MemoryPanel(p: Props) {
  return (
    <div className="mp">
      <p className="insp-note">
        Reflexion-style meta-beliefs the agent carries about its own
        reasoning. Stored as first-class beliefs under <code>self.policy</code>,
        source <code>self_reflection</code>, trust 0.3, decays at 0.05/hr.
      </p>

      <div className="teach">
        <label>Teach the agent a new rule:</label>
        <input
          value={p.newTopic}
          onChange={(e) => p.setNewTopic(e.target.value)}
          placeholder="topic (optional)"
          maxLength={60}
        />
        <textarea
          value={p.newLesson}
          onChange={(e) => p.setNewLesson(e.target.value)}
          placeholder={
            'e.g. "When scanner and ERP disagree by >5 items, recount manually before trusting either."'
          }
          rows={3}
          maxLength={400}
        />
        <button
          className="btn-primary"
          onClick={p.teachReflection}
          disabled={!p.newLesson.trim()}
        >
          Save as belief
        </button>
      </div>

      {p.reflections.length === 0 ? (
        <div className="insp-empty">
          No reflections yet. Teach the agent a rule above — it becomes a
          queryable belief exactly like a sensor reading.
        </div>
      ) : (
        <div className="refls">
          {p.reflections.map((r, i) => {
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
                  conf <b>{eff.toFixed(2)}</b>
                  {Math.abs(eff - r.confidence) > 0.005 && (
                    <span className="was"> / {r.confidence.toFixed(2)} stored</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
