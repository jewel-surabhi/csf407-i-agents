import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useWarehouse } from "../context/WarehouseContext";
import type { EpistemicTraceSummary } from "../api/types";

export default function DashboardPage() {
  const { view, sources, reflections } = useWarehouse();
  const [tests, setTests] = useState<EpistemicTraceSummary[] | null>(null);

  useEffect(() => {
    api.epistemicTests().then(setTests).catch(() => setTests([]));
  }, []);

  const recorded = tests?.filter((t) => t.mode === "recorded") ?? [];
  const live = tests?.filter((t) => t.mode === "live") ?? [];
  const recordedPass = recorded.filter((t) => t.verdict === "pass").length;
  const livePass = live.filter((t) => t.verdict === "pass").length;

  const disputed = view?.shelves.filter((s) => s.stock.length > 1).length ?? 0;
  const lowStock = view?.shelves.filter((s) => s.low).length ?? 0;
  const here = view?.junctions[view.position];

  return (
    <main className="page">
      <div className="page-inner">
        <header className="page-h dash-h">
          <h2>Warehouse 7</h2>
          <p>
            A grounded, three-layer epistemic agent — every claim traces to a
            source with a trust prior, confidence decays over time, and the
            agent learns from its own reasoning. This dashboard is the
            jumping-off point into each part of the system.
          </p>
        </header>

        <section className="kpi-row">
          <KpiCard
            label="Epistemic tests · recorded"
            value={tests ? `${recordedPass}/${recorded.length}` : "…"}
            tone={tests && recordedPass === recorded.length ? "clear" : "warn"}
          />
          <KpiCard
            label="Epistemic tests · live LLM"
            value={tests ? `${livePass}/${live.length}` : "…"}
            tone={tests && livePass === live.length ? "clear" : "warn"}
          />
          <KpiCard label="Registered sources" value={String(sources.length)} tone="neutral" />
          <KpiCard label="Agent reflections" value={String(reflections.length)} tone="neutral" />
          <KpiCard
            label="Disputed / low stock"
            value={`${disputed} / ${lowStock}`}
            tone={disputed > 0 ? "warn" : "clear"}
          />
          <KpiCard
            label="AGV-1 position"
            value={here?.label ?? view?.position ?? "…"}
            tone="neutral"
            mono
          />
        </section>

        <section className="nav-cards">
          <NavCard
            to="/operations"
            eyebrow="Live"
            title="Operations"
            desc="The warehouse floor, move confidence, and a live-streaming agent reasoning panel."
          />
          <NavCard
            to="/epistemic"
            eyebrow="Evidence"
            title="Epistemic Tests"
            desc="The 10 grounded-reasoning tests that back every claim this project makes, with full traces."
          />
          <NavCard
            to="/scenario-b"
            eyebrow="From the proposal"
            title="Scenario B"
            desc="Tracking perspective: user expects red, the camera sees brown, history says blue — all three, never merged."
          />
          <NavCard
            to="/trust"
            eyebrow="Reference"
            title="Trust & Sources"
            desc="Every registered source's trust prior and temporal decay rate, ranked."
          />
          <NavCard
            to="/memory"
            eyebrow="Reflexion"
            title="Agent Memory"
            desc="Meta-beliefs the agent writes about its own reasoning, and a form to teach it new ones."
          />
          <NavCard
            to="/architecture"
            eyebrow="Explainer"
            title="Architecture"
            desc="The three-layer diagram and the reasoning behind grounding, decay, reflection, and events."
          />
        </section>
      </div>
    </main>
  );
}

function KpiCard({
  label, value, tone, mono,
}: { label: string; value: string; tone: "clear" | "warn" | "neutral"; mono?: boolean }) {
  return (
    <div className={"kpi kpi-" + tone}>
      <div className="kpi-v" style={mono ? { fontFamily: "var(--f-mono)", fontSize: 16 } : undefined}>
        {value}
      </div>
      <div className="kpi-l">{label}</div>
    </div>
  );
}

function NavCard({
  to, eyebrow, title, desc,
}: { to: string; eyebrow: string; title: string; desc: string }) {
  return (
    <Link to={to} className="nav-card">
      <span className="nav-card-eyebrow">{eyebrow}</span>
      <h3>{title}</h3>
      <p>{desc}</p>
      <span className="nav-card-go">Open →</span>
    </Link>
  );
}
