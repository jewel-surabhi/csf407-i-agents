import TrustPanel from "../components/TrustPanel";

export default function TrustPage() {
  return (
    <main className="page">
      <div className="page-inner page-inner-narrow">
        <header className="page-h">
          <h2>Trust &amp; Sources</h2>
          <p>
            Every belief in the graph is traceable to a registered source with
            a trust prior and a decay rate. When sources disagree, the agent
            doesn't average or guess — it looks up this table.
          </p>
        </header>
        <TrustPanel />
      </div>
    </main>
  );
}
