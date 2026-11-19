import MemoryPanel from "../components/MemoryPanel";

export default function MemoryPage() {
  return (
    <main className="page">
      <div className="page-inner page-inner-narrow">
        <header className="page-h">
          <h2>Agent Memory</h2>
          <p>
            Reflexion-style meta-beliefs the agent writes about its own
            reasoning after an episode. They aren't a side log — they're
            ordinary beliefs in the same graph, with their own source,
            trust prior, and decay, queryable exactly like a sensor reading.
          </p>
        </header>
        <MemoryPanel />
      </div>
    </main>
  );
}
