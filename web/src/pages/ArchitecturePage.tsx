export default function ArchitecturePage() {
  return (
    <main className="page">
      <div className="page-inner">
        <header className="page-h">
          <h2>Architecture</h2>
          <p>
            Warehouse 7 is a demonstration of a three-layer epistemic agent —
            a system designed so every claim it makes can be traced back to a
            source, and every source carries a trust prior that resolves
            conflicts instead of averaging them away.
          </p>
        </header>

        <section className="arch-diagram">
          <div className="arch-box arch-declarative">
            <h3>Declarative</h3>
            <p>Belief graph + provenance log</p>
            <ul>
              <li>Every claim: subject, predicate, object</li>
              <li>Source + trust prior + confidence</li>
              <li>Perspective: self / user / third-party</li>
              <li>Append-only SQLite history</li>
            </ul>
          </div>
          <div className="arch-arrow">↓ reads / writes</div>
          <div className="arch-box arch-procedural">
            <h3>Procedural</h3>
            <p>ReAct loop + Groq LLM</p>
            <ul>
              <li>Tool calls: get_belief, list_subjects…</li>
              <li>Streams each step live over SSE</li>
              <li>Recipes keep episodes to 2–4 calls</li>
              <li>Auto-finalize safety net</li>
            </ul>
          </div>
          <div className="arch-arrow">↓ senses / acts</div>
          <div className="arch-box arch-sensorimotor">
            <h3>Sensorimotor</h3>
            <p>LiDAR, scanner, actuators</p>
            <ul>
              <li>Live confidence per segment</li>
              <li>Distance-dependent LiDAR reads</li>
              <li>move / rotate / grasp (flagged)</li>
            </ul>
          </div>
        </section>

        <section className="arch-sect">
          <h3>Grounding, not guessing</h3>
          <p>
            When a user asks "is Aisle B clear?", the agent doesn't answer
            from its own impression — it calls <code>get_belief</code>,
            reads the active row for that segment, and reports the stored
            source. If the floor map and the LiDAR disagree, the agent calls
            <code> list_sources</code>, compares trust priors, and writes a
            new belief citing the winner. Nothing is asserted without a
            tool call backing it.
          </p>
        </section>

        <section className="arch-sect">
          <h3>Temporal decay</h3>
          <p>
            Confidence is not static. Each source has a decay rate — LiDAR
            readings lose 0.30 confidence per hour, the floor scanner is
            stable, static map data never decays. The effective confidence
            shown anywhere in the UI is computed live from a belief's age
            and its source's decay rate, so a reading from three hours ago
            is visibly less trustworthy than one from thirty seconds ago.
          </p>
        </section>

        <section className="arch-sect">
          <h3>Reflections</h3>
          <p>
            After an episode, the agent can write a meta-belief about its
            own reasoning — a Reflexion-style lesson stored under
            <code> self.policy</code> with source <code>self_reflection</code>.
            These aren't a separate log: they're ordinary beliefs with their
            own trust prior (0.30) and decay rate, queryable the same way as
            a sensor reading. You can teach the agent new rules directly
            from the Memory page.
          </p>
        </section>

        <section className="arch-sect">
          <h3>Dynamic events</h3>
          <p>
            The warehouse isn't static. A small YAML-driven event engine
            fires belief updates as the session progresses — an ERP sync
            that revises a stock count, a LiDAR obstacle discovered mid-route
            — so the agent has to reconcile information it didn't have when
            the episode started, not just replay a fixed scenario.
          </p>
        </section>

        <section className="arch-sect">
          <h3>Why this matters</h3>
          <p>
            Most LLM agents answer from pattern-matching over their training
            data, with no way to tell a grounded claim from a confident
            guess. This system makes that distinction structural: every
            answer traces to a provenance row, every provenance row has a
            named source and a trust prior, and the UI surfaces that chain
            instead of hiding it behind a single confident-sounding sentence.
          </p>
        </section>
      </div>
    </main>
  );
}
