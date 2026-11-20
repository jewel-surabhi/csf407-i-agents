import { useWarehouse } from "../context/WarehouseContext";
import { ageHours, fmtAge, freshnessTier } from "../util";

export default function MemoryPanel() {
  const {
    reflections, newTopic, setNewTopic, newLesson, setNewLesson, teachReflection,
  } = useWarehouse();

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
          value={newTopic}
          onChange={(e) => setNewTopic(e.target.value)}
          placeholder="topic (optional)"
          maxLength={60}
        />
        <textarea
          value={newLesson}
          onChange={(e) => setNewLesson(e.target.value)}
          placeholder={
            'e.g. "When scanner and ERP disagree by >5 items, recount manually before trusting either."'
          }
          rows={3}
          maxLength={400}
        />
        <button
          className="btn-primary"
          onClick={teachReflection}
          disabled={!newLesson.trim()}
        >
          Save as belief
        </button>
      </div>

      {reflections.length === 0 ? (
        <div className="insp-empty">
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
