import { useMemo } from "react";
import { useWarehouse } from "../context/WarehouseContext";

export default function TrustPanel() {
  const { sources } = useWarehouse();
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
