import { NavLink, Outlet } from "react-router-dom";
import { useWarehouse } from "../context/WarehouseContext";

export default function Layout() {
  const { view, offline, load, toggleTheme, reset, toasts } = useWarehouse();

  if (offline) {
    return (
      <div className="offline">
        <div className="card">
          <h2>Backend not reachable</h2>
          <p>Start the FastAPI server so the warehouse can load its grounded state:</p>
          <pre>$env:PYTHONPATH='src'; python -m uvicorn iagent.api.server:app --reload --port 8000 --app-dir src</pre>
          <p>Then reload this page. The dev server proxies <code>/api</code> to port 8000.</p>
          <button className="btn-primary" onClick={load}>Retry</button>
        </div>
      </div>
    );
  }

  if (!view) {
    return <div className="offline"><div className="card"><h2>Loading Warehouse 7…</h2></div></div>;
  }

  const here = view.junctions[view.position];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          <h1>Warehouse&nbsp;7</h1>
          <span className="brand-sub">Ops&nbsp;AI</span>
        </div>

        <nav className="topnav">
          <NavLink to="/" end className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Dashboard
          </NavLink>
          <NavLink to="/operations" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Operations
          </NavLink>
          <NavLink to="/epistemic" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Epistemic Tests
          </NavLink>
          <NavLink to="/scenario-b" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Scenario B
          </NavLink>
          <NavLink to="/trust" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Trust
          </NavLink>
          <NavLink to="/memory" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Memory
          </NavLink>
          <NavLink to="/architecture" className={({ isActive }) => "topnav-l" + (isActive ? " active" : "")}>
            Architecture
          </NavLink>
        </nav>

        <div className="status">
          <span className="status-l">AGV-1</span>
          <span className="status-v">{here?.label ?? view.position}</span>
          <span className="status-sep" />
          <span className="status-l">Goal</span>
          <span className="status-v">{view.goal}</span>
        </div>

        <div className="actions">
          <button className="icnbtn ghost" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
          </button>
          <button className="icnbtn" onClick={reset} title="Restore the warehouse to its seeded starting state">Reset</button>
        </div>
      </header>

      <Outlet />

      <div className="toasts">
        {toasts.map((t) => (
          <div className="toast" key={t.id}>
            <span className="d" style={{ background: t.dot }} />
            <span dangerouslySetInnerHTML={{ __html: t.html }} />
          </div>
        ))}
      </div>
    </div>
  );
}
