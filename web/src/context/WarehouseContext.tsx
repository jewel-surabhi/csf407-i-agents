import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api/client";
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

export interface Toast {
  id: number;
  dot: string;
  html: string;
}

interface Ctx {
  view: WarehouseView | null;
  offline: boolean;
  load: () => Promise<void>;

  pathMap: boolean;
  setPathMap: (fn: (v: boolean) => boolean) => void;
  toggleTheme: () => void;

  toasts: Toast[];
  toast: (dot: string, html: string) => void;

  // selection
  shelfSel: Shelf | null;
  inv: InventoryView | null;
  bin: ShelfContents | null;
  prov: ProvenanceEntry[] | null;
  selectShelf: (id: string) => Promise<void>;
  closeSelection: () => void;
  recentArrival: string | null;

  // transfer
  xferDest: string;
  setXferDest: (v: string) => void;
  xferQty: number;
  setXferQty: (n: number) => void;
  doTransfer: () => Promise<void>;

  // navigation
  doMove: (to: string) => Promise<void>;
  reset: () => Promise<void>;

  // agent
  busy: boolean;
  answer: AgentResult | null;
  liveSteps: TraceStep[];
  ask: (text: string) => void;
  dismissAnswer: () => void;

  // reference data
  sources: Source[];
  reflections: Belief[];
  newTopic: string;
  setNewTopic: (v: string) => void;
  newLesson: string;
  setNewLesson: (v: string) => void;
  teachReflection: () => Promise<void>;
}

const WarehouseCtx = createContext<Ctx | null>(null);

export function useWarehouse(): Ctx {
  const ctx = useContext(WarehouseCtx);
  if (!ctx) throw new Error("useWarehouse() must be used within <WarehouseProvider>");
  return ctx;
}

export function WarehouseProvider({ children }: { children: ReactNode }) {
  const [view, setView] = useState<WarehouseView | null>(null);
  const [offline, setOffline] = useState(false);
  const [pathMap, setPathMap] = useState(false);

  const [shelfSel, setShelfSel] = useState<Shelf | null>(null);
  const [inv, setInv] = useState<InventoryView | null>(null);
  const [bin, setBin] = useState<ShelfContents | null>(null);
  const [prov, setProv] = useState<ProvenanceEntry[] | null>(null);
  const [recentArrival, setRecentArrival] = useState<string | null>(null);
  const [xferDest, setXferDest] = useState<string>("");
  const [xferQty, setXferQty] = useState<number>(10);

  const [answer, setAnswer] = useState<AgentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveSteps, setLiveSteps] = useState<TraceStep[]>([]);

  const [sources, setSources] = useState<Source[]>([]);
  const [reflections, setReflections] = useState<Belief[]>([]);
  const [newTopic, setNewTopic] = useState("");
  const [newLesson, setNewLesson] = useState("");

  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);

  const toast = useCallback((dot: string, html: string) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, dot, html }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const load = useCallback(async () => {
    try {
      const [v, srcs, refs] = await Promise.all([
        api.getWarehouse(),
        api.sources(),
        api.listReflections(),
      ]);
      setView(v);
      setSources(srcs);
      setReflections(refs);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const refreshReflections = useCallback(async () => {
    try { setReflections(await api.listReflections()); } catch { /* ignore */ }
  }, []);

  const teachReflection = useCallback(async () => {
    const lesson = newLesson.trim();
    if (!lesson) return;
    try {
      const b = await api.addReflection(lesson, newTopic.trim() || "general");
      toast("var(--accent)", `Learned: <b>${b.predicate.split(":").pop()}</b>`);
      setNewLesson("");
      setNewTopic("");
      refreshReflections();
    } catch (e) {
      toast("var(--blocked)", String(e));
    }
  }, [newLesson, newTopic, toast, refreshReflections]);

  const doMove = useCallback(
    async (to: string) => {
      if (!view) return;
      const opt = view.options.find((o) => o.to === to);
      try {
        const next = await api.move(to);
        setView(next);
        if (opt && !opt.passable)
          toast("var(--blocked)", `Overrode agent · <b>${Math.round(opt.confidence * 100)}%</b>`);
        else if (opt)
          toast("var(--clear)", `Moved · <b>${Math.round(opt.confidence * 100)}%</b> confident`);
        for (const ev of next.events_fired ?? []) {
          toast(ev.action === "toast" ? "var(--accent)" : "var(--warn)",
            `⚡ <b>${ev.event_id}</b> · ${ev.message}`);
        }
        if ((next.events_fired ?? []).some((e) => e.action !== "toast") && shelfSel) {
          try {
            const [i, b, pr] = await Promise.all([
              api.inventory(shelfSel.id),
              api.bin(shelfSel.id),
              api.provenance(shelfSel.id),
            ]);
            setInv(i); setBin(b); setProv(pr);
          } catch { /* ignore */ }
        }
        const goal = next.shelves.find((s) => s.id === next.goal);
        if (goal && goal.near === next.position)
          toast("var(--accent)", `Reached <b>${next.goal}</b> ★`);
      } catch (e) {
        toast("var(--blocked)", String(e));
      }
    },
    [view, toast, shelfSel],
  );

  const selectShelf = useCallback(async (id: string) => {
    if (!view) return;
    const sh = view.shelves.find((s) => s.id === id) ?? null;
    setShelfSel(sh);
    setInv(null); setBin(null); setProv(null); setRecentArrival(null);
    const others = view.shelves.filter((s) => s.id !== id);
    setXferDest(others[0]?.id ?? "");
    try {
      const [i, b, pr] = await Promise.all([
        api.inventory(id), api.bin(id), api.provenance(id),
      ]);
      setInv(i); setBin(b); setProv(pr);
    } catch { /* leave null */ }
  }, [view]);

  const closeSelection = useCallback(() => {
    setShelfSel(null);
    setInv(null); setBin(null); setProv(null); setRecentArrival(null);
  }, []);

  const doTransfer = useCallback(async () => {
    if (!shelfSel) return;
    try {
      const res = await api.transfer(shelfSel.id, xferDest, xferQty);
      if (!res.ok) { toast("var(--blocked)", res.message); return; }
      toast("var(--clear)", `Moved <b>${res.quantity} × ${res.sku}</b> → ${res.dest_shelf}`);
      const [v, srcInv, srcBin, destBin, srcProv] = await Promise.all([
        api.getWarehouse(), api.inventory(shelfSel.id),
        api.bin(shelfSel.id), api.bin(xferDest), api.provenance(shelfSel.id),
      ]);
      setView(v); setInv(srcInv); setBin(srcBin); setProv(srcProv);
      setRecentArrival(
        `+${res.quantity} ${res.sku} landed at ${res.dest_shelf} — ${res.dest_shelf} now holds ${destBin.total_units} total units (${destBin.lots.length} lot${destBin.lots.length === 1 ? "" : "s"}).`,
      );
      setTimeout(() => setRecentArrival(null), 6000);
    } catch (e) { toast("var(--blocked)", String(e)); }
  }, [shelfSel, xferDest, xferQty, toast]);

  const ask = useCallback((text: string) => {
    if (!text.trim()) return;
    setBusy(true);
    setAnswer(null);
    setLiveSteps([]);
    api.queryStream(text, {
      onStep: (step) => setLiveSteps((prev) => [...prev, step]),
      onDone: (result) => { setAnswer(result); setBusy(false); },
      onError: (msg) => { toast("var(--blocked)", "Agent error: " + msg); setBusy(false); },
      maxSteps: 6,
    });
  }, [toast]);

  const reset = useCallback(async () => {
    try {
      setView(await api.reset());
      setShelfSel(null); setInv(null); setBin(null); setProv(null);
      setAnswer(null); setLiveSteps([]);
      toast("var(--accent)", "Warehouse reset to start");
    } catch (e) { toast("var(--blocked)", String(e)); }
  }, [toast]);

  const toggleTheme = useCallback(() => {
    const cur = document.documentElement.getAttribute("data-theme");
    const sysDark = window.matchMedia("(prefers-color-scheme:dark)").matches;
    document.documentElement.setAttribute(
      "data-theme",
      cur ? (cur === "dark" ? "light" : "dark") : sysDark ? "light" : "dark",
    );
  }, []);

  const value: Ctx = {
    view, offline, load,
    pathMap, setPathMap, toggleTheme,
    toasts, toast,
    shelfSel, inv, bin, prov, selectShelf, closeSelection, recentArrival,
    xferDest, setXferDest, xferQty, setXferQty, doTransfer,
    doMove, reset,
    busy, answer, liveSteps, ask, dismissAnswer: () => { setAnswer(null); setLiveSteps([]); },
    sources, reflections, newTopic, setNewTopic, newLesson, setNewLesson, teachReflection,
  };

  return <WarehouseCtx.Provider value={value}>{children}</WarehouseCtx.Provider>;
}
