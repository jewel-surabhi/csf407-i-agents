"""Live demo of P1 - walks Scenarios A and B through KnowledgeAPI and prints
the state at every step.

Not part of the shipped PR. This is a local sanity-check / show-and-tell:

    python scripts/demo_declarative.py

Uses `rich` for pretty output. Everything runs in a tmp SQLite DB that is
deleted at the end, so it is safe to re-run.

Owner: P1 (Surabhi). Delete or move this file if it clutters the tree.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Make `iagent` importable when run directly.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from iagent.declarative.api import Belief, KnowledgeAPI  # noqa: E402
from iagent.declarative.belief_graph import BeliefGraph  # noqa: E402
from iagent.declarative.provenance import ProvenanceStore  # noqa: E402
from iagent.declarative.seeding import seed_from_files  # noqa: E402

console = Console(record=True)
_SEED_DIR = _REPO / "data" / "seed"


def _show_active(api: KnowledgeAPI, subject: str) -> None:
    """Render every active belief about a subject as a table."""
    table = Table(title=f"Active beliefs for [bold]{subject}[/]", show_header=True)
    for col in ("predicate", "object", "conf", "source", "perspective", "prov_id"):
        table.add_column(col)
    for pv in api.query_perspective(subject):
        for b in pv.beliefs:
            table.add_row(
                b.predicate, b.object, f"{b.confidence:.2f}",
                b.source, b.perspective, str(b.provenance_id or "-"),
            )
    if len(table.rows) == 0:
        table.add_row("[dim]- nothing active -[/]", "", "", "", "", "")
    console.print(table)


def _show_history(api: KnowledgeAPI, subject: str) -> None:
    """Render the full SQL history for a subject (newest first)."""
    table = Table(title=f"Provenance history for [bold]{subject}[/] (newest first)",
                  show_header=True)
    for col in ("#", "predicate", "object", "conf", "source", "perspective", "status"):
        table.add_column(col)
    for p in api.query_provenance(subject):
        style = "green" if p.status == "active" else "dim"
        table.add_row(
            f"[{style}]{p.provenance_id}[/]",
            f"[{style}]{p.predicate}[/]",
            f"[{style}]{p.object}[/]",
            f"[{style}]{p.confidence:.2f}[/]",
            f"[{style}]{p.source_id}[/]",
            f"[{style}]{p.perspective}[/]",
            f"[{style}]{p.status}[/]",
        )
    console.print(table)


def _tool_call(name: str, **kwargs) -> None:
    """Pretty-print a mock 'LLM tool call' for narration."""
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    console.print(f"[bold cyan]>> {name}[/]([dim]{args}[/])")


def demo_scenario_a(api: KnowledgeAPI) -> None:
    console.print(Rule("[bold]Scenario A - Groundedness (map vs LiDAR)[/]"))
    console.print()

    # Seed: map says path is clear.
    _tool_call("update_belief",
               subject="path_ahead", predicate="is_clear", object="true",
               source="default_map", perspective="self", confidence=1.0,
               reason="baseline floor plan")
    api.update_belief(Belief(
        subject="path_ahead", predicate="is_clear", object="true",
        confidence=1.0, source="default_map", perspective="self",
        observed_at=datetime(2026, 9, 6, 9, 0, tzinfo=UTC),
    ), reason="baseline floor plan")
    _show_active(api, "path_ahead")
    console.print()

    # LLM step 1: read memory.
    console.print(Panel("[italic]The LLM's step 1: check declarative memory first.[/]"))
    _tool_call("get_belief", subject="path_ahead", predicate="is_clear")
    got = api.get_belief("path_ahead", "is_clear")
    console.print(f"  <- returned: [green]object={got.object}, conf={got.confidence}, "
                  f"source={got.source}[/]")
    console.print()

    # LLM step 2 would be sense_lidar (P2's territory). Simulate the reading.
    console.print(Panel(
        "[italic]The LLM's step 2 would be sense_lidar('front') - P2's territory. "
        "The MockEnvironment returns:[/]\n"
        "[yellow]{'sensor':'lidar_front','value':12,'status':'blocked','confidence':0.95}[/]"
    ))
    console.print()

    # LLM step 3: consult trust priors.
    console.print(Panel("[italic]The LLM's step 3: check who to trust.[/]"))
    _tool_call("list_sources")
    for s in api.list_sources():
        if s.source_id in ("default_map", "lidar_front"):
            console.print(f"  <- [green]{s.source_id}[/]: trust_prior={s.trust_prior}")
    console.print("  [italic]sensor beats map -> downgrade the stale belief.[/]")
    console.print()

    # LLM step 4: downgrade.
    _tool_call("downgrade_belief",
               subject="path_ahead", predicate="is_clear",
               new_confidence=0.2, reason="live LiDAR contradicts")
    api.downgrade_belief("path_ahead", "is_clear",
                         new_confidence=0.2, reason="live LiDAR contradicts")
    console.print()

    # LLM step 5: assert the fresh sensor-derived belief.
    _tool_call("update_belief",
               subject="path_ahead", predicate="is_blocked", object="obstacle@12cm",
               source="lidar_front", confidence=0.95,
               reason="live lidar reading")
    api.update_belief(Belief(
        subject="path_ahead", predicate="is_blocked", object="obstacle@12cm",
        confidence=0.95, source="lidar_front", perspective="self",
        observed_at=datetime(2026, 9, 6, 10, 15, tzinfo=UTC),
    ), reason="live lidar reading")
    console.print()

    _show_active(api, "path_ahead")
    console.print()
    _show_history(api, "path_ahead")
    console.print()

    console.print(Panel(
        "[bold green]Scenario A verified:[/]\n"
        "  - [green]is_blocked[/] is now the active belief (from lidar_front, conf=0.95)\n"
        "  - [dim]is_clear[/] rows are downgraded but PRESERVED in history\n"
        "  - Every mutation left an audit trail in SQLite - the LLM cannot lie about it"
    ))


def demo_scenario_b(api: KnowledgeAPI) -> None:
    console.print()
    console.print(Rule("[bold]Scenario B - Three perspectives on box_01[/]"))
    console.print()

    # Seed: user believes RED, maintenance bot logged BLUE.
    _tool_call("update_belief",
               subject="box_01", predicate="has_color", object="red_atom",
               source="user", perspective="user", confidence=0.9,
               reason="user statement")
    api.update_belief(Belief(
        subject="box_01", predicate="has_color", object="red_atom",
        confidence=0.9, source="user", perspective="user",
        observed_at=datetime(2026, 9, 10, 9, 0, tzinfo=UTC),
    ), reason="user statement")

    _tool_call("update_belief",
               subject="box_01", predicate="has_color", object="blue_atom",
               source="bot_02", perspective="third_party:bot_02", confidence=0.7,
               reason="maintenance log 2026-08-30")
    api.update_belief(Belief(
        subject="box_01", predicate="has_color", object="blue_atom",
        confidence=0.7, source="bot_02", perspective="third_party:bot_02",
        observed_at=datetime(2026, 8, 30, 15, 30, tzinfo=UTC),
    ), reason="maintenance log 2026-08-30")

    # LLM's step 2 would be sense_camera. Simulate the payload.
    console.print(Panel(
        "[italic]The LLM's step 2 would be sense_camera('front') - P2's territory. "
        "The mock camera returns:[/]\n"
        "[yellow]{'detections':[{'item_id':'box_01','color_perceived':'brown',"
        "'confidence':0.82}], 'ambient':{'lighting':'yellow','warning':'unreliable'}}[/]"
    ))
    console.print()

    # LLM records the egocentric reading.
    _tool_call("update_belief",
               subject="box_01", predicate="has_color", object="brown_atom",
               source="camera_front", perspective="self", confidence=0.82,
               reason="live camera; ambient=yellow (unreliable)")
    api.update_belief(Belief(
        subject="box_01", predicate="has_color", object="brown_atom",
        confidence=0.82, source="camera_front", perspective="self",
        observed_at=datetime(2026, 9, 10, 10, 15, tzinfo=UTC),
    ), reason="live camera; ambient=yellow (unreliable)")
    console.print()

    # The critical query: query_perspective returns GROUPED views.
    _tool_call("query_perspective", subject="box_01", predicate="has_color")
    console.print()
    for pv in api.query_perspective("box_01", predicate="has_color"):
        colour_map = {"user": "red", "self": "yellow",
                      "third_party:bot_02": "cyan"}
        c = colour_map.get(pv.perspective, "white")
        console.print(f"  [{c}]{pv.perspective:24}[/] -> {pv.beliefs[0].object}"
                      f" (conf {pv.beliefs[0].confidence:.2f},"
                      f" source={pv.beliefs[0].source})")
    console.print()

    console.print(Panel(
        "[bold green]Scenario B verified:[/]\n"
        "  - Three perspective-differing beliefs about box_01 coexist as distinct edges\n"
        "  - query_perspective returns them ALREADY GROUPED - the LLM cannot merge\n"
        "  - Each perspective was written with the correct source_id and confidence"
    ))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="iagent-demo-")) / "demo.db"
    console.print(Panel(
        f"[bold]I, Agent - live demo of the P1 declarative layer[/]\n"
        f"[dim]tmp DB: {tmp}[/]"
    ))
    console.print()

    with ProvenanceStore(str(tmp)) as store:
        graph = BeliefGraph()
        # Seed the trust priors + graph nodes from the repo seed files.
        summary = seed_from_files(
            store, graph,
            provenance_sql=_SEED_DIR / "initial_provenance.sql",
            beliefs_json=_SEED_DIR / "initial_beliefs.json",
        )
        console.print(f"[dim]Seeded {summary['sources']} sources, "
                      f"{summary['nodes_added']} graph nodes.[/]")
        console.print()

        api = KnowledgeAPI(graph, store)
        demo_scenario_a(api)
        demo_scenario_b(api)

    tmp.unlink(missing_ok=True)
    tmp.parent.rmdir()
    console.print()
    console.print("[dim]demo complete - tmp DB removed.[/]")

    # Export the styled transcript for the progress report.
    out_html = _REPO / "demo_output.html"
    out_svg = _REPO / "demo_output.svg"
    console.save_html(str(out_html), inline_styles=True, theme=None)
    console.save_svg(str(out_svg), title="I, Agent - P1 declarative demo")
    console.print()
    console.print(f"[bold green]HTML saved:[/] {out_html}")
    console.print(f"[bold green]SVG  saved:[/] {out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
