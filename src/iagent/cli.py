"""CLI entrypoint. Placeholder until Phase 5.

Intended usage once implemented:
    python -m iagent --scenario a
    python -m iagent --scenario b
    python -m iagent --query "Where am I?"
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iagent", description="I, Agent CLI")
    parser.add_argument("--scenario", choices=["a", "b"], help="Run a canned scenario")
    parser.add_argument("--query", help="Run an ad-hoc natural-language query")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    # EMPTY PLACEHOLDER — IMPLEMENTATION LATER (Phase 5, Person 4).
    print("iagent CLI: not implemented yet. See PROJECT_PLAN.md Phase 5.", file=sys.stderr)
    print(f"parsed args: {vars(args)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
