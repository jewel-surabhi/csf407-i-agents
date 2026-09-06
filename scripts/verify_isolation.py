"""Static layer-isolation check. Run in CI on every push.

Fails if:
  * `src/iagent/procedural/**` imports `networkx`, `sqlite3`, or reaches into
    `iagent.declarative.<internals>` / `iagent.sensorimotor.<internals>`.
    Only `iagent.declarative.api` and `iagent.sensorimotor.api` are allowed.
  * `src/iagent/declarative/**` imports anything under `iagent.sensorimotor` or
    `iagent.procedural`.
  * `src/iagent/sensorimotor/**` imports anything under `iagent.declarative` or
    `iagent.procedural`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "iagent"

# (layer_dir, forbidden_regex, allowed_exceptions_regex_or_None)
RULES: list[tuple[str, str, str | None]] = [
    # Procedural must not import DB/graph internals or other layers' internals.
    ("procedural", r"^\s*(?:from|import)\s+networkx\b", None),
    ("procedural", r"^\s*(?:from|import)\s+sqlite3\b", None),
    (
        "procedural",
        r"^\s*from\s+iagent\.declarative(?:\.[a-zA-Z_]+)?\s+import",
        r"^\s*from\s+iagent\.declarative\.(?:api|models)\s+import",
    ),
    (
        "procedural",
        r"^\s*from\s+iagent\.sensorimotor(?:\.[a-zA-Z_]+)?\s+import",
        r"^\s*from\s+iagent\.sensorimotor\.(?:api|models)\s+import",
    ),
    # Declarative must not know about the other layers.
    ("declarative", r"^\s*from\s+iagent\.sensorimotor", None),
    ("declarative", r"^\s*from\s+iagent\.procedural", None),
    # Sensorimotor must not know about the other layers.
    ("sensorimotor", r"^\s*from\s+iagent\.declarative", None),
    ("sensorimotor", r"^\s*from\s+iagent\.procedural", None),
]


def check() -> list[str]:
    violations: list[str] = []
    for layer, forbidden_pat, allowed_pat in RULES:
        layer_dir = SRC / layer
        if not layer_dir.exists():
            continue
        forbidden = re.compile(forbidden_pat)
        allowed = re.compile(allowed_pat) if allowed_pat else None
        for py in layer_dir.rglob("*.py"):
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.search(line) and not (allowed and allowed.search(line)):
                    rel = py.relative_to(REPO).as_posix()
                    violations.append(f"{rel}:{lineno}: {line.strip()}  (rule: /{forbidden_pat}/)")
    return violations


def main() -> int:
    violations = check()
    if violations:
        print("Layer isolation VIOLATIONS:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nFix: route the call through the layer's api.py facade.",
            file=sys.stderr,
        )
        return 1
    print(f"Layer isolation OK ({sum(1 for _ in SRC.rglob('*.py'))} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
