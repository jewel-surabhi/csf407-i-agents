## Summary
<!-- One or two sentences: what changed and why. -->

## Layer(s) affected
- [ ] Declarative (`src/iagent/declarative/`)
- [ ] Sensorimotor (`src/iagent/sensorimotor/`)
- [ ] Procedural (`src/iagent/procedural/`)
- [ ] Integration / CLI / tests / docs

## Checklist
- [ ] Only modified files inside my `CODEOWNERS` scope (or requested owner review for others).
- [ ] Added / updated unit tests for changed source code.
- [ ] `ruff check .` passes locally.
- [ ] `pytest -q` passes locally.
- [ ] `python scripts/verify_isolation.py` passes locally.
- [ ] If cross-layer JSON changed, `docs/data_contracts.md` updated in this PR.
- [ ] If a scenario changed, attached the new trace JSON from `data/runs/`.

## Related issue / phase
<!-- e.g. Phase 3 of PROJECT_PLAN.md, closes #12 -->
