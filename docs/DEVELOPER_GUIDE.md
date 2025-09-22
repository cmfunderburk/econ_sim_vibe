# Developer Guide

Updated: 2025-09-21

## Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Test Suite
- 197 tests total: 185 unit + 12 validation
- Markers (examples): economic_core, validation, robustness, real_functions
- Run subsets:
```bash
pytest tests/unit -q
pytest tests/validation -q
pytest -m validation
```

## Core Commands
```bash
make test
make validate
make format
make check
```

## Key Modules
| Path | Purpose |
|------|---------|
| `src/core/agent.py` | Agent definition & utility/demand logic |
| `src/econ/equilibrium.py` | Walrasian price solver |
| `src/econ/market.py` | Order generation & constrained clearing |
| `src/spatial/grid.py` | Grid & movement (greedy) |

## Financing Modes
Current: `FinancingMode.PERSONAL` active; `TOTAL_WEALTH` placeholder.
Regression test ensures default call == explicit PERSONAL mode.

## Movement
Greedy Manhattan step toward marketplace; A* not yet implemented.

### Synchronized Return Behavior (2025-09-21)
After a clearing round completes we currently defer transitioning any agent to the `TO_HOME` phase until all agents en‑route to the marketplace have arrived (i.e. every agent in `TO_MARKET` has reached an in‑marketplace cell and is in `AT_MARKET`). This preserves a pedagogical invariant used in tests: Manhattan distance to the nearest marketplace cell is non‑increasing for agents until the cohort is fully assembled. Without this synchronization early arrivals would immediately start homeward travel, temporarily increasing their distance and failing the monotonic distance regression test. A future optional configuration (e.g. `stagger_returns = true`) may relax this for more realistic staggered departures; enabling it will require parameterizing or adapting the invariant test accordingly.

## Travel Costs
Budget adjusted: `w_i = max(0, p·ω_total - κ·d_i)`.

## Invariants Checked
- Numéraire: `p[0] = 1`
- Convergence: `||Z_rest||_∞ < 1e-8`
- Value feasibility (PERSONAL): `p·buys ≤ p·sells + ε`
- Per-good conservation after execution

## Adding Tests
1. Create file under `tests/unit/` or extend validation.
2. Use clear docstring referencing economic property.
3. If adding new mode/feature: add regression test for backward compatibility.

## Common Pitfalls
| Issue | Avoidance |
|-------|-----------|
| Using personal instead of total endowment in theoretical demand | Always compute wealth with total endowment for LTE |
| Relying on Walras' Law for convergence | Use rest-goods infinity norm |
| Forgetting value feasibility scaling | Ensure scaling branch in PERSONAL mode |

## Style & Quality
- Formatting: `black`, `ruff` (via `make format`)
- Types: `mypy src/`
- Lint: `flake8`, `ruff check`

## Structured Logging
Implemented per-agent, per-round structured log (schema version `1.1.0`). Version 1.1.0 added enriched order diagnostics (requested, executed, unmet, fill rates) as additive columns.

File formats:
- Parquet (`*_round_log.parquet`) if `pandas` + parquet engine available
- Fallback JSON Lines (`*_round_log.jsonl`) otherwise
- Metadata sidecar: `*_metadata.json` with `schema_version`, row count, format

Row = `RoundLogRecord` fields:
| Field | Description |
|-------|-------------|
| `core_schema_version` | Schema version string (bump on breaking changes) |
| `core_round` | 1-based round index after completion of round logic |
| `core_agent_id` | Agent identifier |
| `spatial_pos_x`, `spatial_pos_y` | Agent position at end of round |
| `spatial_in_marketplace` | True if agent currently inside marketplace bounds |
| `econ_prices` | Price vector used this round (empty list if no pricing occurred) |
| `econ_executed_net` | Net executed trade quantities (buys minus sells) per good |
| `ration_unmet_demand` | Per-good unmet demand (None when not computed) |
| `ration_unmet_supply` | Per-good unmet supply (None when not computed) |
| `wealth_travel_cost` | Cumulative travel cost (numéraire units) |
| `wealth_effective_budget` | Adjusted wealth after travel cost (future) |
| `financing_mode` | Financing mode tag (future: `personal` / `total_wealth`) |
| `utility` | Cobb-Douglas utility of agent's total endowment snapshot |
| `timestamp_ns` | Monotonic-ish wall clock capture for ordering |

Current limitations / notes:
- Rationing aggregates now populated when clearing occurs; remain `null` on non-clearing rounds.
- Utility included (baseline total endowment); EV trajectory still pending.
- Financing mode placeholder remains `null` until multi-mode integration.

Usage: pass `--output results/` to `run_simulation.py` to auto-generate structured log.

## Contributing Workflow
1. Branch from main
2. Implement feature + tests
3. Run full test suite
4. Update docs only where canonical (STATUS / ROADMAP / SPECIFICATION)
5. Open PR with concise economic rationale

## Future Extensions (Pointers)
See `docs/ROADMAP.md` for prioritized backlog & decision log.
