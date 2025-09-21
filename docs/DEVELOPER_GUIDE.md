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

## Logging (Planned)
Target schema (per agent per round): prices, z_market, executed_net, liquidity_gap, position, distance, utility, travel_cost.

## Contributing Workflow
1. Branch from main
2. Implement feature + tests
3. Run full test suite
4. Update docs only where canonical (STATUS / ROADMAP / SPECIFICATION)
5. Open PR with concise economic rationale

## Future Extensions (Pointers)
See `docs/ROADMAP.md` for prioritized backlog & decision log.
