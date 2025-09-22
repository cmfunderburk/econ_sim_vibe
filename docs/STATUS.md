# Project Status

Updated: 2025-09-22 (Visualization HUD + scrub controls + live RPS)

## Test & Quality Summary
- 250/250 tests passing (expanded unit + validation + visualization + replay: geometry sidecar, frame hash, scrub controls, HUD speed)
- Logging schema version: 1.3.0 (additive)
- FinancingMode toggle implemented: PERSONAL (active), TOTAL_WEALTH (placeholder)
- Regression guard ensures default behavior == explicit PERSONAL
- Movement: Greedy step toward marketplace (A* not yet implemented)
- Travel cost model active: w_i = max(0, p·ω_total - κ·d_i)
- Lint baseline clean (flake8 + ruff + black) with `.flake8` policy (max-line-length=100)

## Feature Matrix
| Area | Status | Notes |
|------|--------|-------|
| Equilibrium Solver | Complete | Closed form + robust fallback (tâtonnement) |
| Market Clearing | Complete | Constrained clearing w/ proportional rationing |
| Financing Mode Toggle | Partial | Enum + regression guard; TOTAL_WEALTH semantics pending |
| Travel Costs | Integrated | Budget-side deduction implemented |
| Movement | Basic | Greedy Manhattan; A* planned |
| Logging | Enhanced | Schema 1.3.0 + spatial aggregates + compression |
| Replay / Geometry | Added | Sidecar + deterministic + fallback inference |
| Visualization | MVP | pygame + ASCII HUD, solver diagnostics, replay scrub |
| Pathfinding (A*) | Not Implemented | Pending priority decision |

## Recent Milestones
- Added spatial aggregate metrics: `max_distance_to_market`, `avg_distance_to_market` (schema 1.1.0)
- Introduced geometry sidecar spec enabling deterministic spatial replay
- Added compression support (gzip) for JSONL & Parquet logging
- Enriched per-agent diagnostics (requested/executed orders, fill rates, unmet components)
- Strengthened invariant checks (movement monotonicity, randomized conservation fuzz test, value feasibility proxy)
- Implemented equilibrium solver fallback (adaptive tâtonnement) with non-worsening residual guarantee
- Lint & style cleanup establishing stable baseline (flake8 + ruff + black)
- FinancingMode regression guard retained (PERSONAL default equivalence)
- Convergence index utility & provider baseline capture (HUD-ready)
- Frame hashing utility & provider frame digest helper (foundation for replay integrity tests)
- Replay scrub controls (play/pause, ±1, ±10, home/end) with HUD solver metrics (2025-09-22)
- Live simulation RPS control & HUD speed indicator (2025-09-22)

## Near-Term Priorities
1. Define & implement TOTAL_WEALTH financing semantics (wealth-backed order sizing)
2. Pathfinding decision: implement A* or explicitly document greedy as canonical for Phase 2 baseline
3. Recording pipeline: video export + manifest hashing (post HUD delivery)
4. Scenario preset library with annotated pedagogy notes
5. Performance benchmarking harness + CI profiling gate
6. Replay regression harness using per-frame digest sequence (golden log test)

## Economic Invariants (Current Enforcement)
- Numéraire: p[0] = 1.0
- Convergence: ||Z_rest||_∞ < 1e-8
- Per-agent value feasibility (PERSONAL): p·buys ≤ p·sells + ε
- Conservation per good: total buys == total sells
- No negative inventories produced
- Spatial monotonicity: Manhattan distance to marketplace weakly decreases after move (unless already 0)
- Liquidity gap diagnostics: z_market - executed_net ≥ 0 where agent is sell-constrained (logged)

## Known Gaps / Open Questions
- TOTAL_WEALTH semantics not yet defined (requires revised feasibility invariant)
- A* path optimality unverified (implementation pending)
- Visualization HUD lacks video export + agent inspector
- No dynamic congestion model (throughput caps off by default)
- No persistent scenario metadata catalog (ad-hoc YAML only)

## Changelog (Recent)
- 2025-09-21: Spatial logging upgrade (schema 1.1.0), aggregates, compression, diagnostics, invariants, solver fallback, test count 232
- 2025-09-21: Earlier - FinancingMode regression guard & documentation consolidation
