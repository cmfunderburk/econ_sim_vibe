# Roadmap

Updated: 2025-09-21

## Phase Overview
| Phase | Goal | Status |
|-------|------|--------|
| 1 | Pure Exchange Walrasian Engine | Complete |
| 2 | Spatial Extension (Global LTE + Movement Costs) | In Progress |
| 3 | Local Price Formation & Microstructure | Planned |
| 4 | Production, Money, Institutions | Future |

## Current Focus (Phase 2)
- Maintain economic invariants with spatial frictions
- Measure welfare loss from access & travel cost
- Incremental infrastructure (logging, optional viz)

## Priority Backlog (Ranked)
1. TOTAL_WEALTH financing semantics (design + implementation + tests)
2. Structured Parquet logging (round-level schema: prices, orders, executed, liquidity gap)
3. Optional: A* pathfinding OR formalize greedy as canonical (update spec)
4. Performance profiling & lightweight vectorization pass
5. Visualization prototype (pygame or textual grid) for teaching
6. Extended welfare analytics (EV decomposition, liquidity wedge metrics)
7. Capacity / congestion regime toggle (guard economics with test markers)

## Stretch / Research Items
- Bilateral bargaining module
- Continuous double auction microstructure
- Spatially varying prices & arbitrage agents
- Production & factor markets (Cobb-Douglas firm layer)
- Credit / monetary extensions (beyond barter)

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-09-21 | Keep TOTAL_WEALTH placeholder until semantics defined | Avoid premature complexity |
| 2025-09-21 | Consolidated docs | Reduce drift & duplication risk |

## Acceptance Criteria Examples
- Financing TOTAL_WEALTH: p·buys ≤ p·ω_total + ε (still per-good sell caps) with new tests verifying liquidity gap shrinkage
- Logging: After 1 simulation run, Parquet file contains rows = agents × rounds with invariant columns present, schema version pinned
- A* Pathfinding (if chosen): Path cost equals Manhattan distance; test verifies no longer path than greedy for empty grid

## Risk Register
| Risk | Impact | Mitigation |
|------|--------|------------|
| Feature drift in multiple docs | Medium | Single-source STATUS + pointers |
| Over-specified placeholder mode | Low | Gate behind explicit flag & tests |
| Performance regressions with logging | Medium | Benchmark before/after & batch writes |

## Sunset / Decommission Notes
- Legacy roadmap & step guides replaced by this file on 2025-09-21
