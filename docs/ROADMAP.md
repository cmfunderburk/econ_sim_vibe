# Roadmap

Updated: 2025-09-21 (Visualization Track Integration)

## Phase Overview
| Phase | Goal | Status |
|-------|------|--------|
| 1 | Pure Exchange Walrasian Engine | Complete |
| 2 | Spatial Extension (Global LTE + Movement Costs) | In Progress |
| 3 | Local Price Formation & Microstructure | Planned |
| 4 | Production, Money, Institutions | Future |

## Current Focus (Phase 2)
Primary near-term objective: Deliver a classroom-ready educational visualization and replay workflow without regressing core economic correctness.

Focus pillars:
1. Observability: Rich per-agent trade diagnostics (schema v1.1.0) surfaced in real-time overlays.
2. Replayability: Deterministic log → frame reconstruction (no recomputation required for lecture playback).
3. Pedagogy: Highlight spatial access, rationing, liquidity gaps, convergence, welfare effects.
4. Stability: Preserve existing invariants (rest-goods convergence, value feasibility, conservation) while adding interface features.

Deferral principle: Expansion of core economic regimes (TOTAL_WEALTH execution semantics, local price formation) resumes after visualization MVP is validated for teaching impact.

## Priority Backlog (Ranked – Visualization Track Emphasis)
1. Visualization Data Plumbing (frame enrichment: requested/executed/unmet/fill, travel cost, utility, convergence metrics)
2. HUD & Overlay Layer (prices, avg fill rate, unmet share, participants, solver residual)
3. Replay Loader & Controls (`--replay`, pause/play, step, scrub, speed)
4. Scenario Preset Library (curated YAMLs + pedagogical annotations)
5. Snapshot & Export (PNG capture; round summary CSV/Parquet)
6. Pathfinding Decision: Implement A* (preferred) OR explicitly canonize greedy in docs/tests
7. Performance Profiling & Frame Diff Optimization
8. Interactive Agent Inspector (click/hover: orders, fills, utility, liquidity gap)
9. Live Charts (aggregate fill rate, distance convergence, total travel cost)
10. Financing Mode Groundwork (UI hooks to contrast PERSONAL vs TOTAL_WEALTH later)
11. Extended Welfare Analytics (equivalent variation decomposition overlays)
12. Capacity / Congestion Regime Toggle (visual queue pressure indicators)
13. TOTAL_WEALTH Financing Semantics (post-viz MVP implementation)
14. Local Price Formation (Phase 3 entry point – deferred)

## Stretch / Research Items (Post Visualization MVP)
- Bilateral bargaining module
- Continuous double auction microstructure
- Spatially varying prices & arbitrage agents
- Production & factor markets (Cobb-Douglas firm layer)
- Credit / monetary extensions (beyond barter)
- Real-time EV trajectory embedding
- Multi-market spatial arbitrage agents

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-09-21 | Keep TOTAL_WEALTH placeholder until semantics defined | Avoid premature complexity |
| 2025-09-21 | Consolidated docs | Reduce drift & duplication risk |

## Acceptance Criteria (Representative)
Visualization Data Plumbing
- Frame object includes per-agent: requested_buys, executed_buys, unmet_buys, fill_rate_buys (and analogous sell arrays), utility, travel_cost, distance_to_market.
- Aggregates: avg_buy_fill_rate, max_distance, total_unmet_buy_share, solver_rest_goods_norm.

HUD & Overlay
- Prices rendered each round (p[0] ≡ 1 annotated).
- Displays: Round number, participants (k / N), avg fill rate (≥0, ≤1), unmet % (≥0), rest-good norm < tolerance when converged.

Replay Loader
- `econ-sim --replay path.parquet` reproduces identical sequence of frames (hash of serialized per-round core fields matches live run).
- Supports seeking to arbitrary round ≤ T in O(log T) or better (index precomputed).

Scenario Presets
- Each preset YAML contains comment block: learning objective, expected phenomenon, key metrics.
- Edgeworth 2×2 preset validated: computed prices within 1e-8 of analytic; classroom annotation visible.

Snapshot & Export
- Pressing 'S' produces PNG (non-zero size) with timestamped filename.
- Optional round summary export includes columns: round, avg_fill_rate, unmet_buy_share, total_travel_cost.

Pathfinding (if A*)
- Test: A* path length == Manhattan distance on empty grid for ≥10 random start positions.
- Performance: Path computation amortized ≤ O(1) per step via cached distance field or heuristic (A* expansions < 4×distance).

Interactive Inspector (Optional Milestone)
- Clicking an agent highlights cell; side panel shows last round order vectors and fill rates (numbers sum within FEASIBILITY_TOL of requested − unmet).

No Regression Invariants (Global)
- Existing 217 tests (or expanded suite) all pass after visualization features enabled or disabled.
- Logging schema version remains stable (1.1.0) unless additional additive fields introduced with minor bump.

TOTAL_WEALTH (Deferred)
- Activation flag introduces new tests verifying liquidity gap shrinkage; financing logic isolated behind strategy interface.

## Risk Register
| Risk | Impact | Mitigation |
|------|--------|------------|
| Frame rendering performance degradation | Medium | Early profiling, diff-based redraw, throttle control |
| Visualization obscures economic invariants | Medium | Always display convergence & fill metrics; add regression tests |
| Replay divergence (serialization drift) | High | Hash-based golden log tests; strict schema guard |
| Pathfinding complexity vs benefit | Low | Keep A* minimal; fallback to greedy with doc clarity if issues arise |
| UI feature creep delaying pedagogy | Medium | Enforce phased milestones; freeze new features after MVP until evaluation |
| Logging schema churn | Low | Additive changes only; versioned, guard-tested |

## Milestones & Target Timeline (Indicative)
| Milestone | Scope | Target |
|-----------|-------|--------|
| M1: Data & HUD | Frame enrichment + basic HUD + avg fill metric | Week 1 |
| M2: Replay Core | Log loader + pause/play/step/scrub | Week 2 |
| M3: Scenarios & Export | Preset YAMLs + PNG snapshot + summary CSV | Week 3 |
| M4: Inspector & Charts | Agent inspector + simple live charts | Week 4 |
| M5: Performance Hardening | Profiling + diff redraw + pathfinding decision | Week 5 |
| M6: Pedagogical Release | Educator guide + notebook + evaluation run | Week 6 |

## Sunset / Decommission Notes
- Legacy roadmap & step guides replaced by this visualization-integrated roadmap on 2025-09-21.
- Pre-visualization backlog items deferred explicitly are reintroduced only after M3 validation unless they unblock pedagogy.
