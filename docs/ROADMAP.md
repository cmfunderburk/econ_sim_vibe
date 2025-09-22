# Roadmap

Updated: 2025-09-21 (Visualization Track Integration – Spatial Fidelity Step D Complete)

## Phase Overview
| Phase | Goal | Status |
|-------|------|--------|
| 1 | Pure Exchange Walrasian Engine | Complete |
| 2 | Spatial Extension (Global LTE + Movement Costs) | In Progress |
| 3 | Local Price Formation & Microstructure | Planned |
| 4 | Production, Money, Institutions | Future |

## Recently Completed (Visualization Track)
Foundational visualization & observability layers now in place:
1. Movement Abstraction: `MovementPolicy` + `GreedyManhattanPolicy` (future A* ready).
2. Playback Core: `PlaybackController` (play/pause/step/speed) integrated live.
3. Frame Streams: `LiveSimulationStream` + internal `LogReplayStream` (CLI pending).
4. HUD / Overlay: Prices, participants, fill & unmet metrics, controller state.
5. Logging Deduplication: Unified helper for per-round record assembly (schema v1.1.0 intact).
6. Snapshot Infrastructure: JSON + optional PNG (`--snapshot-dir`, `--snapshot-every`).
7. Enriched Frame Data: Per-agent requested/executed/unmet/fill, travel cost, max distance.
8. Live RPS playback control (speed adjustments + HUD speed indicator) integrated into live simulation loop.

Partial / Open Gaps:
- Mouse/timeline scrub UI (keyboard seek delivered; GUI slider pending).
- Marketplace geometry obstacles & historical parity validation (fallback inference live; obstacle layers future).
- No video/animation export (ffmpeg) yet.
- A* pathfinding deferred; greedy documented as canonical interim policy.
- Per-frame digest not yet logged (hash utility + provider digest ready).
Completed from prior gap list:
- Public replay CLI (`--replay <log>`) headless + GUI.
- Seek/scrub backend (random access indexing, frame_at, prev_frame).
- Round summary aggregate export (CSV) with prices, participation, execution aggregates.
- Spatial agent position reconstruction (basic fidelity; geometry still pending).

## Current Focus (Phase 2)
Primary near-term objective: Deliver a classroom-ready educational visualization and replay workflow without regressing core economic correctness.

Focus pillars:
1. Observability: Rich per-agent trade diagnostics (schema v1.1.0) surfaced in real-time overlays.
2. Replayability: Deterministic log → frame reconstruction (no recomputation required for lecture playback).
3. Pedagogy: Highlight spatial access, rationing, liquidity gaps, convergence, welfare effects.
4. Stability: Preserve existing invariants (rest-goods convergence, value feasibility, conservation) while adding interface features.

Deferral principle: Expansion of core economic regimes (TOTAL_WEALTH execution semantics, local price formation) resumes after visualization MVP is validated for teaching impact.

## Priority Backlog (Ranked – Post Step D Spatial Fidelity)
1. Video / Animation Export: PNG sequence + ffmpeg MP4 (manifest + integrity digest extension + geometry hash inclusion).
2. Scenario Preset Library: Curated YAMLs with annotated pedagogical objectives.
3. Live Charts & Historical Metrics: Rolling convergence & participation trends (leveraging new distance columns).
4. Performance Profiling & Diff Redraw: Benchmark renderer; implement dirty-rect / diff HUD updates.
5. Interactive Agent Inspector: Click/hover details (orders, fills, utility, liquidity gap).
6. Live Charts Overlay: Rolling plots (fill, unmet share, travel cost, rest-goods norm, convergence index history).
7. Financing Mode Groundwork: UI plumbing for future TOTAL_WEALTH comparison.
8. Extended Welfare Analytics: EV decomposition overlays & per-agent sparkline.
9. Pathfinding Evaluation: Minimal A* or formalize greedy with explicit tests.
10. Capacity / Congestion Toggle: Visual throughput & queue metrics (defer unless needed).
11. TOTAL_WEALTH Financing Semantics: Strategy activation & invariant tests.
12. Local Price Formation (Phase 3 entry point – deferred).

Completed Backlog Items Removed:

Ordering Rationale: Keyboard scrub + solver HUD shipped (Step E), so export & scenario breadth (1–2) now lead, followed by time-series overlays and performance hardening before deeper analytics (5–12 retained, renumbered).

- HUD Integration (2025-09-21): Added convergence index (avg/initial max distance), per-frame digest snippet, max/avg distance display, and unified rendering across Pygame + ASCII. Replay pipeline reconstructs HUD fields (convergence baseline approximated from first round) and tests cover both live and replay availability.
- Scrub Controls + Solver HUD (2025-09-22): Added ±1/±10/home/end keyboard seek, paused scrubbing without tick delay, solver residual/status + unmet ratios rendered in HUD with alert colouring, and geometry fallback for legacy logs.
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
- STATUS: `--replay` CLI (headless + GUI), minimal run-level integrity digest (`*_integrity.json`), random access (frame_at, prev_frame), spatial agent positions reconstructed.
- NEXT: Per-round frame hash verification (optional), GUI backward/seek bindings, distance & geometry reconstruction.
- FUTURE: Full spatial parity (market bounds, distances, potential path traces) & hash of extended frame signature.

Scenario Presets
- Each preset YAML contains comment block: learning objective, expected phenomenon, key metrics.
- Edgeworth 2×2 preset validated: computed prices within 1e-8 of analytic; classroom annotation visible.

Snapshot & Export
- STATUS: PNG + JSON snapshot operational; round summary CSV export implemented (prices, participation, execution & fill aggregates).
- NEXT: Add solver residuals & distance metrics to summary (once replay geometry ready); optional Parquet path.
- FUTURE: MP4 video export (constant frame rate) + integrity manifest (extended hashes) for reproducibility.

Pathfinding (Decision Pending)
- If implemented: A* path length == Manhattan distance on empty grid (≥10 random starts) & expansions < 4×distance.
- Else: Greedy documented as canonical with explicit disclaimer tests.

Interactive Inspector (Optional Milestone)
- Clicking an agent highlights cell; side panel shows last round order vectors and fill rates (numbers sum within FEASIBILITY_TOL of requested − unmet).

No Regression Invariants (Global)
- Existing 217 tests (or expanded suite) all pass after visualization features enabled or disabled.
- Logging schema version advanced to 1.3.0 (1.2.0 introduced per-frame hash; 1.3.0 adds spatial fidelity columns). Future additive fields continue minor bumps; parity & tamper tests enforce spatial invariants.

## Step D (Spatial Geometry & Distance Fidelity) – Completion Summary
Status: COMPLETE (Schema 1.3.0)

Additions:
- New per-row columns: spatial_distance_to_market, spatial_max_distance_round, spatial_avg_distance_round, spatial_initial_max_distance.
- Geometry sidecar now hashed (geometry_hash) and included in integrity digest.
- Parity test: Recomputes Manhattan distances from sidecar bounds; asserts per-agent and aggregate equality.
- Tamper regression test: Mutates marketplace bounds; detects mismatch (fails without parity safeguards).

Reproducibility Guarantees:
- Distances and convergence metrics are now cryptographically anchored via geometry_hash + deterministic recomputation logic.
- Golden replay digest (Step E) + spatial parity combine to detect both economic & spatial drift.

Impact:
- Unlocks safe HUD integration (distance-derived convergence index) and future pathfinding upgrades without invalidating historical logs.
- Establishes baseline for exporting stable spatial KPIs (avg/max distance trajectories) and for any EV / welfare spatial decomposition overlays.

Next Immediate Action:
- Implement HUD surfaces (Step A) leveraging 1.3.0 columns & existing convergence index utility.

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
