## Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog principles (dates in ISO 8601) and Semantic Versioning where applicable. Internal research milestones may appear between public version tags while the project remains pre-1.0.

### [Unreleased]
- A* pathfinding implementation (planned)
- TOTAL_WEALTH financing mode (planned)
- Performance harness & warm-start heuristics (planned)
- Compressed / selective logging options (planned)

### 2025-09-21 – Phase A Completion (Diagnostics & Solver Unification)
Summary: Consolidated solver implementations, added runtime diagnostic layer, introduced feature gating for travel-cost wealth adjustment, expanded test suite (254 tests total) ensuring economic invariants remain stable.

Added:
- Unified `solve_walrasian_equilibrium` implementation (removed legacy minimal variant)
- Environment flag `ECON_SOLVER_ASSERT` (shape/positivity/NaN guards)
- Environment flag `ECON_ENABLE_TRAVEL_BUDGET` (toggle travel-cost budget deduction)
- Diagnostics / shape guard tests (rest-goods norm, normalization invariants)
- Documentation section: “Solver & Runtime Diagnostics” in `SPECIFICATION.md`

Changed:
- README updated to reflect unified solver and new environment flags
- Market order generation respects travel-cost toggle
- Test count increased (250 → 254) with added diagnostic coverage

Removed:
- Legacy duplicate Walrasian solver variant (reduces maintenance risk)

Integrity / Invariants:
- All economic, conservation, and feasibility tests pass (254/254)
- No API changes to external solver interface; status labels extended for clarity

### 2025-09-15 – Logging & Replay Hardening
- Introduced geometry sidecar & frame hash digestion
- Added schema guard (1.3.0) for logging format
- Replay parity tests ensuring deterministic reconstruction

### 2025-09-05 – Spatial Extension Baseline
- Implemented greedy Manhattan movement
- Added travel-cost wealth deduction (initial always-on behavior)
- Integrated money-metric welfare reporting
- Established spatial validation scenarios (V2–V5)

### 2025-08-20 – Phase 1 Economic Engine Complete
- Walrasian equilibrium solver (Cobb-Douglas closed form + fallback)
- Constrained clearing with proportional rationing
- Validation suite V1 plus core unit tests
- Conservation & feasibility invariants enforced

---
Historical pre-changelog commits (prior to 2025-08-20) established initial repository scaffolding, agent framework, and configuration system.
# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to **semantic versioning** in spirit (internal research tool; versions may be date-stamped between formal releases).

## [Unreleased]
- Placeholder for upcoming changes.

### Fixed (post-2025-09-21)
- Replay & logging tests now format-agnostic: accept either JSONL or Parquet (`*_round_log.(jsonl|parquet[.gz])`) restoring green suite when pandas enables Parquet output.
- Resolved numpy array truth-value ambiguity in playback frame construction (`LogReplayStream._build_frame_from_rows`) and golden digest test logic (explicit list/length checks instead of implicit boolean evaluation).
- Normalized Parquet-to-JSONL conversion in regression replay test with numpy array serialization guard (arrays converted to lists pre-dump).

### Maintenance
- Added defensive normalization for `econ_prices` arrays during HUD replay to prevent ValueError in environments with numpy returning ndarray rows from Parquet loader.
- Test suite remains at 250/250 passing following these adjustments; no economic logic modified.

## [2025-09-21] Spatial Logging & Diagnostics Upgrade
### Added
- Logging schema version bumped additively to `1.1.0` (from `1.0.0`) introducing two spatial aggregate metrics: `max_distance_to_market` and `avg_distance_to_market` (repeated per row for analytic convenience).
- Enriched per-agent diagnostic fields: requested/executed buys & sells breakdowns, unmet components, and proportional fill rates.
- Geometry sidecar (`geometry.sidecar.v1.json`) enabling deterministic spatial replay & integrity validation.
- Compression option for run logging (`RunLogger(compress=True)`) supporting gzipped JSONL and Parquet outputs.
- Financing mode tagging field (`financing_mode`), currently always `PERSONAL` (foundation for future `TOTAL_WEALTH`).
- Configuration validation routine (`validate_simulation_config`) with early rejection of inconsistent grid/marketplace setups.

### Changed
- Documentation updated to clarify movement is greedy (A* planned but not implemented) and to describe new spatial convergence metrics.
- Strengthened invariants: movement monotonicity check, randomized conservation fuzzing, per-agent value feasibility proxy integrated into tests.

### Fixed
- Resolved duplicate function definition in robust equilibrium solver module.
- Eliminated lingering lint violations (unused imports, duplicate definitions, placeholder f-strings, E402 import order issues) establishing a clean baseline with `.flake8` policy (`max-line-length=100`).

### Notes
- Schema change is additive: existing downstream parsers remain backward compatible (unknown columns safely ignored).
- `avg_distance_to_market` chosen over median for linear decomposability and smoother convergence tracking; can construct a simple spatial convergence index via `avg_distance_to_market / max_distance_to_market_initial`.
- No changes to economic core algorithms (equilibrium, clearing) beyond instrumentation; all 232 tests pass post-update.
