# NEXT STEPS (2025-09-21)

This document captures the prioritized remediation and enhancement plan following recent logging/replay test adjustments and a successful full test suite (250/250) pass.

## Phase A – Core Economic Correctness (High Priority)

1. Solver Shape Mismatch Reproduction (Todo #19)
   - Goal: Surface and reliably reproduce any dimension mismatch warnings in `solve_walrasian_equilibrium`.
   - Actions:
     - Add diagnostic assertions in the solver: `assert prices.shape[0] == n_goods` and `assert excess_demand.shape == (n_goods,)` before returning.
     - Create a minimal pytest (e.g., `test_equilibrium_shape_guard.py`) using a 3-good config to ensure dimensions align.
     - Fail fast rather than logging silent warnings.
   - Risks: Could expose latent issues causing current scenarios to fail; mitigated via targeted test gating behind an environment variable until stabilized.

2. Unify Solver Implementation (Todo #20)
   - Consolidate duplicate solver definitions into a single authoritative function with a clear contract (inputs, outputs, convergence metrics, status enum).
   - Add docstring referencing rest-goods norm convergence criterion and numéraire normalization.
   - Update all imports and add a regression test confirming only one public solver symbol exported.

3. Travel-Cost Adjusted Order Budgets (Todo #21)
   - Implement wealth deduction: `w_i = max(0, p · (ω_home + ω_personal) - κ * d_i)` for order sizing (execution stage only; pricing still uses total endowment).
   - Add unit test asserting when κ>0 and distance>0, computed demand scales down vs κ=0 baseline.
   - Update validation scenario (spatial null κ=0) to assert identical behavior to prior baseline.

4. Solver Diagnostics Tests (Todo #25)
   - Add tests for: no NaNs in price vector, `p[0] == 1.0`, `||Z_rest||_∞ < SOLVER_TOL` on convergence.
   - Add test ensuring fallback (tâtonnement) path sets a distinct status when used.

## Phase B – Architecture & Feature Enablement (Medium Priority)

5. Position Type Unification (Todo #22)
   - Search for redundant `Position` definitions—ensure only `spatial.grid.Position` remains.
   - Refactor imports and add a test that all modules reference the canonical class (e.g., using `inspect.getmodule`).

6. Movement Policy Alignment (Todo #23)
   - Option A: Implement A* pathfinding (Manhattan heuristic, early exit, deterministic tie-breaking) and add optimality test under static cost regime.
   - Option B: Update all documentation to explicitly state greedy movement (if deferring A* for performance simplicity).
   - Decision Gate: Choose based on immediate research needs; default to doc alignment if no path optimality analysis required.

7. FinancingMode TOTAL_WEALTH (Todo #24)
   - Add enum branch allowing order financing against total endowment value (still constrained by personal inventory on sells).
   - Extend logging with `financing_mode` (already present) and new wealth fields: `total_wealth_lte`, `travel_adjusted_wealth`.
   - Comparative test: Same scenario under PERSONAL vs TOTAL_WEALTH; assert liquidity gap shrinks or stays equal.

8. Logging Schema Increment (Todo #26)
   - Bump schema minor version (e.g., 1.2.0) when adding new wealth fields.
   - Update replay loader to ignore unknown fields gracefully.
   - Add schema compatibility test: old (synthetic) 1.1.0 log still replays.

## Phase C – Performance & Instrumentation (Lower Priority but Valuable)

9. Performance Benchmark Harness (Todo #27)
   - Add `scripts/benchmark.py` to profile 100+ agents, capturing: avg solver time, pathfinding time, rounds/sec.
   - Create pytest marker `@pytest.mark.performance` (skipped by default in CI) summarizing metrics vs baseline thresholds.

10. Documentation Updates (Todo #28)
   - SPECIFICATION.md: Reflect travel-cost adjusted budget usage and movement policy status.
   - README: Add note on new financing mode toggle and schema version shift.
   - CHANGELOG: Add entries for each completed high-level change.

## Cross-Cutting Concerns

- Invariants Preservation: After budget adjustment & financing mode addition, re-run all V1–V10 validation scenarios and confirm conservation/value feasibility invariants.
- Backward Compatibility: Maintain prior behavior when κ=0 and financing mode = PERSONAL.
- Determinism: Ensure A* (if implemented) enforces tie-breaking consistent with current greedy lexicographic ordering to preserve replay hashes (update golden digest if inherently unavoidable, with justification).

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Solver refactor introduces regression | High | Incremental PR: add tests before removal; feature flag temporary asserts |
| Budget adjustment changes expected test outcomes | Medium | Baseline snapshots with κ=0; isolate κ>0 new expectations |
| Schema bump breaks replay tooling | Medium | Add compatibility test with frozen 1.1.0 sample log |
| A* pathfinding performance degradation | Medium | Provide flag to select movement policy; benchmark pre/post |
| Financing mode complicates clearing invariants | Medium | Add per-mode invariant tests (value feasibility branching) |

## Suggested Implementation Order (Sprint Skeleton)

Sprint 1 (Correctness): Tasks 19, 20, 21, 25
Sprint 2 (Architecture/Features): Tasks 22, 23, 24, 26
Sprint 3 (Performance & Docs): Tasks 27, 28 plus benchmarking & doc updates

## Definition of Done Per Major Task
- Solver Unification: Single function export, all solver tests pass, no duplicate definitions via grep.
- Travel-Cost Budget: New tests pass; welfare metrics adjust; κ=0 regression green.
- FinancingMode: Mode switchable via config; logs reflect mode; comparative test included.
- Movement Policy: Either A* present with optimality test OR docs updated across README/SPEC/CHANGELOG.
- Logging Schema: Version increment + compatibility test + new fields in sample row.
- Performance Harness: Script + at least one recorded benchmark output (not enforced threshold initially).

---
Generated: 2025-09-21
