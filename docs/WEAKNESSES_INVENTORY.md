# Current Codebase Weaknesses Inventory (2025-09-21)

Purpose: Canonical snapshot of concrete issues to address in upcoming hardening tasks. Each item includes impact, scope, and proposed remediation path. This document is source-of-truth for tasks 1–4 in the hardening plan and should be updated as issues are resolved.

---
## 1. Architectural / Structural

### 1.1 Duplicate `SimulationState` Definitions *(Resolved 2025-09-21)*
- Location(s): `scripts/run_simulation.py` (local dataclass), likely also `src/core/types.py` (not yet imported in runner).
- Impact: Divergence risk between logging, market invariants, and external interfaces; increases maintenance cost.
- Proposed Fix: Create a single canonical `SimulationState` in `src/core/state.py` (new) or reuse existing definition in `core/types.py`; adapt runner to use it. Provide migration shim for backward compatibility if tests depend on old import path.

### 1.2 `sys.path.append` Hack in Runner
- Location: `scripts/run_simulation.py` line near top.
- Impact: Fragile import mechanics; breaks when packaged/installed; hinders console_scripts packaging.
- Proposed Fix: Convert repository to rely purely on editable install (`pip install -e .`). Add entry point `econ-sim-run` referencing `scripts.run_simulation:main`. Remove path mutation.

### 1.3 Script-Centric Logic Instead of Library API
- Location: `scripts/run_simulation.py` bundles initialization, loop control, logging.
- Impact: Hard to unit test round-level logic; cannot easily reuse simulation core in benchmarks or alternative frontends.
- Proposed Fix: Extract pure function `run_round(state, config)` and `initialize(config)` into `src/core/simulation.py` with no prints; keep script as thin CLI wrapper.

---
## 2. Solver & Numerical Stability

### 2.1 No Fallback When `fsolve` Poorly Converges *(Resolved 2025-09-21)*
- Location: `src/econ/equilibrium.py` (`solve_walrasian_equilibrium`) – status `poor_convergence` only logs warning.
- Impact: Downstream may proceed with imprecise prices; rationing analysis distorted; potential invariant drift.
- Proposed Fix: Implement tâtonnement fallback when `z_rest_norm >= SOLVER_TOL` or on exception. Adaptive step: Δp_rest = η * sign(Z_rest); decrease η geometrically until improvement; cap iterations; tag status `fallback_converged` or `fallback_failed`.

### 2.2 Viability Filtering Per Iteration Returns Sentinel Residual `1e6`
- Impact: Large constant may produce ill-conditioned Jacobian approximations; wastes iterations.
- Proposed Fix: Return scaled residual based on previous iterate norm or project to feasible agent subset before root solve to reduce oscillation.

### 2.3 Price Floor Projection Post-Solve Without Revalidation
- Impact: Projection can shift solution outside root set; no re-check of residuals performed.
- Proposed Fix: Recompute residuals & optionally re-run small refinement after projection.

### 2.4 Missing Deterministic Warm Starts Across Rounds
- Impact: Each round uses uniform initial guess unless caller supplies one; slower convergence, higher variance of iteration counts.
- Proposed Fix: Accept previous round `p_rest` as `initial_guess`; store in simulation state.

---
## 3. Invariants & Testing Gaps

### 3.1 Movement Monotonicity Untested *(Resolved 2025-09-21)*
- Impact: Regression risk if movement logic changes (e.g., future A* integration) breaking expected greedy monotonic distance descent.
- Proposed Fix: Add test ensuring Manhattan distance to ANY marketplace cell is non-increasing for agents in `TO_MARKET` phase.

### 3.2 Per-Agent Value Feasibility Not Revalidated in Runner
- Impact: Clearing function presumably enforces, but runner does not assert post-application.
- Proposed Fix: Add invariant hook after `apply_trades_to_agents` verifying Σ p·buys ≤ Σ p·sells + ε.

### 3.3 No Fuzz / Property Tests for Conservation Under Randomized Seeds *(Resolved basic 2025-09-21)*
- Impact: Edge cases (near-zero wealth, concentrated endowment) might slip.
- Proposed Fix: Hypothesis-based test generating small economies (2–5 agents, 2–4 goods) validating conservation and non-negativity.

### 3.4 Logging Row Count vs. Rounds Not Verified *(Resolved 2025-09-21)*
- Impact: Potential silent truncation or buffering loss if finalize fails.
- Proposed Fix: Test that number of serialized agent records = Σ_round agents_in_simulation (or marketplace filter if scoped) and finalize ensures flush.

### 3.5 Liquidity / Rationing Diagnostics Coverage Missing
- Impact: `unmet_demand` and `unmet_supply` fields not validated to satisfy sign / aggregation relationships.
- Proposed Fix: When present, assert Σ executed_net + unmet_supply - unmet_demand ≈ theoretical Z (needs retrieval from solver path or reconstruction).

---
## 4. Logging Layer Weaknesses

### 4.1 Schema Backward Compatibility Policy Not Codified *(Resolved 2025-09-21)*
- Impact: Additive changes could accidentally remove fields; no automated guard.
- Proposed Fix: Introduce `tests/unit/test_logging_schema_guard.py` loading a canonical expected field list for `SCHEMA_VERSION`.

### 4.2 Missing Compression Option *(Resolved 2025-09-21)*
- Impact: Large runs generate large JSONL/Parquet files.
- Proposed Fix: Add `--log-compress` flag; Parquet compression (snappy) or gzip JSONL.

### 4.3 No Effective Budget / Financing Mode Population *(Partially Resolved 2025-09-21)*
- Impact: Analysis downstream cannot distinguish financing regimes or travel-adjusted wealth.
- Proposed Fix: Compute travel-adjusted wealth w_i inside runner when pricing occurs; populate `wealth_effective_budget`; set `financing_mode='PERSONAL'` constant for now.

### 4.4 Lack of Flush on Exception Before Finalize
- Impact: Crash mid-run loses buffered records.
- Proposed Fix: Context manager or try/except invoking `flush()` in finally block each round.

---
## 5. Packaging / Distribution

### 5.1 Missing `setup.cfg` or `pyproject.toml` Consolidation
- Impact: Hard to enforce tool versions; configuration fragmentation.
- Proposed Fix: Migrate to `pyproject.toml` (PEP 621) with optional extras: `dev`, `logging` (pyarrow), `bench`.

### 5.2 Absent Console Entry Points
- Impact: Users call scripts via repo-relative path; reduces portability.
- Proposed Fix: Add entry points: `econ-sim-run=src.cli:run_sim`, `econ-sim-validate=src.cli:validate`.

---
## 6. Configuration Validation

### 6.1 No Structural Validation of YAML *(Resolved 2025-09-21)*
- Impact: Typos silently default (e.g., mis-typed `marketplace_size`); hidden errors.
- Proposed Fix: `pydantic` model or manual validator verifying positive ints, goods ≥2, marketplace dims ≤ grid dims.

### 6.2 Lack of Dimension Cross-Checks
- Impact: If `n_goods` mismatches lengths of drawn arrays (future manual config), could misalign.
- Proposed Fix: Post-initialization assertion verifying all alpha vectors length == n_goods.

---
## 7. Type & Static Analysis

### 7.1 Optional Logging Imports Trigger `Any` Pollution
- Impact: mypy cannot enforce structure for `run_logger` usages.
- Proposed Fix: Provide `Protocol` for logger methods; use `typing.TYPE_CHECKING` import stanza.

### 7.2 Agents Type Not Explicit in Solver
- Impact: `List` untyped; relying on dynamic attribute presence (`alpha`, endowments).
- Proposed Fix: Define `AgentProtocol` with required attributes for solver; update signatures.

---
## 8. Performance / Benchmarking

### 8.1 No Baseline Timing Harness
- Impact: Changes could degrade performance unnoticed.
- Proposed Fix: `scripts/benchmark.py` producing JSON baseline (agent_count, goods, median_round_time, solver_iters if exposed).

### 8.2 Lack of Price Solver Iteration Metrics
- Impact: Hard to attribute slowdown (solver vs clearing vs movement).
- Proposed Fix: Instrument solver to optionally return `n_function_evals` or attach callback counting calls to `excess_demand_rest_goods`.

---
## 9. Documentation Gaps / Drift Risks

### 9.1 Movement Policy Wording Inconsistent (Greedy vs 'myopic A*')
- Proposed Fix: Standardize language: "current implementation: greedy Manhattan descent (no A*)."

### 9.2 Financing Mode Future Semantics Not Linked to Logging Fields
- Proposed Fix: Add note in logging docs mapping `financing_mode` field to enum and placeholder semantics.

### 9.3 Missing Troubleshooting Section
- Proposed Fix: Add section for common warnings: poor_convergence, zero-wealth exclusion, logging fallback to JSONL.

---
## 10. Reliability / Failure Handling

### 10.1 Solver Exception Swallowing in Runner (Generic Warning)
- Location: `run_simulation_round` broad except prints warning only.
- Impact: Silent degradation; no structured indicator for analysis.
- Proposed Fix: Propagate structured status flag in state; log into structured records (per-round field or metadata) with error code.

### 10.2 No Early Abort on Repeated Failures
- Impact: Could spin useless rounds with invalid prices.
- Proposed Fix: Counter of consecutive non-converged rounds -> terminate.

---
## 11. Rationing Diagnostics Completeness

### 11.1 No Link Between Unmet Demand/Supply Arrays and Executed Trades
- Impact: Hard to verify diagnostic correctness.
- Proposed Fix: Add checks: `executed_volume_g + unmet_demand_g == requested_buys_g` (if supply>=demand) symmetrical; requires exposing requested totals in `MarketResult`.

---
## Prioritization (Suggested Order)
1. Safety & Correctness: 2.1, 10.1, 10.2, 3.2, 3.3
2. Observability: 4.1, 4.3, 11.1, 3.4
3. Architecture: 1.1, 1.2, 1.3, 7.2
4. Robustness: 2.2, 2.3, 2.4
5. Tooling & Packaging: 5.x, 7.1, 6.x
6. Performance & Benchmarks: 8.x
7. Documentation & UX: 9.x

---
## Change Log for This Inventory
- 2025-09-21: Initial creation.
- 2025-09-21: Updated resolved items (solver fallback, movement test, schema guard, compression, config validation).

Update this file when items are resolved; annotate with PR numbers and resolution notes.
