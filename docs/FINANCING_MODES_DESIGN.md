# Financing Modes Design

Date: 2025-09-21
Status: Draft (Implementation Not Yet Started)
Schema Impact: Anticipated additive bump to 1.2.0

## 1. Motivation
Current simulation enforces PERSONAL financing: agents may only finance purchases with value of goods physically present (personal inventory). Research requires contrasting this constrained barter regime with a hypothetical wealth-backed regime where agents can commit the value of total endowment irrespective of composition (TOTAL_WEALTH). This isolates liquidity constraints from allocative fundamentals and refines welfare counterfactuals.

## 2. Mode Definitions
### PERSONAL (Existing)
- Wealth used for demand sizing: w_personal = p · personal_endowment (implicitly via order sizing because personal inventory both supplies and finances).
- Feasibility invariant (executed): ∑_g p_g * buys_exec_{i,g} ≤ ∑_g p_g * sells_exec_{i,g} + ε
- Liquidity Gap Interpretation: Theoretical excess demand (total) − executed (personal constrained).

### TOTAL_WEALTH (Proposed)
- Wealth used for demand sizing: w_total = max(0, p · (home + personal) − travel_cost_deduction)
- Order computation: Target optimal bundle x*_i(p, w_total) (Cobb-Douglas). Desired net = x*_i − personal_inventory.
- Execution constraints:
  1. Sell quantities still capped by personal inventory (physical availability constraint unaffected).
  2. Value feasibility relaxes: purchases may exceed value of same-round sells up to total endowment value.
- Updated Feasibility Invariant (executed): ∑_g p_g * buys_exec_{i,g} ≤ p · (home + personal)_i + ε
- Liquidity Gap Adjustment: Under TOTAL_WEALTH, gap should shrink (or remain equal) relative to PERSONAL: liquidity_gap_total ≤ liquidity_gap_personal component-wise in aggregate expectations.

## 3. Economic Invariants (Both Modes)
Invariant | PERSONAL | TOTAL_WEALTH | Comment
--------- | -------- | ------------ | -------
Numéraire | p[0] = 1 | p[0] = 1 | Unchanged
Rest-Goods Convergence | ||Z_rest||_∞ < SOLVER_TOL | Same | Price formation unchanged (LTE participants total endowment)
Goods Conservation | ✓ | ✓ | Execution transfer only
Value Feasibility | p·buys_exec ≤ p·sells_exec + ε | p·buys_exec ≤ p·ω_total + ε | Relaxed upper bound in TOTAL_WEALTH
Sell Capacity | sells_exec_g ≤ personal_stock_g | Same | Physical constraint persists
Liquidity Gap Sign | gap ≥ 0 where constrained | Expect weak reduction | Compare across modes
Travel Cost Deduction | Budget side only | Budget side only | Deduct κ·d_i from w_total

## 4. Algorithmic Changes
1. Order Sizing:
   - PERSONAL: Keep existing logic (implicitly limited by personal stock & value feasibility post execution).
   - TOTAL_WEALTH: Compute ideal demand with w_total, derive desired net = x*_i − personal_inventory.
2. Clearing:
   - Keep proportional rationing by aggregate supply/demand.
   - Modify value feasibility check branch based on mode.
3. Post-Execution Validation:
   - PERSONAL: existing per-agent p·buys_exec ≤ p·sells_exec + FEASIBILITY_TOL
   - TOTAL_WEALTH: new check p·buys_exec ≤ p·ω_total + FEASIBILITY_TOL
4. Logging Additions (Schema 1.2.0 tentative fields):
   - financing_mode (already present)
   - total_wealth_LTE (p·(home+personal) at pricing)
   - travel_cost_deduction (κ·d_i term)
   - effective_budget (w_total after travel cost)
   - value_feasibility_slack (p·ω_bound − p·buys_exec)

## 5. Data Flow Summary
```
Round:
  Movement → (positions, distances)
  Pricing (LTE using total endowments) → prices p
  For each agent in market:
    w_total = max(0, p·(home+personal) − κ·d_i)
    x*_i = α_i * w_total / p (Cobb-Douglas)
    desired_net = x*_i − personal_inventory
    separate desired_net into buys/sells (positive/negative)
  Aggregate orders → execute_constrained_clearing(mode)
  Rationing + logging (with mode-dependent feasibility checks)
```

## 6. Edge Cases
Case | Handling
---- | --------
Zero total wealth | Exclude from LTE already; w_total = 0, desired_net = −personal_inventory (sell what you brought) | identical
All positive desired_net (pure buyer) with limited personal inventory | PERSONAL disallows value financing; TOTAL_WEALTH allows buys funded by home endowment value | buy execution limited by available sell supply from others
Infinite κ (distance prohibitive) | w_total may truncate to 0 → identical behavior in both modes | robust
Single participant | No pricing; mode irrelevant | skip

## 7. Testing Plan
Category | Test
-------- | ----
Invariant | test_total_wealth_value_feasibility
Invariant | test_personal_vs_total_liquidity_gap (aggregate gap reduction)
Schema | test_schema_1_2_0_additive_fields
Behavior | test_total_wealth_pure_buyer_executes (buys > sells possible)
Behavior | test_total_wealth_reverts_when_equal (if personal == total)
Performance | test_total_wealth_no_solver_regression (same convergence norm distribution)
Edge | test_zero_total_wealth_agent_stability

## 8. Implementation Steps
1. Introduce enum FinancingMode with both members (already partially present) and pass mode into order generation & clearing.
2. Extend order generation: branch on mode for budget sizing.
3. Adjust clearing feasibility branch; add slack computation.
4. Extend logging (schema bump to 1.2.0) with additive fields.
5. Add tests enumerated above; backfill documentation (SPECIFICATION + STATUS + CHANGELOG).
6. Provide migration note: existing logs parsed seamlessly; new columns optional.

## 9. Risks & Mitigation
Risk | Mitigation
---- | ---------
Mode-specific branching causes code drift | Isolate mode logic in helper (compute_financing_bounds(agent, prices, mode))
Liquidity gap semantics confusion | Explicit doc section explaining gap meaning per mode
Performance overhead (extra value computation) | Precompute p·ω_total vectorized
Test brittleness with randomness | Seed fixation + scenario-specific deterministic configs

## 10. Open Questions
- Should travel_cost_deduction be logged even in PERSONAL for symmetry? (Recommendation: yes for consistency.)
- Should liquidity_gap be recomputed post-mode or have separate metrics (gap_personal, gap_total)? (Recommendation: keep single gap reflecting mode.)
- Introduce warning if mode TOTAL_WEALTH yields no gap reduction in synthetic scenario (diagnostic)?

## 11. Acceptance Criteria
- All prior 232 tests still pass.
- New tests pass; schema guard updated.
- Documentation updated (SPECIFICATION: Financing Modes, logging fields; STATUS & CHANGELOG updated).
- Performance delta < 5% on benchmark scenario (100 agents × 100 rounds) relative to PERSONAL baseline.

## 12. Out-of-Scope (For This Iteration)
- Credit intermediation or interest-bearing assets.
- Partial collateralization or leverage limits.
- Adaptive switching of financing mode mid-run.

---
End of Design Document.
