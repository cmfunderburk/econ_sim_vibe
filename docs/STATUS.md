# Project Status

Updated: 2025-09-21

## Test & Quality Summary
- 217/217 tests passing (205 unit + 12 validation)
- FinancingMode toggle implemented: PERSONAL (active), TOTAL_WEALTH (placeholder)
- Regression guard ensures default behavior == explicit PERSONAL
- Movement: Greedy step toward marketplace (A* not yet implemented)
- Travel cost model active: w_i = max(0, p·ω_total - κ·d_i)

## Feature Matrix
| Area | Status | Notes |
|------|--------|-------|
| Equilibrium Solver | Complete | Cobb-Douglas closed form + checks |
| Market Clearing | Complete | Constrained clearing w/ proportional rationing |
| Financing Mode Toggle | Partial | Enum + regression guard; second mode TBD |
| Travel Costs | Integrated | Budget-side deduction implemented |
| Movement | Basic | Greedy Manhattan; A* planned |
| Logging | Partial | Structured schema planned; placeholder only |
| Visualization | Not Implemented | Planned pygame layer |
| Pathfinding (A*) | Not Implemented | Pending priority decision |

## Recent Milestones
- Synchronized documentation test counts to 217/217
- Added FinancingMode regression test
- Clarified movement is greedy (not A*) across docs
- Consolidated overlapping documentation into single status/roadmap/developer set

## Near-Term Priorities
1. Decide: Implement TOTAL_WEALTH semantics or remove placeholder
2. Optional: A* pathfinding vs confirm greedy as canonical for Phase 2
3. Implement structured Parquet logging (prices, trades, liquidity gap)
4. Basic visualization or analytic notebooks expansion

## Economic Invariants (Current Enforcement)
- Numéraire: p[0] = 1.0
- Convergence: ||Z_rest||_∞ < 1e-8
- Per-agent value feasibility (PERSONAL): p·buys ≤ p·sells + ε
- Conservation per good: total buys == total sells
- No negative inventories produced

## Known Gaps / Open Questions
- TOTAL_WEALTH semantics not yet defined
- No A* path optimality proofs since not implemented
- Logging layer lacks persistent Parquet output
- Visualization absent; headless only

## Changelog (Recent)
- 2025-09-21: Doc consolidation (STATUS/ROADMAP/DEVELOPER_GUIDE) and test count sync
- 2025-09-21: FinancingMode regression guard included in test suite
