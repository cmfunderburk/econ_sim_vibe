# Extended Round Summary Metrics Plan
Date: 2025-09-21
Status: Draft
Target Schema Bump: Logging schema 1.2.0 (additive)

## Goals
Augment existing per-round CSV summary with additional low-cost metrics enabling faster exploratory analysis without loading full row log.

## Current Columns (1.1.0)
`round, agents, participants, prices, executed_net, executed_buys, executed_sells, unmet_buys, unmet_sells, avg_buy_fill, avg_sell_fill`

## Proposed Additive Columns
| Column | Type | Description |
|--------|------|-------------|
| `avg_distance` | float | Mean Manhattan distance to marketplace among all agents (requires geometry sidecar) |
| `max_distance` | float | Max Manhattan distance this round |
| `at_market_full` | int | Count of agents with distance == 0 (redundant with participants when policy simple; future-proofing) |
| `cum_travel_cost` | float | Sum of `wealth_travel_cost` across agents (end-of-round) |
| `exec_net_norm_inf` | float | Infinity norm of aggregated executed_net vector (market activity magnitude) |
| `solver_z_rest_norm` | float? | If available from equilibrium solver that round (None if no solve) |
| `solver_status` | string? | e.g. `converged`, `max_iter`, `skipped` |
| `efficiency_loss_partial` | float? | Optional placeholder for running efficiency/welfare loss metric (future) |

## Serialization
- Numeric scalars: formatted with `'{x:.10g}'` for consistency.
- Missing values: empty string.
- Columns appended to right to preserve backward compatibility.

## Data Sources
- Distances: compute on the fly using geometry sidecar formula during summary generation (no need to store in each row).
- Travel cost: already stored per-agent (`wealth_travel_cost`).
- Solver residual / status: expose from `solve_walrasian_equilibrium` via `RuntimeSimulationState.last_market_result` or additional state fields.

## Implementation Steps
1. Ensure solver returns (z_rest_norm, status) every clearing round (already partly captured).
2. Add distance computation helper in summary writer path if geometry file present.
3. Aggregate travel cost sum.
4. Compute exec_net infinity norm: `max(abs(x))` over aggregated executed_net list.
5. Append new headers & values.
6. Bump schema version to `1.2.0` (additive columns) in `run_logger.py`.
7. Update tests: verify new headers present; spot-check distance with small deterministic config.

## Tests
- `test_round_summary_extended_columns_present`.
- `test_round_summary_distance_matches_manual` (small grid; compare known positions).
- `test_round_summary_solver_fields_optional` (rounds without clearing leave fields empty).

## Performance Considerations
Distance computation O(A); negligible relative to existing aggregation.

## Future Metrics (Not in 1.2.0)
- Median fill rate per side.
- Liquidity gap measure (value of unmet vs executed by price weight).
- Gini coefficient of executed trade values.

## Rationale
Keeps row log lean while giving analysts quick time-series vectors for spatial convergence (distance collapse), market intensity (norm), and solver health.
