# Near-Term Backlog Reprioritization Proposal
Date: 2025-09-21
Status: Draft

## Guiding Principles
1. Solidify spatial + replay fidelity before feature layering (hashing, advanced metrics).
2. Maintain green test suite after each incremental addition.
3. Prefer additive artifacts (sidecars, new columns) over mutating existing schemas.

## Proposed Sequence (Next 2–3 Iterations)
1. Geometry Sidecar (DONE spec) + Implementation & Tests
2. Replay Distance Integration (agent + aggregate stats)
3. Extended Summary Metrics (distance, travel cost, norms) – schema 1.2.0
4. Optional Frame Hash (opt-in) + Verification CLI
5. GUI Keybindings Implementation + Help Overlay
6. Bookmark / Markers (optional: first clearing round, all-arrived round)
7. Staggered Return Mode Flag (if pedagogical need arises)
8. Pathfinding Decision: either implement minimal A* or formally de-scope for current research phase.

## Deferrals / Lower Priority
- Video export / animated GIF generation (post pedagogy validation)
- Liquidity gap advanced metric (requires conceptual definition validation)
- Obstacle layer & congestion regimes (requires economic rationale)

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Schema churn causing replay incompatibility | Strict additive strategy; sidecars separate concerns |
| Over-complication of hash inputs | Keep minimal subset; document exclusions |
| Movement policy divergence from tests | Parameterize tests when enabling staggered returns |

## Decision Points Upcoming
- A* necessity vs documentation clarity.
- Whether to pin geometry sidecar schema at v1 or include placeholder for obstacles now.

## Success Criteria
- All new artifacts (geometry, hash, extended summary) have dedicated tests.
- Replay UI demonstrates smooth navigation, distance overlay correctness.
- No increase in baseline test runtime >5%.

