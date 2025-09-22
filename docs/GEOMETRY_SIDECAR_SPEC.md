# Geometry Sidecar Specification

Status: Draft
Date: 2025-09-21
Target Release: Logging schema 1.2.0 (additive)

## Purpose
Provide a lightweight, deterministic spatial context artifact accompanying each simulation run so replay tooling can:
- Reconstruct agent distance-to-market metrics post hoc without recomputation ambiguity.
- Derive marketplace bounds & dimensions without parsing core code.
- Validate positional invariants (agents always inside grid; marketplace inside grid).
- Support future pathfinding upgrades (A*) or obstacle layers without mutating historical round logs.

## File
`<run_name>_geometry.json`

## JSON Schema (informal)
```
{
  "schema": "geometry.sidecar.v1",      // bump ONLY on breaking changes
  "run_name": string,
  "generated_at_ns": int,                // wall clock capture
  "grid": {
    "width": int,                        // total grid width (X dimension)
    "height": int                        // total grid height (Y dimension)
  },
  "marketplace": {
    "x_min": int,
    "x_max": int,                        // inclusive bounds
    "y_min": int,
    "y_max": int,
    "width": int,                        // cached = x_max - x_min + 1
    "height": int                        // cached = y_max - y_min + 1
  },
  "movement_policy": "greedy",          // identifier (matches SimulationConfig.movement_policy)
  "random_seed": int,                    // simulation seed
  "notes": string?                       // optional freeform
}
```

Future additive fields (safe): `obstacles: [{x,y},...]`, `distance_cache_version`, `a_star_heuristic`.

Breaking change examples (require schema bump): rename keys, change bounds semantics, remove required field.

## Distance Computation Guidance
Given a cell (x,y) and marketplace bounds, Manhattan distance to the closest marketplace cell is:
```
if x between [x_min, x_max] and y between [y_min, y_max]:
    distance = 0
else:
    dx = 0 if x_min <= x <= x_max else min(abs(x - x_min), abs(x - x_max))
    dy = 0 if y_min <= y <= y_max else min(abs(y - y_min), abs(y - y_max))
    distance = dx + dy
```
Replay layer SHOULD implement this formula directly (O(1)) rather than building a full grid distance map unless obstacles are introduced.

## Writer Responsibilities
Implemented outside core logger to keep logging I/O focused. Simulation driver (e.g. `run_simulation.py`) will:
1. Construct dictionary according to spec.
2. Write JSON prior to `RunLogger.finalize()` (ordering not critical but recommended before finalize for metadata augmentation).
3. Optionally update `*_metadata.json` with `geometry_file` key referencing sidecar filename (performed during or after finalize).

## Validation Tests To Add
- `test_geometry_sidecar_written_and_loadable`: ensures required keys present.
- `test_distance_formula_matches_reference`: random sample vs brute-force enumeration on small grid.
- `test_replay_distance_reconstruction`: frame builder attaches distance and aggregate stats when sidecar present.

## Replay Integration (Planned)
When loading a run, replay stream will attempt to read `<run_name>_geometry.json`. If present:
- Expose `frame.agent_distances` (list parallel to agents in frame) and `frame.distance_stats = {min, max, mean}`.
- If missing, distances remain `None` (graceful fallback).

## Metadata Augmentation
On successful finalize, if geometry file exists, append to metadata JSON under key `geometry_file`.

## Versioning
Initial version labeled `geometry.sidecar.v1`. Future additive changes do NOT change this unless backward incompatibility introduced.

## Rationale
Separating geometry avoids duplicating invariant spatial structure in every per-agent row (space savings) and isolates spatial evolution concerns from economic logging schema, enabling independent evolution.
