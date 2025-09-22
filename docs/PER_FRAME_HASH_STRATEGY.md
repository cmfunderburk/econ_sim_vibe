# Per-Frame Hash Strategy (Draft)
Date: 2025-09-22
Status: Draft (pending geometry sidecar integration)
Target: Optional integrity enhancement (opt-in flag `--frame-hash`)

## Objectives
1. Detect subtle replay drift (ordering differences, position mis-reconstruction).
2. Keep cost low (hash O(N_agents) not O(N_fields * N_agents)).
3. Maintain forward compatibility (ignore newly added non-critical fields).

## Canonical Hash Input (Frame-Level)
Implementation v1 (2025-09-22) hashes a compact subset of `FrameData` fields via `dataclasses.asdict`:

- `round`
- `participation_count`
- `total_agents`
- `max_distance_to_market`
- `avg_distance_to_market`
- `prices` (list preserved as string representation)

`extra_fields` can be supplied by callers when additional invariants need to be tracked (e.g., digesting `hud_round_digest`). The selection intentionally omits floating fields prone to numerical jitter (solver residuals, travel cost) unless explicitly requested.

## Hash Algorithm
`blake2b` (16 byte digest → 32 hex chars) over the UTF-8 encoded field/value pairs joined with `|`. The implementation lives in `src/visualization/metrics.py` and is used by both live HUD builders and replay integrity tests. BLAKE2b was chosen for its speed and keyed hash support should we later need authenticated digests.

## Output File
`<run_name>_frame_hashes.json`
```
{
  "schema": "frame.hash.v1",
  "hash_algorithm": "blake2b-128",
  "digest_fields": ["round","participation_count","total_agents","max_distance_to_market","avg_distance_to_market","prices"],
  "frames": [
     {"round": 0, "hash": "..."},
     {"round": 1, "hash": "..."}
  ]
}
```

## Activation
Only generated if user passes `--frame-hash` to run script (default: off). This avoids overhead in large exploratory runs.

## Replay Verification Mode (Future)
Provide CLI: `python scripts/verify_run.py --run outputs/run_001` which recomputes hashes from round log + geometry and compares to stored file.

## Edge Cases
- Missing prices → serialized as `[]` in digest input (still hashed deterministically).
- Agent count changes mid-run (should not happen) → verification flags mismatch because `total_agents` is part of the digest.
- Optional extensions must be added via `extra_fields` to avoid silently dropping newly important invariants.

## Testing Plan
1. `test_frame_hash_determinism`: two loads produce identical hash file.
2. `test_frame_hash_subset_insensitivity`: Modify a non-hashed field (e.g., utility) in a copied log and ensure hashes remain unchanged.
3. `test_frame_hash_position_perturbation_detected`: Shuffle one agent's position → hash mismatch.

## Performance Notes
Complexity per round: O(A + G). Memory: trivial.

## Future Evolution
- v2 could incorporate travel cost or average distance once stabilized.
- Consider Blake3 if performance bottleneck emerges (unlikely at current scale).
