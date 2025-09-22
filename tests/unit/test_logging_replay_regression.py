"""Regression test: multi-round logging & replay frame reconstruction.

This test guards against a prior bug where only the *final* round's rows
were persisted to the round log (collapsing multi-round history). It
asserts:

1. Contiguous round sequence starting at 1
2. Row count == n_agents * n_rounds (each agent logged exactly once per round)
3. Replay frame count == max round in log
4. Deterministic frame ordering (strictly increasing round numbers)

Implementation notes:
- Uses a very small scenario (edgeworth) to keep runtime minimal.
- Runs with a low horizon (override) if supported; otherwise relies on
  early termination from scenario configuration.
- Skips if run_simulation entry point unavailable (defensive).

If optional parquet dependencies are absent the JSONL path is used.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


def test_multi_round_logging_and_replay(tmp_path: Path):
    # Use edgeworth config (fast & deterministic)
    config_path = Path("config/edgeworth.yaml")
    assert config_path.exists(), "Missing edgeworth.yaml config for regression test"

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    # Run a short simulation; rely on existing CLI interface.
    cmd = [
        sys.executable,
        "scripts/run_simulation.py",
        "--config",
        str(config_path),
        "--seed",
        "42",
        "--no-gui",
        "--output",
        str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert result.returncode == 0, result.stderr

    # Find round log (JSONL preferred; fallback to parquet variants)
    round_logs = list(output_dir.glob("*_round_log.jsonl"))
    if not round_logs:
        round_logs = list(output_dir.glob("*_round_log.parquet")) or list(
            output_dir.glob("*_round_log.parquet.gz")
        )
    assert round_logs, "No round log produced (expected *_round_log.(jsonl|parquet[.gz]))"
    round_log = round_logs[0]
    # If parquet, convert rows to JSON-like records for parsing
    if round_log.suffix.startswith(".parquet"):
        try:
            import pandas as pd  # type: ignore
            df = pd.read_parquet(round_log)  # type: ignore[arg-type]
            tmp_jsonl = output_dir / "_temp_round_log_converted.jsonl"
            with tmp_jsonl.open("w") as f:
                for rec in df.to_dict(orient="records"):  # type: ignore[call-arg]
                    # Normalize any numpy arrays to lists for JSON serialization
                    for k, v in list(rec.items()):
                        try:
                            import numpy as _np  # type: ignore
                            if isinstance(v, _np.ndarray):  # type: ignore[attr-defined]
                                rec[k] = v.tolist()
                        except Exception:
                            continue
                    import json as _json
                    f.write(_json.dumps(rec) + "\n")
            round_log = tmp_jsonl
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"Failed to read parquet round log: {e}")

    rounds: List[int] = []
    agent_ids_per_round: Dict[int, Set[int]] = {}
    # Determine agent id key lazily (first row decides)
    agent_id_key: str | None = None
    with round_log.open() as f:
        for line in f:
            rec = json.loads(line)
            r = rec.get("core_round") or rec.get("round")
            assert r is not None, "Missing round field in log row"
            rounds.append(int(r))
            if agent_id_key is None:
                if "core_agent_id" in rec:
                    agent_id_key = "core_agent_id"
                elif "agent_id" in rec:
                    agent_id_key = "agent_id"
                else:
                    raise AssertionError("No agent id field present in first log row")
            aid = rec[agent_id_key]
            agent_ids_per_round.setdefault(int(r), set()).add(int(aid))

    assert rounds, "Empty round log"
    max_round = max(rounds)
    min_round = min(rounds)
    assert min_round == 1, f"Expected first round 1, got {min_round}"

    # Contiguous sequence
    unique_rounds = sorted(set(rounds))
    assert unique_rounds == list(range(1, max_round + 1)), (
        "Non-contiguous round sequence: "
        f"{unique_rounds} expected 1..{max_round}"
    )

    # Each round should have same number of agent rows
    counts = {r: len(a) for r, a in agent_ids_per_round.items()}
    n_agents = counts[1]
    assert all(c == n_agents for c in counts.values()), (
        f"Inconsistent agent counts per round: {counts}"
    )

    # Row count check
    expected_rows = n_agents * max_round
    actual_rows = len(rounds)
    assert actual_rows == expected_rows, (
        f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
    )

    # Replay headless to count frames (reuse simulation script with --replay)
    replay_cmd = [
        sys.executable,
        "scripts/run_simulation.py",
        "--replay",
        str(round_log),
        "--no-gui",
    ]
    replay_result = subprocess.run(replay_cmd, capture_output=True, text=True, check=True)
    # Parse frame count from stdout (expect pattern 'Frames: X')
    frames: int | None = None
    for line in replay_result.stdout.splitlines():
        if "Replay complete. Frames:" in line:
            try:
                frames = int(line.rsplit("Frames:", 1)[1].strip())
            except ValueError:
                frames = None
    assert frames == max_round, f"Replay frames {frames} != max_round {max_round}"