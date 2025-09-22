import json
from pathlib import Path
import tempfile
from typing import List, Dict, Any
import pytest

from scripts.run_simulation import run_simulation

REQUIRED_FIELDS: set[str] = {
    "core_schema_version",
    "core_round",
    "core_agent_id",
    "spatial_pos_x",
    "spatial_pos_y",
    "spatial_in_marketplace",
    "econ_prices",
    "econ_executed_net",
    "econ_requested_buys",
    "econ_requested_sells",
    "econ_executed_buys",
    "econ_executed_sells",
    "econ_unmet_buys",
    "econ_unmet_sells",
    "econ_fill_rate_buys",
    "econ_fill_rate_sells",
    "ration_unmet_demand",
    "ration_unmet_supply",
    "wealth_travel_cost",
    "wealth_effective_budget",
    "utility",
    "financing_mode",
    "core_frame_hash",
    "timestamp_ns",
    # Step D spatial fidelity additions (schema 1.3.0)
    "spatial_distance_to_market",
    "spatial_max_distance_round",
    "spatial_avg_distance_round",
    "spatial_initial_max_distance",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            obj: Dict[str, Any] = json.loads(line)
            rows.append(obj)
    return rows


def test_logging_schema_guard_edgeworth():
    """Run a short simulation and verify structured log schema.

    Guard rails:
    1. schema_version consistency (sidecar vs per-row core_schema_version)
    2. Required field presence (no missing columns)
    3. Non-empty financing_mode (currently default 'PERSONAL')
    4. Stable row count = agents * rounds_emitted
    """
    config_path = Path("config/edgeworth.yaml")
    assert config_path.exists(), "Edgeworth config missing"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = run_simulation(config_path, out_dir, seed_override=42)
        # Structured logging path should exist
        log_path_str = result.get("structured_log_path")
        assert log_path_str, "Simulation did not produce structured log path"
        log_path = Path(log_path_str)
        assert log_path.exists(), "Structured log file missing"

        # Load metadata sidecar
        # Metadata file name uses simulation name (not config stem) per run_simulation run_name convention
        meta_path = (
            out_dir
            / f"{result['config']['name']}_seed{result['config']['random_seed']}_metadata.json"
        )
        assert meta_path.exists(), "Metadata sidecar missing"
        meta = json.loads(meta_path.read_text())
        sidecar_version = meta["schema_version"]

        # Load rows (Parquet or JSONL)
        rows: List[Dict[str, Any]]
        if log_path.suffix == ".jsonl":
            rows = _load_jsonl(log_path)
        else:
            try:
                import pandas as pd  # type: ignore

                df = pd.read_parquet(log_path)
                rows = df.to_dict(orient="records")  # type: ignore[assignment]
            except Exception as e:  # pragma: no cover - fallback safety
                pytest.skip(f"Parquet read failed ({e}); environment lacks support")

        assert rows, "No rows logged"

        # Per-row checks
        for r in rows:
            missing = REQUIRED_FIELDS - set(r.keys())
            assert not missing, f"Row missing fields: {missing}"
            assert r["core_schema_version"] == sidecar_version, (
                "Schema version mismatch"
            )
            assert r["financing_mode"] == "PERSONAL", (
                "financing_mode must be 'PERSONAL'"
            )

        # Row count expectation: at least 1 round * 2 agents; stop may occur early
        agent_ids = {r["core_agent_id"] for r in rows}
        rounds = {r["core_round"] for r in rows}
        assert len(agent_ids) == result["config"]["n_agents"]
        # Each round should have one record per agent
        for rd in rounds:
            per_round = [r for r in rows if r["core_round"] == rd]
            assert len(per_round) == len(agent_ids), (
                f"Round {rd} has incomplete agent rows"
            )

        # Sidecar row count matches actual
        assert meta["rows"] == len(rows), "Sidecar row count mismatch"
