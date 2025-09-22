import json
from pathlib import Path
import tempfile
from typing import Dict, Any, List

from scripts.run_simulation import run_simulation
from src.logging.geometry import load_geometry_sidecar, manhattan_distance_to_market


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")  # type: ignore[return-value]
    except Exception:
        return []


def test_spatial_distance_parity_edgeworth():
    config_path = Path("config/edgeworth.yaml")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = run_simulation(config_path, out_dir, seed_override=77)
        log_path = Path(result["structured_log_path"])  # type: ignore[index]
        rows = _load_rows(log_path)
        assert rows, "No rows loaded"
        geom_path = out_dir / f"{result['config']['name']}_seed{result['config']['random_seed']}_geometry.json"
        sidecar = load_geometry_sidecar(geom_path)
        assert sidecar, "Geometry sidecar missing"
        m = sidecar["marketplace"]
        gx_min, gx_max = m["x_min"], m["x_max"]
        gy_min, gy_max = m["y_min"], m["y_max"]

        # Group by round to get aggregates for parity check
        by_round: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_round.setdefault(r["core_round"], []).append(r)

        baseline_initial = None
        for rd, recs in by_round.items():
            recomputed: List[int] = []
            for rec in recs:
                d = manhattan_distance_to_market(
                    rec["spatial_pos_x"], rec["spatial_pos_y"], gx_min, gx_max, gy_min, gy_max
                )
                # Per-agent parity
                assert d == rec["spatial_distance_to_market"], f"Distance mismatch agent={rec['core_agent_id']} round={rd}"
                recomputed.append(d)
                if baseline_initial is None:
                    baseline_initial = rec["spatial_initial_max_distance"]
            # Aggregate parity
            max_logged = {rec["spatial_max_distance_round"] for rec in recs}
            assert len(max_logged) == 1
            avg_logged = {rec["spatial_avg_distance_round"] for rec in recs}
            assert len(avg_logged) == 1
            max_recomp = max(recomputed) if recomputed else 0
            avg_recomp = (sum(recomputed) / len(recomputed)) if recomputed else 0.0
            assert max_logged.pop() == max_recomp, f"Max distance mismatch round={rd}"
            assert abs(avg_logged.pop() - avg_recomp) < 1e-12, f"Avg distance mismatch round={rd}"
        assert baseline_initial is not None
        # Baseline must be >= any recomputed max
        assert all(
            row["spatial_initial_max_distance"] >= row["spatial_max_distance_round"] for row in rows
        )
