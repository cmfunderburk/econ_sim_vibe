import json
from pathlib import Path
import tempfile
import pytest
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


def test_spatial_distance_parity_tamper_detection():
    """Negative test: tamper geometry sidecar -> parity should fail.

    We artificially shift marketplace bounds by +1 in x_min to break distances.
    If parity logic is sound, at least one mismatch appears.
    """
    config_path = Path("config/edgeworth.yaml")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = run_simulation(config_path, out_dir, seed_override=88)
        log_path = Path(result["structured_log_path"])  # type: ignore[index]
        rows = _load_rows(log_path)
        assert rows, "No rows loaded"
        geom_file = out_dir / f"{result['config']['name']}_seed{result['config']['random_seed']}_geometry.json"
        data = json.loads(geom_file.read_text())
        # Tamper: shift rectangle diagonally and enlarge height
        data["marketplace"]["x_min"] += 2
        data["marketplace"]["x_max"] += 2
        data["marketplace"]["y_min"] += 1
        data["marketplace"]["y_max"] += 2  # expand height by 1
        geom_file.write_text(json.dumps(data, indent=2))
        tampered = load_geometry_sidecar(geom_file)
        assert tampered, "Failed to load tampered sidecar"
        m = tampered["marketplace"]
        gx_min, gx_max = m["x_min"], m["x_max"]
        gy_min, gy_max = m["y_min"], m["y_max"]
        mismatch_found = False
        for r in rows:
            d = manhattan_distance_to_market(
                r["spatial_pos_x"], r["spatial_pos_y"], gx_min, gx_max, gy_min, gy_max
            )
            if d != r["spatial_distance_to_market"]:
                mismatch_found = True
                break
        assert mismatch_found, "Tampering geometry did not trigger any distance mismatch"
