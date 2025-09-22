"""Regression test for solver diagnostic logging fields (schema >=1.4.1).

This test runs a minimal simulation (single config assumptions) and ensures
the structured per-agent log includes the solver_* diagnostic columns and
that at least the fundamental fields (status, rest norm) have sensible values.

We do not depend on specific timing numbers—only presence and basic
sanity constraints. The test skips gracefully if the structured logging
layer is unavailable.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from scripts.run_simulation import run_simulation  # type: ignore  # noqa: E402


def _write_minimal_config(tmp: Path) -> Path:
    """Write a small YAML configuration file for a short run.

    Uses 3 goods / 4 agents small grid so that equilibrium calculation
    definitely occurs at least one round.
    """
    # Explicitly typed configuration dict
    content: dict = {
        "simulation": {"name": "solver_logging_test", "max_rounds": 2, "seed": 7},
        "agents": {"count": 4},
        "economy": {
            "goods": 3,
            "grid_size": [7, 7],
            "marketplace_size": [2, 2],
            "movement_cost": 0.0,
            "movement_policy": "greedy",
        },
    }
    import yaml

    cfg_path = tmp / "solver_logging_test.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(content, f)
    return cfg_path


def test_solver_metrics_logged():  # type: ignore[return-type]
    try:
        from src.logging.run_logger import SCHEMA_VERSION  # noqa: F401  (access ensures import path works)
    except Exception:  # pragma: no cover - logging layer absent
        import pytest
        pytest.skip("Structured logging layer not available")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = _write_minimal_config(tmp)
        results = run_simulation(cfg, output_path=tmp, seed_override=11)

        log_path_str = results.get("structured_log_path")
        assert log_path_str, "Structured log path missing from results"
        log_path = Path(log_path_str)
        assert log_path.exists(), "Structured log file not created"

        # Read first few rows from structured log supporting parquet or jsonl
        rows = []
        if log_path.suffix == ".parquet":
            try:
                import pandas as _pd  # type: ignore
                df = _pd.read_parquet(log_path)
                rows = df.head(5).to_dict(orient="records")
            except Exception as e:  # pragma: no cover
                import pytest
                pytest.skip(f"Parquet not readable ({e})")
        else:
            import json as _json
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        rows.append(_json.loads(line))
                    except Exception:
                        # Skip malformed line (defensive); continue collecting
                        continue
        assert rows, "No log rows parsed"

        required_fields = [
            "solver_status",
            "solver_rest_norm",
            "solver_walras_dot",
            "solver_total_time",
            "solver_fsolve_time",
            "solver_fallback_used",
            "solver_method",
        ]
        for field in required_fields:
            assert field in rows[0], f"Missing solver field: {field}"

        # Basic sanity: status not None, rest norm numeric >= 0
        statuses = {r["solver_status"] for r in rows}
        assert None not in statuses, "solver_status should not be None"

        rest_norms = [r["solver_rest_norm"] for r in rows if r["solver_rest_norm"] is not None]
        assert rest_norms, "No rest norm values present"
        assert all(isinstance(v, (int, float)) and v >= 0 for v in rest_norms)

        # Fallback usage may be True/False/None; ensure field present (already checked) and type is bool or None
        assert all((isinstance(r["solver_fallback_used"], (bool, type(None))) for r in rows))

        # Method should be a string or None
        assert all((isinstance(r["solver_method"], (str, type(None))) for r in rows))
