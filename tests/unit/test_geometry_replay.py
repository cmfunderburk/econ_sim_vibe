import json
from pathlib import Path
from src.logging.geometry import (
    GeometrySpec,
    write_geometry_sidecar,
    load_geometry_sidecar,
    GEOMETRY_SIDECAR_SCHEMA,
)
from src.visualization.playback import LogReplayStream

# Minimal synthetic log lines for replay: two rounds, two agents
SAMPLE_LOG = (
    """
{"core_round": 1, "core_agent_id": 0, "spatial_pos_x": 2, "spatial_pos_y": 2, "spatial_in_marketplace": true, "econ_prices": [1.0, 2.0]}
{"core_round": 1, "core_agent_id": 1, "spatial_pos_x": 0, "spatial_pos_y": 0, "spatial_in_marketplace": false, "econ_prices": [1.0, 2.0]}
{"core_round": 2, "core_agent_id": 0, "spatial_pos_x": 3, "spatial_pos_y": 2, "spatial_in_marketplace": true, "econ_prices": [1.0, 2.0]}
{"core_round": 2, "core_agent_id": 1, "spatial_pos_x": 0, "spatial_pos_y": 1, "spatial_in_marketplace": false, "econ_prices": [1.0, 2.0]}
""".strip()
    + "\n"
)


def test_geometry_sidecar_write_and_load(tmp_path: Path):
    spec = GeometrySpec(
        run_name="demo_run",
        grid_width=5,
        grid_height=5,
        market_x_min=2,
        market_x_max=3,
        market_y_min=2,
        market_y_max=3,
        movement_policy="greedy",
        random_seed=42,
    )
    sidecar_path = write_geometry_sidecar(tmp_path, spec)
    loaded = load_geometry_sidecar(sidecar_path)
    assert loaded is not None
    assert loaded["schema"] == GEOMETRY_SIDECAR_SCHEMA
    assert loaded["grid"]["width"] == 5
    assert loaded["marketplace"]["x_min"] == 2


def test_replay_with_geometry_distances(tmp_path: Path):
    # Write synthetic log
    log_path = tmp_path / "demo_run_round_log.jsonl"
    log_path.write_text(SAMPLE_LOG)
    # Sidecar matching naming convention
    spec = GeometrySpec(
        run_name="demo_run",
        grid_width=5,
        grid_height=5,
        market_x_min=2,
        market_x_max=3,
        market_y_min=2,
        market_y_max=3,
        movement_policy="greedy",
        random_seed=7,
    )
    write_geometry_sidecar(tmp_path, spec)
    stream = LogReplayStream(log_path=log_path)
    f1 = stream.frame_at(1)
    assert f1 is not None
    # Agent 0 inside marketplace -> distance 0
    a0 = [a for a in f1.agents if a.agent_id == 0][0]
    assert a0.distance_to_market == 0
    # Agent 1 at (0,0); marketplace rectangle x:[2,3], y:[2,3]
    # Manhattan distance = |2-0| + |2-0| = 4
    a1 = [a for a in f1.agents if a.agent_id == 1][0]
    assert a1.distance_to_market == 4
    # Max & average distance checks
    assert f1.max_distance_to_market == 4
    assert abs((f1.avg_distance_to_market or 0) - 2.0) < 1e-9


def test_replay_without_geometry_sidecar(tmp_path: Path):
    log_path = tmp_path / "orphan_round_log.jsonl"
    log_path.write_text(SAMPLE_LOG)
    stream = LogReplayStream(log_path=log_path)
    f1 = stream.frame_at(1)
    assert f1 is not None
    assert f1.grid_width > 0 and f1.grid_height > 0
    # Fallback inference should still compute meaningful distances
    dist_map = {a.agent_id: a.distance_to_market for a in f1.agents}
    assert dist_map[0] == 0
    assert dist_map[1] == 4
    assert f1.max_distance_to_market == 4
    assert abs((f1.avg_distance_to_market or 0) - 2.0) < 1e-9
