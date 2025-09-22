from pathlib import Path
import json
from typing import List, Dict, Any

from src.visualization.playback import LogReplayStream


def _write_jsonl(tmp_path: Path, rows: List[Dict[str, Any]]) -> Path:
    p = tmp_path / "test_round_log.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_log_replay_basic(tmp_path):
    # Create minimal two-round log for two agents
    rows = [
        {
            "core_round": 1,
            "core_agent_id": 1,
            "econ_prices": [1.0, 0.5],
            "spatial_in_marketplace": True,
        },
        {
            "core_round": 1,
            "core_agent_id": 2,
            "econ_prices": [1.0, 0.5],
            "spatial_in_marketplace": False,
        },
        {
            "core_round": 2,
            "core_agent_id": 1,
            "econ_prices": [1.0, 0.6],
            "spatial_in_marketplace": True,
        },
        {
            "core_round": 2,
            "core_agent_id": 2,
            "econ_prices": [1.0, 0.6],
            "spatial_in_marketplace": True,
        },
    ]
    log_path = _write_jsonl(tmp_path, rows)
    stream = LogReplayStream(log_path=log_path)

    f1 = stream.next_frame()
    assert f1 is not None
    assert f1.round == 1
    assert f1.prices == [1.0, 0.5]
    assert f1.participation_count == 1
    assert f1.total_agents == 2

    f2 = stream.next_frame()
    assert f2 is not None
    assert f2.round == 2
    assert f2.prices == [1.0, 0.6]
    assert f2.participation_count == 2

    # End of stream
    assert stream.next_frame() is None


def test_log_replay_seek(tmp_path):
    rows = [
        {
            "core_round": 1,
            "core_agent_id": 1,
            "econ_prices": [1.0],
            "spatial_in_marketplace": True,
        },
        {
            "core_round": 2,
            "core_agent_id": 1,
            "econ_prices": [1.0],
            "spatial_in_marketplace": True,
        },
        {
            "core_round": 3,
            "core_agent_id": 1,
            "econ_prices": [1.0],
            "spatial_in_marketplace": True,
        },
    ]
    log_path = _write_jsonl(tmp_path, rows)
    stream = LogReplayStream(log_path=log_path)
    stream.seek(2)
    f = stream.next_frame()
    assert f is not None and f.round == 3
    # Reset and read again
    stream.reset()
    f_reset = stream.next_frame()
    assert f_reset is not None and f_reset.round == 1
