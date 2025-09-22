from pathlib import Path
import json
from typing import List, Dict, Any

from src.visualization.playback import LogReplayStream


def _write_jsonl(
    tmp_path: Path, rows: List[Dict[str, Any]], name: str = "scrub_log.jsonl"
) -> Path:
    p = tmp_path / name
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_replay_frame_at_and_prev(tmp_path: Path):
    # Three rounds, two agents with changing prices and marketplace flags
    rows: List[Dict[str, Any]] = [
        {
            "core_round": 1,
            "core_agent_id": 1,
            "econ_prices": [1.0, 0.5],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 0,
            "spatial_pos_y": 0,
        },
        {
            "core_round": 1,
            "core_agent_id": 2,
            "econ_prices": [1.0, 0.5],
            "spatial_in_marketplace": False,
            "spatial_pos_x": 1,
            "spatial_pos_y": 0,
        },
        {
            "core_round": 2,
            "core_agent_id": 1,
            "econ_prices": [1.0, 0.6],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 0,
            "spatial_pos_y": 1,
        },
        {
            "core_round": 2,
            "core_agent_id": 2,
            "econ_prices": [1.0, 0.6],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 1,
            "spatial_pos_y": 1,
        },
        {
            "core_round": 3,
            "core_agent_id": 1,
            "econ_prices": [1.0, 0.7],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 0,
            "spatial_pos_y": 2,
        },
        {
            "core_round": 3,
            "core_agent_id": 2,
            "econ_prices": [1.0, 0.7],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 1,
            "spatial_pos_y": 2,
        },
    ]
    log_path = _write_jsonl(tmp_path, rows)
    stream = LogReplayStream(log_path=log_path)

    # Direct random access
    f2 = stream.frame_at(2)
    assert f2 is not None and f2.round == 2 and f2.prices == [1.0, 0.6]
    # Sequential advance
    f1_seq = stream.next_frame()
    assert f1_seq is not None and f1_seq.round == 1
    f2_seq = stream.next_frame()
    assert f2_seq is not None and f2_seq.round == 2
    f3_seq = stream.next_frame()
    assert f3_seq is not None and f3_seq.round == 3
    # Backward navigation
    f2_back = stream.prev_frame()
    assert f2_back is not None and f2_back.round == 2
    f1_back = stream.prev_frame()
    # prev at beginning returns first frame
    assert f1_back is not None and f1_back.round == 1


def test_replay_spatial_agent_frames(tmp_path: Path):
    rows: List[Dict[str, Any]] = [
        {
            "core_round": 1,
            "core_agent_id": 5,
            "econ_prices": [1.0],
            "spatial_in_marketplace": True,
            "spatial_pos_x": 3,
            "spatial_pos_y": 4,
        },
        {
            "core_round": 1,
            "core_agent_id": 6,
            "econ_prices": [1.0],
            "spatial_in_marketplace": False,
            "spatial_pos_x": 10,
            "spatial_pos_y": 2,
        },
    ]
    log_path = _write_jsonl(tmp_path, rows, name="spatial_log.jsonl")
    stream = LogReplayStream(log_path=log_path)
    frame = stream.next_frame()
    assert frame is not None
    # Expect two agent frames with recorded positions
    assert len(frame.agents) == 2
    ids = {a.agent_id for a in frame.agents}
    assert ids == {5, 6}
    pos_map = {a.agent_id: (a.x, a.y, a.in_marketplace) for a in frame.agents}
    assert pos_map[5] == (3, 4, True)
    assert pos_map[6] == (10, 2, False)
