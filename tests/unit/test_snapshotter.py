from pathlib import Path
from src.visualization.snapshot import Snapshotter
from src.visualization.frame_data import FrameData


def _minimal_frame(round_num: int = 1) -> FrameData:
    return FrameData(
        round=round_num,
        grid_width=1,
        grid_height=1,
        market_x0=0,
        market_y0=0,
        market_width=1,
        market_height=1,
        agents=[],
        prices=[1.0],
        participation_count=0,
        total_agents=0,
    )


def test_snapshotter_json_only(tmp_path: Path):
    s = Snapshotter(tmp_path)
    frame = _minimal_frame()
    meta_path = s.save(frame)
    assert meta_path.exists()
    assert meta_path.suffix == ".json"
    # Basic content check
    import json

    data = json.loads(meta_path.read_text())
    assert data["round"] == 1
    assert "snapshot_timestamp_ms" in data
