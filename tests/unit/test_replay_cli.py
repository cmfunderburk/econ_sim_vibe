import json
import subprocess
import sys
from pathlib import Path


def _write_minimal_log(path: Path) -> None:
    """Create a tiny JSONL structured log with two rounds, two agents.

    Only fields required by LogReplayStream are included: core_round, core_agent_id,
    econ_prices (list), spatial_in_marketplace (bool).
    """
    lines = []
    # Round 1 agent 1 & 2
    lines.append(
        {
            "core_round": 1,
            "core_agent_id": 1,
            "econ_prices": [1.0, 2.0],
            "spatial_in_marketplace": True,
        }
    )
    lines.append(
        {
            "core_round": 1,
            "core_agent_id": 2,
            "econ_prices": [1.0, 2.0],
            "spatial_in_marketplace": False,
        }
    )
    # Round 2 agent 1 & 2
    lines.append(
        {
            "core_round": 2,
            "core_agent_id": 1,
            "econ_prices": [1.0, 2.1],
            "spatial_in_marketplace": True,
        }
    )
    lines.append(
        {
            "core_round": 2,
            "core_agent_id": 2,
            "econ_prices": [1.0, 2.1],
            "spatial_in_marketplace": True,
        }
    )
    with open(path, "w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def test_replay_cli_headless(tmp_path: Path):
    log_file = tmp_path / "mini_log.jsonl"
    _write_minimal_log(log_file)
    # Invoke the script in replay mode (headless)
    cmd = [
        sys.executable,
        "scripts/run_simulation.py",
        "--replay",
        str(log_file),
        "--no-gui",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
    )
    assert proc.returncode == 0, proc.stderr
    # Expect completion message and frame count >= 2
    assert "Replay complete." in proc.stdout
    assert "Frames:" in proc.stdout
