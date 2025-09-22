from pathlib import Path

from scripts.run_simulation import load_config
from src.core.simulation import initialize_runtime_state, run_round
from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION
from src.visualization.playback import LogReplayStream


def test_hud_fields_present_replay(tmp_path: Path):
    cfg = load_config(Path("config/small_market.yaml"))
    state = initialize_runtime_state(cfg)
    out_dir = tmp_path / "logs"
    out_dir.mkdir()
    logger = RunLogger(out_dir, "hud_replay_test")
    # Run a few rounds and log
    # Minimal per-round logging using required fields only
    for _ in range(3):
        run_round(state, cfg)
        snapshot = state.to_snapshot()
        records = []
        for agent in snapshot.agents:
            pos = state.grid.get_position(agent.agent_id)
            records.append(
                RoundLogRecord(
                    core_schema_version=SCHEMA_VERSION,
                    core_round=snapshot.round_number,
                    core_agent_id=agent.agent_id,
                    spatial_pos_x=pos.x,
                    spatial_pos_y=pos.y,
                    spatial_in_marketplace=state.grid.is_in_marketplace(pos),
                    econ_prices=list(snapshot.prices) if snapshot.prices is not None else [],  # type: ignore[arg-type]
                    econ_executed_net=[0.0 for _ in snapshot.prices] if snapshot.prices is not None else [],
                )
            )
        logger.log_round(records)
    logger.finalize()
    # Find log file
    log_files = list(out_dir.glob("*_round_log.jsonl"))
    assert log_files, "No round log produced"
    replay = LogReplayStream(log_files[0])
    f1 = replay.next_frame()
    assert f1 is not None
    assert f1.hud_round_digest is not None
    # Advance
    f2 = replay.next_frame()
    assert f2 is not None
    assert f2.hud_round_digest is not None
    # Convergence index should be between 0 and 1
    if f2.convergence_index is not None:
        assert 0.0 <= f2.convergence_index <= 1.0
