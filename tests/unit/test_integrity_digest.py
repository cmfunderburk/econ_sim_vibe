from pathlib import Path
import json
from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION


def _record(round_: int, agent: int, prices, executed):
    return RoundLogRecord(
        core_schema_version=SCHEMA_VERSION,
        core_round=round_,
        core_agent_id=agent,
        spatial_pos_x=0,
        spatial_pos_y=0,
        spatial_in_marketplace=True,
        econ_prices=prices,
        econ_executed_net=executed,
    )


def test_integrity_digest_stable(tmp_path: Path):
    # Create logger and log rounds with shuffled agent order
    logger1 = RunLogger(tmp_path, "runA", prefer_parquet=False)
    logger1.log_round(
        [_record(1, 2, [1.0, 2.0], [0.0, 0.1]), _record(1, 1, [1.0, 2.0], [0.2, -0.1])]
    )
    logger1.log_round(
        [_record(2, 1, [1.0, 2.1], [0.1, 0.0]), _record(2, 2, [1.0, 2.1], [0.0, 0.0])]
    )
    path1 = logger1.finalize()
    integrity1 = json.loads((tmp_path / "runA_integrity.json").read_text())
    # Re-log in consistent agent order
    logger2 = RunLogger(tmp_path, "runB", prefer_parquet=False)
    logger2.log_round(
        [_record(1, 1, [1.0, 2.0], [0.2, -0.1]), _record(1, 2, [1.0, 2.0], [0.0, 0.1])]
    )
    logger2.log_round(
        [_record(2, 1, [1.0, 2.1], [0.1, 0.0]), _record(2, 2, [1.0, 2.1], [0.0, 0.0])]
    )
    path2 = logger2.finalize()
    integrity2 = json.loads((tmp_path / "runB_integrity.json").read_text())

    assert path1.exists() and path2.exists()
    # Digest should match irrespective of agent write order differences
    assert integrity1["digest"] == integrity2["digest"], (integrity1, integrity2)
    assert integrity1["lines"] == 2
    assert integrity1["digest_algorithm"] == "sha256"
