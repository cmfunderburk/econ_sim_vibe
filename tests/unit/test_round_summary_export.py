from pathlib import Path
from typing import List

from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION


def _make_record(
    round_no: int,
    agent_id: int,
    prices: List[float],
    executed_net: List[float],
    in_market: bool = True,
) -> RoundLogRecord:
    return RoundLogRecord(
        core_schema_version=SCHEMA_VERSION,
        core_round=round_no,
        core_agent_id=agent_id,
        spatial_pos_x=agent_id,  # simple distinct positions
        spatial_pos_y=agent_id + 1,
        spatial_in_marketplace=in_market,
        econ_prices=prices,
        econ_executed_net=executed_net,
        econ_executed_buys=[max(0.0, x) for x in executed_net],
        econ_executed_sells=[max(0.0, -x) for x in executed_net],
        econ_unmet_buys=[0.0 for _ in executed_net],
        econ_unmet_sells=[0.0 for _ in executed_net],
        econ_fill_rate_buys=[1.0 for _ in executed_net],
        econ_fill_rate_sells=[1.0 for _ in executed_net],
        wealth_travel_cost=0.0,
    )


def test_round_summary_export_created(tmp_path: Path):
    logger = RunLogger(tmp_path, run_name="summary_run", prefer_parquet=False)
    # Two rounds, two agents
    logger.log_round(
        [
            _make_record(0, 1, [1.0, 2.0], [0.5, -0.5]),
            _make_record(0, 2, [1.0, 2.0], [-0.5, 0.5]),
        ]
    )
    logger.log_round(
        [
            _make_record(1, 1, [1.1, 1.9], [0.2, -0.2]),
            _make_record(1, 2, [1.1, 1.9], [-0.2, 0.2]),
        ]
    )
    out_path = logger.finalize()
    summary_path = tmp_path / "summary_run_round_summary.csv"
    assert summary_path.exists(), "Round summary CSV not written"
    content = summary_path.read_text().strip().splitlines()
    assert content, "Summary file empty"
    header = content[0].split(",")
    # Expected columns subset
    expected_cols = {"round", "agents", "participants", "prices", "executed_net"}
    assert expected_cols.issubset(set(header)), (
        f"Missing expected columns: {expected_cols - set(header)}"
    )
    # Two data lines
    assert len(content) == 1 + 2
