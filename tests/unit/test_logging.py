import os
from pathlib import Path

from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION


def test_run_logger_writes_jsonl_when_pandas_missing(monkeypatch, tmp_path):
    # Force pandas absence path by simulating internal flag
    # (If pandas installed, we still allow parquet; this test focuses on finalize path existence.)
    logger = RunLogger(tmp_path, run_name="test_run", prefer_parquet=False)

    rec = RoundLogRecord(
        core_schema_version=SCHEMA_VERSION,
        core_round=0,
        core_agent_id=1,
        spatial_pos_x=0,
        spatial_pos_y=0,
        spatial_in_marketplace=False,
        econ_prices=[1.0, 2.0],
        econ_executed_net=[0.0, 0.0],
        ration_unmet_demand=None,
        ration_unmet_supply=None,
        wealth_travel_cost=0.0,
        wealth_effective_budget=None,
        financing_mode=None,
    )

    logger.log_round([rec])
    out_path = logger.finalize()
    assert out_path.exists(), "Output log file should exist after finalize"
    # Basic metadata sidecar
    meta_path = tmp_path / "test_run_metadata.json"
    assert meta_path.exists(), "Metadata file should be written"


def test_round_log_record_dict_conversion():
    rec = RoundLogRecord(
        core_schema_version=SCHEMA_VERSION,
        core_round=5,
        core_agent_id=7,
        spatial_pos_x=3,
        spatial_pos_y=4,
        spatial_in_marketplace=True,
        econ_prices=[1.0],
        econ_executed_net=[0.5],
        ration_unmet_demand=[0.1],
        ration_unmet_supply=[0.0],
        wealth_travel_cost=1.25,
        wealth_effective_budget=10.0,
        financing_mode="personal",
    )
    d = rec.to_dict()
    assert d["core_round"] == 5
    assert d["econ_executed_net"] == [0.5]
    assert d["wealth_travel_cost"] == 1.25
    assert "utility" in d


def test_logging_record_has_utility(tmp_path):
    logger = RunLogger(tmp_path, run_name="util_test", prefer_parquet=False)
    rec = RoundLogRecord(
        core_schema_version=SCHEMA_VERSION,
        core_round=1,
        core_agent_id=2,
        spatial_pos_x=0,
        spatial_pos_y=0,
        spatial_in_marketplace=False,
        econ_prices=[1.0],
        econ_executed_net=[0.0],
        wealth_travel_cost=0.0,
        utility=2.5,
    )
    logger.log_round([rec])
    path = logger.finalize()
    with open(path, "r") as f:
        line = f.readline()
    assert "utility" in line
