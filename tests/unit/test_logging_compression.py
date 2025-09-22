from pathlib import Path
import tempfile
import json
import gzip

from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION


def test_run_logger_jsonl_compression():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RunLogger(
            Path(tmpdir), "test_run", prefer_parquet=False, compress=True
        )
        rec = RoundLogRecord(
            core_schema_version=SCHEMA_VERSION,
            core_round=0,
            core_agent_id=1,
            spatial_pos_x=0,
            spatial_pos_y=0,
            spatial_in_marketplace=False,
            econ_prices=[1.0, 2.0],
            econ_executed_net=[0.0, 0.0],
            financing_mode="PERSONAL",
        )
        logger.log_round([rec])
        path = logger.finalize()
        assert path.suffix == ".gz"
        assert path.exists()
        # Read compressed content
        with gzip.open(path, "rt") as f:
            line = f.readline()
            obj = json.loads(line)
            assert obj["core_agent_id"] == 1
            assert obj["financing_mode"] == "PERSONAL"
