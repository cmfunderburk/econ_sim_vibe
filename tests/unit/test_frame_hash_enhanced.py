from pathlib import Path
import json
import tempfile

from scripts.run_simulation import run_simulation


def _load_rows(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    else:
        try:
            import pandas as pd  # type: ignore
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")  # type: ignore
        except Exception:
            return []


def test_frame_hash_presence_and_variation():
    config_path = Path("config/edgeworth.yaml")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = run_simulation(config_path, out_dir, seed_override=7)
        log_path = Path(result["structured_log_path"])
        rows = _load_rows(log_path)
        assert rows, "No log rows produced"
        # All rows have hash
        hashes = [r["core_frame_hash"] for r in rows]
        assert all(hashes), "Missing frame hashes"
        # Expect multiple distinct hashes across different agents/rounds
        assert len(set(hashes)) > 1, "Hashes are not varying as expected"
