import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from scripts.run_simulation import run_simulation

Row = Dict[str, Any]


def _load_rows(path: Path) -> List[Row]:
    """Load structured log rows from JSONL or Parquet path.

    Returns a list of dict rows. Falls back to empty list on failure.
    """
    if path.suffix == ".jsonl":
        rows: List[Row] = []
        with path.open() as f:  # type: ignore[call-arg]
            for line in f:
                rows.append(json.loads(line))
        return rows
    # Parquet branch
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        # Orient records -> List[Dict[str, Any]]
        recs: List[Row] = df.to_dict(orient="records")  # type: ignore[assignment]
        return recs
    except Exception:
        return []


def _build_digest(rows: Sequence[Row]) -> str:
    by_round: Dict[int, List[Row]] = {}
    for r in rows:
        rd = int(r["core_round"])  # robust cast
        by_round.setdefault(rd, []).append(r)
    parts: List[str] = []
    for rd in sorted(by_round.keys()):
        recs = by_round[rd]
        # Participation count (bool field may be stored as int/bool)
        participants = sum(1 for rr in recs if bool(rr.get("spatial_in_marketplace", False)))
        # Canonical prices (first row with non-empty econ_prices list)
        prices_list: Iterable[float] = ()
        for rr in recs:
            prices_candidate = rr.get("econ_prices")
            if prices_candidate:
                prices_list = prices_candidate
                break
        price_str = ";".join(f"{float(p):.6g}" for p in prices_list)
        # Agent hashes sorted by agent id
        agent_hash_pairs: List[Tuple[int, str]] = sorted(
            (int(rr["core_agent_id"]), str(rr["core_frame_hash"])) for rr in recs
        )
        agent_hash_str = ",".join(f"{aid}:{h}" for aid, h in agent_hash_pairs)
        parts.append(f"{rd}|P={participants}|{price_str}|{agent_hash_str}")
    return "\n".join(parts)


def test_golden_replay_digest_determinism():
    """Golden digest test: run deterministic small simulation twice and compare digests.

    Digest composition per round:
      round | P=participants | price_vector | aid:hash,... (agent id order ascending)
    """
    config_path = Path("config/edgeworth.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir1 = Path(tmpdir) / "run1"
        out_dir2 = Path(tmpdir) / "run2"
        out_dir1.mkdir(parents=True)
        out_dir2.mkdir(parents=True)

        r1 = run_simulation(config_path, out_dir1, seed_override=101)
        r2 = run_simulation(config_path, out_dir2, seed_override=101)

        rows1 = _load_rows(Path(r1["structured_log_path"]))
        rows2 = _load_rows(Path(r2["structured_log_path"]))
        assert rows1 and rows2, "No rows generated in one of the runs"

        digest1 = _build_digest(rows1)
        digest2 = _build_digest(rows2)
        assert digest1 == digest2, "Golden digest mismatch across identical runs"

        # Basic sanity: multiple lines, at least one hash per line
        lines = digest1.split("\n")
        assert len(lines) >= 1
        assert all(":" in ln for ln in lines)
