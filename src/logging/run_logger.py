"""Run logging utilities for economic simulation.

This module provides a lightweight, dependency-tolerant structured logging
layer that records per-round, per-agent economic and spatial state in a
tabular format suitable for downstream analysis. The initial implementation
targets a minimal viable schema aligned with the project SPECIFICATION while
remaining easy to extend (schema versioning + additive columns).

Design Principles:
1. Append-only row model (one row per agent per round)
2. Deterministic field order for stable parquet/JSON encodings
3. Graceful degradation: prefer Parquet if pandas+pyarrow available, else
   fall back to JSON Lines (.jsonl). CSV avoided to preserve numeric fidelity.
4. Explicit SCHEMA_VERSION to coordinate breaking changes.
5. No heavy coupling to simulation internals: caller passes already-computed
   economic values; logger focuses only on I/O and light validation.

Extension Hooks (future):
 - financing_mode specific fields (net_financing_gap)
 - equivalent variation trajectory (ev_round)
 - carry-over queues (unexecuted orders) once promoted to first-class

Author: AI Assistant
Date: 2025-09-21
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
import gzip
import json
import time
import logging

try:  # Optional heavy deps
    import pandas as _pd  # type: ignore
    _PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - environment without pandas
    _PANDAS_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Increment only on breaking (non-backward-compatible) changes
SCHEMA_VERSION = "1.0.0"
# Schema Evolution Guidance:
# - Bump minor version for additive, backward-compatible column additions.
# - Bump major version for breaking changes (renames/removals/semantic shifts).
# - When bumping, update any schema guard tests (test_logging_schema.py) and
#   ensure migration or backward compatibility strategy is documented.


@dataclass
class RoundLogRecord:
    """Single per-agent, per-round observation.

    Field grouping (prefix rationale):
    core_*      : round + identity + schema metadata
    spatial_*   : spatial state & marketplace access
    econ_*      : prices & executed net trades (buys - sells)
    ration_*    : unmet demand/supply (post-clearing aggregates where defined)
    wealth_*    : travel cost & effective budget snapshot

    Notes:
    - econ_executed_net is a list[float] length G (positive = net buyer)
    - econ_prices is identical across agents within a round (redundant but
      preserves row self-containment) and may be empty if no pricing round.
    - ration_unmet_demand / ration_unmet_supply are per-good MARKET totals
      duplicated per agent for easier row-based filtering; None if no pricing.
    - wealth_travel_cost is cumulative travel cost at the END of the round.
    - financing_mode reserved for future multi-mode comparisons.
    """

    core_schema_version: str
    core_round: int
    core_agent_id: int

    spatial_pos_x: int
    spatial_pos_y: int
    spatial_in_marketplace: bool

    econ_prices: List[float]  # length G or [] if no pricing
    econ_executed_net: List[float]  # length G or [] if no trades/prices

    ration_unmet_demand: Optional[List[float]] = None
    ration_unmet_supply: Optional[List[float]] = None

    wealth_travel_cost: float = 0.0
    wealth_effective_budget: Optional[float] = None  # adjusted wealth if available

    utility: Optional[float] = None  # agent utility (total_endowment bundle)

    financing_mode: Optional[str] = None
    # Future: ev_round, liquidity_gap, etc.

    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class RunLogger:
    """Buffered run logger for structured economic simulation output.

    Usage:
        logger = RunLogger(output_dir, run_name)
        logger.log_round(batch_of_records)  # list[RoundLogRecord]
        ... after final round ...
        path = logger.finalize()  # returns written file path

    Buffering Strategy:
        - Accumulate records in memory until finalize(). For modest scale
          (<= 100 agents * <= 1000 rounds) this is acceptable (< ~100k rows).
        - If memory growth becomes a concern, a future enhancement can spill
          to temporary JSONL shards earlier.
    """

    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        prefer_parquet: bool = True,
        compress: bool = False,
        flush_interval: int = 0,
    ):
        self.output_dir = output_dir
        self.run_name = run_name
        self.prefer_parquet = prefer_parquet
        self.compress = compress
        self._buffer: List[RoundLogRecord] = []
        self._flushed_partial: List[RoundLogRecord] = []  # holds flushed rows before finalize
        self._finalized = False
        self._flush_interval = flush_interval if flush_interval > 0 else 0
        self._rounds_logged = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            f"Initialized RunLogger(run_name={run_name}, prefer_parquet={prefer_parquet}, compress={compress}, pandas={_PANDAS_AVAILABLE})"
        )

    def log_round(self, records: Iterable[RoundLogRecord]) -> None:
        if self._finalized:
            raise RuntimeError("Cannot log after finalize()")
        count = 0
        for r in records:
            if r.core_schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"Schema version mismatch: record={r.core_schema_version} expected={SCHEMA_VERSION}"
                )
            self._buffer.append(r)
            count += 1
        LOGGER.debug(f"Buffered {count} records (total={len(self._buffer)})")
        self._rounds_logged += 1
        if self._flush_interval and (self._rounds_logged % self._flush_interval == 0):
            try:
                self.flush()
            except Exception as e:  # pragma: no cover - defensive
                LOGGER.warning(f"Periodic flush failed: {e}")

    def flush(self) -> None:
        """Flush buffered records to an intermediate JSONL shard (pre-finalize).

        This preserves durability in long runs. Flushed shards are merged on finalize.
        Parquet is deferred to finalization for schema cohesiveness.
        """
        if self._finalized:
            raise RuntimeError("Cannot flush after finalize()")
        if not self._buffer:
            return
        shard_name = f"{self.run_name}_partial_{len(self._flushed_partial)}.jsonl"
        shard_path = self.output_dir / shard_name
        with shard_path.open("a") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec.to_dict()) + "\n")
        self._flushed_partial.extend(self._buffer)
        self._buffer.clear()
        LOGGER.info(f"Flushed intermediate shard: {shard_path}")

    def _write_jsonl(self, path: Path) -> Path:
        if self.compress:
            gz_path = path.with_suffix(path.suffix + '.gz')
            with gzip.open(gz_path, 'wt') as f:  # type: ignore[arg-type]
                for rec in self._buffer:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            return gz_path
        else:
            with path.open("w") as f:
                for rec in self._buffer:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            return path

    def _write_parquet(self, path: Path) -> Path:
        if not _PANDAS_AVAILABLE:
            raise RuntimeError("pandas not available for parquet write")
        import pandas as pd  # local alias

        df = pd.DataFrame([rec.to_dict() for rec in self._buffer])
        try:
            if self.compress:
                # Rely on pyarrow/fastparquet engine compression param if available; fallback to gzip suffix rename if engine missing
                try:
                    df.to_parquet(path.with_suffix(path.suffix + '.gz'), index=False, compression='gzip')
                    return path.with_suffix(path.suffix + '.gz')
                except Exception:
                    # Attempt uncompressed then gzip file manually
                    df.to_parquet(path, index=False)
                    raw = path.read_bytes()
                    gz_path = path.with_suffix(path.suffix + '.gz')
                    with gzip.open(gz_path, 'wb') as gz:
                        gz.write(raw)
                    path.unlink(missing_ok=True)
                    return gz_path
            else:
                df.to_parquet(path, index=False)
        except Exception as e:  # fallback to jsonl if pyarrow/fastparquet missing
            LOGGER.warning(f"Parquet write failed ({e}); falling back to JSONL")
            return self._write_jsonl(path.with_suffix('.jsonl'))
        return path

    def finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("RunLogger already finalized")
        self._finalized = True
        # Bring any partially flushed buffer into a combined list
        if not self._buffer and not self._flushed_partial:
            raise RuntimeError("No records buffered; nothing to write")

        # Combine flushed partials + current buffer (do not re-write partial shards individually)
        all_records = self._flushed_partial + self._buffer
        self._buffer = all_records  # reuse existing write path

        base = self.output_dir / f"{self.run_name}_round_log"
        if self.prefer_parquet and _PANDAS_AVAILABLE:
            out_path = self._write_parquet(base.with_suffix('.parquet'))
        else:
            out_path = self._write_jsonl(base.with_suffix('.jsonl'))

        # Write sidecar metadata
        meta = {
            "schema_version": SCHEMA_VERSION,
            "rows": len(self._buffer),
            "run_name": self.run_name,
            "format": out_path.suffix.lstrip('.'),
            "compressed": self.compress,
            "partial_flushes": 1 if self._flushed_partial else 0,
            "flush_interval": self._flush_interval,
        }
        with (self.output_dir / f"{self.run_name}_metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

        LOGGER.info(f"Wrote {len(self._buffer)} records to {out_path}")
        return out_path


__all__ = [
    "SCHEMA_VERSION",
    "RoundLogRecord",
    "RunLogger",
]
