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
import hashlib
import csv

try:  # Optional heavy deps
    # import pandas as _pd  # type: ignore
    _pandas_available = True
except Exception:  # pragma: no cover - environment without pandas
    _pandas_available = False

LOGGER = logging.getLogger(__name__)

# Increment only on breaking (non-backward-compatible) changes
SCHEMA_VERSION = (
    "1.3.0"  # Added spatial distance fidelity columns (Step D) – additive
)
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

    # New enriched diagnostics (schema 1.1.0, additive):
    econ_requested_buys: Optional[List[float]] = (
        None  # raw requested buy quantities (length G) per agent
    )
    econ_requested_sells: Optional[List[float]] = (
        None  # raw requested sell quantities (length G) per agent
    )
    econ_executed_buys: Optional[List[float]] = (
        None  # executed buy quantities (length G)
    )
    econ_executed_sells: Optional[List[float]] = (
        None  # executed sell quantities (length G)
    )
    econ_unmet_buys: Optional[List[float]] = (
        None  # unmet portion of buy orders (length G)
    )
    econ_unmet_sells: Optional[List[float]] = (
        None  # unmet portion of sell offers (length G)
    )
    econ_fill_rate_buys: Optional[List[float]] = None  # per-good buy fill rates (0-1)
    econ_fill_rate_sells: Optional[List[float]] = None  # per-good sell fill rates (0-1)

    ration_unmet_demand: Optional[List[float]] = None
    ration_unmet_supply: Optional[List[float]] = None

    wealth_travel_cost: float = 0.0
    wealth_effective_budget: Optional[float] = None  # adjusted wealth if available

    utility: Optional[float] = None  # agent utility (total_endowment bundle)

    financing_mode: Optional[str] = None
    core_frame_hash: Optional[str] = None  # stable hash for replay verification (v1.2.0)
    # New spatial fidelity fields (Step D additive): placed after existing required fields
    spatial_distance_to_market: Optional[int] = None  # Manhattan distance this round
    spatial_max_distance_round: Optional[int] = None  # max distance among all agents this round
    spatial_avg_distance_round: Optional[float] = None  # average distance among all agents this round
    spatial_initial_max_distance: Optional[int] = None  # baseline max distance at t=0
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
        self._flushed_partial: List[
            RoundLogRecord
        ] = []  # holds flushed rows before finalize
        self._finalized = False
        self._flush_interval = flush_interval if flush_interval > 0 else 0
        self._rounds_logged = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            f"Initialized RunLogger(run_name={run_name}, prefer_parquet={prefer_parquet}, compress={compress}, pandas={_pandas_available})"
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
            # Ensure frame hash present (defensive). If absent, compute deterministic
            # multi-field digest capturing key economic + spatial signals.
            if r.core_frame_hash is None:
                price_part = ";".join(f"{p:.10g}" for p in r.econ_prices)
                executed_part = ";".join(
                    f"{x:.10g}" for x in (r.econ_executed_net or [])
                )
                requested_buy_part = ";".join(
                    f"{x:.6g}" for x in (r.econ_requested_buys or [])
                )
                pos_part = f"{r.spatial_pos_x},{r.spatial_pos_y},{int(r.spatial_in_marketplace)}"
                base = (
                    f"r={r.core_round};a={r.core_agent_id};p=[{price_part}];"
                    f"net=[{executed_part}];reqB=[{requested_buy_part}];pos={pos_part}"
                )
                r.core_frame_hash = hashlib.blake2b(
                    base.encode("utf-8"), digest_size=16
                ).hexdigest()
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
            gz_path = path.with_suffix(path.suffix + ".gz")
            with gzip.open(gz_path, "wt") as f:  # type: ignore[arg-type]
                for rec in self._buffer:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            return gz_path
        else:
            with path.open("w") as f:
                for rec in self._buffer:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            return path

    def _write_parquet(self, path: Path) -> Path:
        if not _pandas_available:
            raise RuntimeError("pandas not available for parquet write")
        import pandas as pd  # local alias

        df = pd.DataFrame([rec.to_dict() for rec in self._buffer])
        try:
            if self.compress:
                # Rely on pyarrow/fastparquet engine compression param if available; fallback to gzip suffix rename if engine missing
                try:
                    df.to_parquet(
                        path.with_suffix(path.suffix + ".gz"),
                        index=False,
                        compression="gzip",
                    )
                    return path.with_suffix(path.suffix + ".gz")
                except Exception:
                    # Attempt uncompressed then gzip file manually
                    df.to_parquet(path, index=False)
                    raw = path.read_bytes()
                    gz_path = path.with_suffix(path.suffix + ".gz")
                    with gzip.open(gz_path, "wb") as gz:
                        gz.write(raw)
                    path.unlink(missing_ok=True)
                    return gz_path
            else:
                df.to_parquet(path, index=False)
        except Exception as e:  # fallback to jsonl if pyarrow/fastparquet missing
            LOGGER.warning(f"Parquet write failed ({e}); falling back to JSONL")
            return self._write_jsonl(path.with_suffix(".jsonl"))
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
        if self.prefer_parquet and _pandas_available:
            out_path = self._write_parquet(base.with_suffix(".parquet"))
        else:
            out_path = self._write_jsonl(base.with_suffix(".jsonl"))

        # Write sidecar metadata
        meta: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rows": len(self._buffer),
            "run_name": self.run_name,
            "format": out_path.suffix.lstrip("."),
            "compressed": self.compress,
            "partial_flushes": 1 if self._flushed_partial else 0,
            "flush_interval": self._flush_interval,
        }
        with (self.output_dir / f"{self.run_name}_metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

        # Minimal integrity digest (NOT per-round full hash):
        # Construct deterministic sequence: for each round r ascending ->
        #   append r, then price vector (joined), then sorted agent id list.
        # This detects drift in price path or participant identity sets.
        try:
            grouped: dict[int, list[RoundLogRecord]] = {}
            for rec in self._buffer:
                grouped.setdefault(rec.core_round, []).append(rec)
            digest_parts: list[str] = []
            for r in sorted(grouped.keys()):
                recs = grouped[r]
                # Extract canonical prices (first non-empty econ_prices if any)
                prices: list[float] = []
                for rec in recs:
                    if rec.econ_prices:
                        prices = rec.econ_prices
                        break
                price_str = ";".join(f"{p:.10g}" for p in prices)
                agent_ids = sorted(rec.core_agent_id for rec in recs)
                agent_str = ",".join(str(a) for a in agent_ids)
                digest_parts.append(f"{r}|{price_str}|{agent_str}")
            full_str = "\n".join(digest_parts)
            sha = hashlib.sha256(full_str.encode("utf-8")).hexdigest()
            # Attempt to load geometry sidecar (same directory, naming convention: run_name_geometry.json)
            geometry_hash = None
            geom_path = self.output_dir / f"{self.run_name}_geometry.json"
            if geom_path.exists():
                try:
                    geom_raw = geom_path.read_bytes()
                    geometry_hash = hashlib.sha256(geom_raw).hexdigest()
                except Exception:
                    geometry_hash = None
            integrity: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "digest_algorithm": "sha256",
                "digest_fields": ["round", "econ_prices", "agent_id_set"],
                "lines": len(digest_parts),
                "digest": sha,
                "note": "Minimal integrity digest; not a full per-round frame hash (future upgrade).",
                **({"geometry_hash": geometry_hash} if geometry_hash else {}),
            }
            with (self.output_dir / f"{self.run_name}_integrity.json").open(
                "w"
            ) as f_int:
                json.dump(integrity, f_int, indent=2)
        except Exception as e:  # pragma: no cover - best effort
            LOGGER.warning(f"Failed to write integrity digest: {e}")

        # ---------- Round Summary Export (CSV) ----------
        # Lightweight per-round aggregation for fast plotting / inspection.
        # Columns:
        # round, agents, participants, prices, executed_net, executed_buys,
        # executed_sells, unmet_buys, unmet_sells, avg_buy_fill, avg_sell_fill
        # Vector columns serialized as semicolon-separated numeric strings.
        try:
            summary_path = self.output_dir / f"{self.run_name}_round_summary.csv"
            # Build grouping (reuse earlier grouping if kept; recompute for clarity)
            grouped: dict[int, list[RoundLogRecord]] = {}
            for rec in self._buffer:
                grouped.setdefault(rec.core_round, []).append(rec)

            def _sum_vectors(vectors: list[list[float]]) -> list[float]:
                if not vectors:
                    return []
                length = len(vectors[0])
                acc = [0.0] * length
                for v in vectors:
                    if len(v) != length:
                        # Skip malformed length (defensive)
                        continue
                    for i, val in enumerate(v):
                        acc[i] += val
                return acc

            def _avg_vectors(vectors: list[list[float]]) -> list[float]:
                if not vectors:
                    return []
                summed = _sum_vectors(vectors)
                n = len(vectors)
                if n == 0:
                    return []
                return [x / n for x in summed]

            def _fmt(vec: list[float]) -> str:
                if not vec:
                    return ""
                return ";".join(f"{x:.10g}" for x in vec)

            with summary_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "round",
                        "agents",
                        "participants",
                        "prices",
                        "executed_net",
                        "executed_buys",
                        "executed_sells",
                        "unmet_buys",
                        "unmet_sells",
                        "avg_buy_fill",
                        "avg_sell_fill",
                    ]
                )
                for r in sorted(grouped.keys()):
                    recs = grouped[r]
                    n_agents_round = len(recs)
                    participants = sum(1 for rec in recs if rec.spatial_in_marketplace)
                    # Prices: first non-empty
                    prices: list[float] = []
                    for rec in recs:
                        if rec.econ_prices:
                            prices = rec.econ_prices
                            break
                    # Collect vectors for aggregation (skip None)
                    exec_net = _sum_vectors(
                        [rec.econ_executed_net for rec in recs if rec.econ_executed_net]
                    )
                    exec_buys = _sum_vectors(
                        [
                            rec.econ_executed_buys
                            for rec in recs
                            if rec.econ_executed_buys
                        ]
                    )  # type: ignore[arg-type]
                    exec_sells = _sum_vectors(
                        [
                            rec.econ_executed_sells
                            for rec in recs
                            if rec.econ_executed_sells
                        ]
                    )  # type: ignore[arg-type]
                    unmet_buys = _sum_vectors(
                        [rec.econ_unmet_buys for rec in recs if rec.econ_unmet_buys]
                    )  # type: ignore[arg-type]
                    unmet_sells = _sum_vectors(
                        [rec.econ_unmet_sells for rec in recs if rec.econ_unmet_sells]
                    )  # type: ignore[arg-type]
                    avg_buy_fill = _avg_vectors(
                        [
                            rec.econ_fill_rate_buys
                            for rec in recs
                            if rec.econ_fill_rate_buys
                        ]
                    )  # type: ignore[arg-type]
                    avg_sell_fill = _avg_vectors(
                        [
                            rec.econ_fill_rate_sells
                            for rec in recs
                            if rec.econ_fill_rate_sells
                        ]
                    )  # type: ignore[arg-type]

                    writer.writerow(
                        [
                            r,
                            n_agents_round,
                            participants,
                            _fmt(prices),
                            _fmt(exec_net),
                            _fmt(exec_buys),
                            _fmt(exec_sells),
                            _fmt(unmet_buys),
                            _fmt(unmet_sells),
                            _fmt(avg_buy_fill),
                            _fmt(avg_sell_fill),
                        ]
                    )
            # Enhance metadata with reference
            try:
                meta_path = self.output_dir / f"{self.run_name}_metadata.json"
                if meta_path.exists():
                    meta_json = json.loads(meta_path.read_text())
                    meta_json["round_summary_file"] = summary_path.name
                    meta_path.write_text(json.dumps(meta_json, indent=2))
            except Exception as e:  # pragma: no cover - best effort
                LOGGER.warning(f"Failed to augment metadata with summary file: {e}")
        except Exception as e:  # pragma: no cover - best effort
            LOGGER.warning(f"Failed to write round summary CSV: {e}")

        LOGGER.info(f"Wrote {len(self._buffer)} records to {out_path}")
        return out_path


__all__ = [
    "SCHEMA_VERSION",
    "RoundLogRecord",
    "RunLogger",
]
