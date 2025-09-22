"""Playback & control abstractions for visualization.

This module decouples simulation advancement (frame production) from
rendering cadence and user controls. It enables interactive features:
  - Play / Pause
  - Single-step advance
  - Adjustable speed (rounds per second)
  - (Future) Seeking / scrubbing for replay streams

Two primary components:
  1. FrameStream protocol: produces successive FrameData instances.
  2. PlaybackController: manages timing & control state for a FrameStream.

Current scope only implements LiveSimulationStream (drives the live
simulation by calling `run_round` then building a frame). A LogReplayStream
will be added subsequently (next todo) to read from structured logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Any, cast, Tuple, TypedDict
import time
import json
from pathlib import Path

try:  # Optional pandas for parquet
    import pandas as _pd  # type: ignore

    HAS_PANDAS = True
except Exception:  # pragma: no cover
    HAS_PANDAS = False

from .frame_data import FrameData, build_frame, AgentFrame
from .metrics import spatial_convergence_index, frame_hash
from src.core.simulation import RuntimeSimulationState, SimulationConfig, run_round


class FrameStream(Protocol):  # pragma: no cover - structural typing
    def next_frame(self) -> Optional[FrameData]: ...
    def seek(self, round_number: int) -> None: ...  # optional for replay
    def reset(self) -> None: ...


@dataclass
class LiveSimulationStream:
    """Frame stream that advances an in-memory simulation state."""

    state: RuntimeSimulationState
    config: SimulationConfig

    def next_frame(self) -> Optional[FrameData]:
        # Advance simulation one round IF not past max_rounds
        if self.state.round >= self.config.max_rounds:
            return None
        self.state = run_round(self.state, self.config)
        return build_frame(self.state, self.config)

    # The following operations are not meaningful for live mode yet
    def seek(self, round_number: int) -> None:  # pragma: no cover - future extension
        raise NotImplementedError("Seek not supported for live simulation stream")

    def reset(self) -> None:  # pragma: no cover - could reinitialize in future
        raise NotImplementedError("Reset not implemented for live simulation stream")


@dataclass
class PlaybackController:
    """Manages playback controls for a FrameStream.

    Timing model: fixed target rounds-per-second. The controller accumulates
    wall-clock time and requests a new frame when the elapsed time exceeds
    the current frame interval OR when in single-step mode.
    """

    stream: FrameStream
    rounds_per_second: float = 5.0
    is_playing: bool = True
    last_tick: float = 0.0
    _last_frame: Optional[FrameData] = None

    def __post_init__(self) -> None:
        self.last_tick = time.perf_counter()
        self._clamp_speed()

    def _clamp_speed(self) -> None:
        # Prevent nonsensical or runaway speed values
        if self.rounds_per_second < 0.25:
            self.rounds_per_second = 0.25
        if self.rounds_per_second > 60.0:
            self.rounds_per_second = 60.0

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.rounds_per_second

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_tick = time.perf_counter()

    def step_once(self) -> Optional[FrameData]:
        """Advance exactly one frame and pause."""
        self.is_playing = False
        return self._advance_manual(direction=1)

    def set_speed(self, new_rps: float) -> None:
        self.rounds_per_second = new_rps
        self._clamp_speed()

    def speed_up(self) -> None:
        self.set_speed(self.rounds_per_second * 2.0)

    def slow_down(self) -> None:
        self.set_speed(self.rounds_per_second / 2.0)

    def update(self) -> Optional[FrameData]:
        """Produce the next frame when playing according to timing."""
        if not self.is_playing:
            return None
        now = time.perf_counter()
        elapsed = now - self.last_tick
        if elapsed < self.frame_interval:
            return None

        self.last_tick = now
        frame = self.stream.next_frame()
        if frame is None:
            self.is_playing = False
            return None
        self._last_frame = frame
        return frame

    def step_back(self) -> Optional[FrameData]:
        """Move one frame backward if stream supports it."""
        if not hasattr(self.stream, "prev_frame"):
            return None
        self.is_playing = False
        return self._advance_manual(direction=-1)

    def jump(self, delta: int) -> Optional[FrameData]:
        """Jump by delta rounds (positive or negative)."""
        if delta == 0:
            return None
        current = self.current_round or 0
        target = current + delta
        return self.goto(target)

    def goto(self, round_number: int) -> Optional[FrameData]:
        """Seek to an absolute round if stream supports it."""
        if not hasattr(self.stream, "seek"):
            return None
        self.is_playing = False
        try:
            target_index = max(round_number, 0)
            max_round = getattr(self.stream, "_max_round", None)
            if isinstance(max_round, int) and max_round > 0:
                if target_index > max_round:
                    target_index = max_round
            seek_to = target_index - 1 if target_index > 0 else 0
            self.stream.seek(seek_to)
            frame = self.stream.next_frame()
        except Exception:
            frame = None
        if frame is not None:
            self._last_frame = frame
            self.last_tick = time.perf_counter()
        return frame

    @property
    def current_round(self) -> Optional[int]:
        return self._last_frame.round if self._last_frame is not None else None

    def _advance_manual(self, direction: int) -> Optional[FrameData]:
        frame: Optional[FrameData]
        if direction > 0:
            frame = self.stream.next_frame()
        elif direction < 0 and hasattr(self.stream, "prev_frame"):
            try:
                frame = getattr(self.stream, "prev_frame")()
            except Exception:
                frame = None
        else:
            frame = None
        if frame is not None:
            self._last_frame = frame
            self.last_tick = time.perf_counter()
        return frame


# ----------------------------- Replay Stream ---------------------------------
@dataclass
class _MarketplaceRect(TypedDict):
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    width: int
    height: int


class _GeometrySidecar(TypedDict, total=False):
    schema: str
    grid: Dict[str, int]
    marketplace: _MarketplaceRect
    movement_policy: str
    seed: int


@dataclass
class LogReplayStream:
    """Frame stream that replays previously logged simulation rounds.

    Supports JSONL (default) and parquet logs produced by RunLogger. Each
    record corresponds to one agent & round; we group by round to build
    FrameData snapshots. Economic overlays are *not* currently reconstructed
    beyond prices & participation counts (future: integrate unmet metrics if
    needed for replay overlays).
    """

    log_path: Path
    geometry_path: Optional[Path] = None
    _records: List[Dict[str, Any]] | None = None
    _round_index: int = 0  # last produced round (0 means none yet)
    _max_round: int = 0
    _round_map: Dict[int, Tuple[int, int]] | None = (
        None  # round -> (start_idx, end_idx)
    )
    _sorted_rounds: List[int] | None = None

    def _load(self) -> None:
        if self._records is not None:
            return
        suffix = self.log_path.suffix.lower()
        rows: List[Dict[str, Any]] = []
        if suffix.endswith("jsonl") or suffix == ".jsonl":
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        elif suffix.endswith("parquet") and HAS_PANDAS:  # pragma: no cover
            df = _pd.read_parquet(self.log_path)  # type: ignore
            rows = df.to_dict(orient="records")  # type: ignore
        else:
            raise ValueError(
                f"Unsupported log format or missing dependency: {self.log_path}"
            )
        rows.sort(key=lambda r: (r.get("core_round", 0), r.get("core_agent_id", 0)))
        self._records = rows
        self._max_round = 0 if not rows else max(r.get("core_round", 0) for r in rows)
        self._round_index = 0
        round_map: Dict[int, Tuple[int, int]] = {}
        current_round: Optional[int] = None
        start_idx = 0
        for idx, rec in enumerate(rows):
            rnum = rec.get("core_round", 0)
            if current_round is None:
                current_round = rnum
                start_idx = idx
            elif rnum != current_round:
                round_map[current_round] = (start_idx, idx)
                current_round = rnum
                start_idx = idx
        if current_round is not None:
            round_map[current_round] = (start_idx, len(rows))
        self._round_map = round_map
        self._sorted_rounds = sorted(round_map.keys())
        # Geometry sidecar discovery
        self._geometry_cache: Optional[_GeometrySidecar] = None
        if self.geometry_path is None:
            stem = self.log_path.stem
            base = stem.replace("_round_log", "")
            candidate = self.log_path.parent / f"{base}_geometry.json"
            if candidate.exists():
                self.geometry_path = candidate
        if self.geometry_path and self.geometry_path.exists():
            try:
                raw_any: Any = json.loads(self.geometry_path.read_text())
                if (
                    isinstance(raw_any, dict)
                    and "schema" in raw_any
                    and str(raw_any["schema"]) == "geometry.sidecar.v1"
                ):
                    raw_geom: Dict[str, Any] = raw_any  # refine
                    self._geometry_cache = cast(_GeometrySidecar, raw_geom)
            except Exception:
                self._geometry_cache = None

    # --- Geometry helpers ---
    def _compute_distance(self, x: int, y: int) -> int:
        if self._geometry_cache is None:
            return 0
        mkt = self._geometry_cache.get("marketplace")
        if mkt is None:
            return 0
        # Live mode uses (x0,y0,width,height) rectangle; sidecar stores min/max and width/height.
        x_min, x_max = int(mkt["x_min"]), int(mkt["x_max"])
        y_min, y_max = int(mkt["y_min"]), int(mkt["y_max"])
        # Convert to origin + width/height matching frame_data logic
        width = int(mkt.get("width", x_max - x_min + 1))
        height = int(mkt.get("height", y_max - y_min + 1))
        # Replicate _manhattan_distance_to_rect semantics (rectangle inclusive of bounds)
        rx, ry = x_min, y_min
        rw, rh = width, height
        # Distance 0 if inside
        inside_x = rx <= x < rx + rw
        inside_y = ry <= y < ry + rh
        if inside_x and inside_y:
            return 0
        dx = 0 if inside_x else (rx - x if x < rx else x - (rx + rw - 1))
        dy = 0 if inside_y else (ry - y if y < ry else y - (ry + rh - 1))
        return abs(dx) + abs(dy)

    def _group_round(self, round_number: int) -> List[Dict[str, Any]]:
        assert self._records is not None and self._round_map is not None
        span = self._round_map.get(round_number)
        if span is None:
            return []
        s, e = span
        return self._records[s:e]

    # --- Public helpers ---
    def frame_at(self, round_number: int) -> Optional[FrameData]:
        """Return FrameData for a specific round without changing sequential state."""
        self._load()
        if round_number <= 0 or round_number > self._max_round:
            return None
        rows = self._group_round(round_number)
        if not rows:
            return None
        return self._build_frame_from_rows(round_number, rows)

    def prev_frame(self) -> Optional[FrameData]:
        """Move one round backward (if possible) and return that frame."""
        self._load()
        if self._round_index <= 1:
            # Either at beginning or no frames produced yet -> produce first frame via next_frame instead
            return self.frame_at(1)
        target = self._round_index - 1
        rows = self._group_round(target)
        if not rows:
            return None
        self._round_index = target
        return self._build_frame_from_rows(target, rows)

    def _build_frame_from_rows(
        self, target_round: int, round_rows: List[Dict[str, Any]]
    ) -> FrameData:
        # Derive basic frame attributes (prices assumed identical across rows)
        # Extract prices without triggering ambiguous truth evaluation on numpy arrays
        raw_prices_any: Any = round_rows[0].get("econ_prices", [])
        try:  # Normalize numpy array to list if present
            import numpy as _np  # type: ignore
            if isinstance(raw_prices_any, _np.ndarray):  # type: ignore[attr-defined]
                raw_prices_any = raw_prices_any.tolist()
        except Exception:  # pragma: no cover - defensive
            pass
        prices: List[float] = []
        if isinstance(raw_prices_any, list):
            for _p in raw_prices_any:  # type: ignore[assignment]
                if isinstance(_p, (int, float, str)):
                    try:
                        prices.append(float(_p))
                    except Exception:
                        continue
        part = sum(1 for r in round_rows if r.get("spatial_in_marketplace"))
        total_agents = len(round_rows)
        agent_rows: List[Tuple[int, int, int, bool]] = []
        for row in round_rows:
            try:
                ax = int(row.get("spatial_pos_x", 0))
                ay = int(row.get("spatial_pos_y", 0))
                aid = int(row.get("core_agent_id", 0))
                in_mkt = bool(row.get("spatial_in_marketplace", False))
                agent_rows.append((aid, ax, ay, in_mkt))
            except Exception:
                continue
        # Derive geometry metrics (use sidecar if present)
        if (
            self._geometry_cache is not None
            and "marketplace" in self._geometry_cache
            and "grid" in self._geometry_cache
        ):
            mkt = self._geometry_cache["marketplace"]
            grid_info = self._geometry_cache["grid"]
            # Defensive conversions
            grid_w = int(grid_info.get("width", 0))
            grid_h = int(grid_info.get("height", 0))
            frame_market_x0 = int(mkt.get("x_min", 0))
            frame_market_y0 = int(mkt.get("y_min", 0))
            frame_market_w = int(mkt.get("width", 0))
            frame_market_h = int(mkt.get("height", 0))
            distance_fn = self._compute_distance
        else:
            xs = [ax for _, ax, _, _ in agent_rows]
            ys = [ay for _, _, ay, _ in agent_rows]
            grid_w = max(xs) + 1 if xs else 1
            grid_h = max(ys) + 1 if ys else 1
            in_market_positions = [
                (ax, ay) for _, ax, ay, in_m in agent_rows if in_m
            ]
            if in_market_positions:
                min_x = min(p[0] for p in in_market_positions)
                max_x = max(p[0] for p in in_market_positions)
                min_y = min(p[1] for p in in_market_positions)
                max_y = max(p[1] for p in in_market_positions)
            else:
                min_x = min_y = 0
                max_x = max_y = 0
            frame_market_x0 = min_x
            frame_market_y0 = min_y
            frame_market_w = (max_x - min_x + 1) or 1
            frame_market_h = (max_y - min_y + 1) or 1

            def _fallback_distance(px: int, py: int) -> int:
                rx, ry = frame_market_x0, frame_market_y0
                rw, rh = frame_market_w, frame_market_h
                inside_x = rx <= px < rx + rw
                inside_y = ry <= py < ry + rh
                if inside_x and inside_y:
                    return 0
                dx = 0 if inside_x else (rx - px if px < rx else px - (rx + rw - 1))
                dy = 0 if inside_y else (ry - py if py < ry else py - (ry + rh - 1))
                return abs(dx) + abs(dy)

            distance_fn = _fallback_distance

        distances: List[int] = []
        distance_sum = 0
        agents_frames: List[AgentFrame] = []
        for aid, ax, ay, in_mkt in agent_rows:
            dist = distance_fn(ax, ay)
            distances.append(dist)
            distance_sum += dist
            agents_frames.append(
                AgentFrame(
                    agent_id=aid,
                    x=ax,
                    y=ay,
                    in_marketplace=in_mkt,
                    distance_to_market=dist,
                )
            )
        max_dist_val = max(distances) if distances else 0
        frame = FrameData(
            round=target_round,
            grid_width=grid_w,
            grid_height=grid_h,
            market_x0=frame_market_x0,
            market_y0=frame_market_y0,
            market_width=frame_market_w,
            market_height=frame_market_h,
            agents=agents_frames,
            prices=prices,
            participation_count=part,
            total_agents=total_agents,
            max_distance_to_market=max_dist_val,
            avg_distance_to_market=(
                (distance_sum / len(distances)) if distances else 0.0
            ),
        )
        # HUD fields reconstruction (baseline approximated by first round's max distance)
        if self._round_index <= 1:
            self._baseline_max = max_dist_val if max_dist_val else 0  # type: ignore[attr-defined]
        try:
            baseline = getattr(self, "_baseline_max", None)
            if baseline and frame.avg_distance_to_market is not None:
                ci = spatial_convergence_index(frame.avg_distance_to_market, baseline)
            else:
                ci = None
        except Exception:
            ci = None
        try:
            dgst = frame_hash(frame)[0:12]
        except Exception:
            dgst = None
        # Rebuild with HUD fields (dataclass is frozen; create new instance if adding)
        frame = FrameData(
            round=frame.round,
            grid_width=frame.grid_width,
            grid_height=frame.grid_height,
            market_x0=frame.market_x0,
            market_y0=frame.market_y0,
            market_width=frame.market_width,
            market_height=frame.market_height,
            agents=frame.agents,
            prices=frame.prices,
            participation_count=frame.participation_count,
            total_agents=frame.total_agents,
            max_distance_to_market=frame.max_distance_to_market,
            avg_distance_to_market=frame.avg_distance_to_market,
            convergence_index=ci,
            hud_round_digest=dgst,
        )
        return frame

    def next_frame(self) -> Optional[FrameData]:  # type: ignore[override]
        self._load()
        assert self._records is not None
        if self._round_index >= self._max_round:
            return None
        target_round = self._round_index + 1
        round_rows = self._group_round(target_round)
        if not round_rows:
            self._round_index = self._max_round
            return None
        frame = self._build_frame_from_rows(target_round, round_rows)
        self._round_index = target_round
        return frame

    def seek(self, round_number: int) -> None:  # pragma: no cover - simple setter
        self._load()
        if round_number < 0:
            round_number = 0
        if round_number > self._max_round:
            round_number = self._max_round
        self._round_index = round_number

    def reset(self) -> None:  # pragma: no cover
        self._round_index = 0
