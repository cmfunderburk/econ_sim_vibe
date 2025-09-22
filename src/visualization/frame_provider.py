"""Frame provider interfaces for live vs replay visualization pipelines.

This scaffolding supports the upcoming Visualization MVP.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Iterable, Callable

from .frame_data import FrameData
from .metrics import spatial_convergence_index, frame_hash


class FrameProvider(ABC):
    """Abstract base class for frame sources.

    Implementations must provide sequential access and optional random access.
    """

    @abstractmethod
    def current(self) -> FrameData:
        """Return the current frame (no advancement)."""

    @abstractmethod
    def advance(self) -> bool:
        """Advance to next frame.

        Returns True if advanced, False if at end.
        """

    def seek(self, index: int) -> FrameData:  # pragma: no cover (default behavior)
        raise NotImplementedError("seek not implemented for this provider")

    def has_next(self) -> bool:  # pragma: no cover (default behavior)
        return True

    # --- Added metrics helpers ---
    @property
    def initial_max_distance(self) -> Optional[int]:  # pragma: no cover
        return None

    def convergence_index(self) -> Optional[float]:  # pragma: no cover
        """Return spatial convergence index if distances are available."""
        cur = self.current()
        base = self.initial_max_distance
        if base is None or cur.max_distance_to_market is None or cur.avg_distance_to_market is None:
            return None
        return spatial_convergence_index(cur.avg_distance_to_market, base)

    def frame_digest(self) -> str:  # pragma: no cover
        return frame_hash(self.current())


class LiveFrameProvider(FrameProvider):  # pragma: no cover (placeholder)
    def __init__(self, builder: Callable[[], FrameData], baseline: Optional[int] = None):
        self._builder: Callable[[], FrameData] = builder
        self._frame: Optional[FrameData] = None
        self._baseline: Optional[int] = baseline

    def current(self) -> FrameData:
        if self._frame is None:
            frame = self._builder()
            self._frame = frame
            if self._baseline is None and frame.max_distance_to_market is not None:
                self._baseline = frame.max_distance_to_market
        # At this point _frame is guaranteed not None
        return self._frame  # type: ignore[return-value]

    def advance(self) -> bool:
        frame = self._builder()
        self._frame = frame
        if self._baseline is None and frame.max_distance_to_market is not None:
            self._baseline = frame.max_distance_to_market
        return True

    @property
    def initial_max_distance(self) -> Optional[int]:
        return self._baseline


class ReplayFrameProvider(FrameProvider):  # pragma: no cover (placeholder)
    def __init__(self, frames: Iterable[FrameData]):
        self._frames = list(frames)
        self._idx = 0
        self._baseline: Optional[int] = None

    def current(self) -> FrameData:
        frame = self._frames[self._idx]
        if self._baseline is None:
            self._baseline = frame.max_distance_to_market
        return frame

    def advance(self) -> bool:
        if self._idx + 1 >= len(self._frames):
            return False
        self._idx += 1
        return True

    def seek(self, index: int) -> FrameData:
        if not (0 <= index < len(self._frames)):
            raise IndexError(index)
        self._idx = index
        frame = self._frames[self._idx]
        if self._baseline is None:
            self._baseline = frame.max_distance_to_market
        return frame

    def has_next(self) -> bool:
        return self._idx + 1 < len(self._frames)

    @property
    def initial_max_distance(self) -> Optional[int]:
        return self._baseline


__all__ = [
    "FrameProvider",
    "LiveFrameProvider",
    "ReplayFrameProvider",
]
