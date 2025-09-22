"""Spatial & visualization metrics utilities.

This module provides:
- spatial_convergence_index: avg_distance / initial_max_distance (safe handling for zero baseline)
- frame_hash: Stable hash of a subset of frame fields for regression detection.

Hash intentionally excludes floating fields prone to tiny numerical jitter (e.g., solver norms) unless explicitly requested.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import blake2b
from typing import Iterable, Optional

from .frame_data import FrameData

_HASH_FIELDS_DEFAULT = (
    "round",
    "participation_count",
    "total_agents",
    "max_distance_to_market",
    "avg_distance_to_market",
    # Prices hashed by value list (deterministic order)
    "prices",
)


def spatial_convergence_index(
    avg_distance: float, initial_max_distance: int | float
) -> float:
    """Compute spatial convergence index in [0,1].

    Returns 0.0 if baseline is 0 (all agents started inside marketplace).
    Clamps negative or NaN inputs to 0 defensively.
    """
    if initial_max_distance <= 0:
        return 0.0
    ratio = avg_distance / float(initial_max_distance)
    if ratio < 0 or ratio != ratio:  # NaN check
        return 0.0
    # Mild upper bound guard in case of rounding spillover
    return min(ratio, 1.0)


def frame_hash(
    frame: FrameData,
    *,
    extra_fields: Optional[Iterable[str]] = None,
    hash_len: int = 16,
) -> str:
    """Compute stable short hash for a frame.

    Parameters
    ----------
    frame : FrameData
        Frame to hash.
    extra_fields : Iterable[str], optional
        Additional attribute names to include (must exist on FrameData).
    hash_len : int
        Digest length in bytes (default 16 -> 32 hex chars).
    """
    data = asdict(frame)
    fields = list(_HASH_FIELDS_DEFAULT)
    if extra_fields:
        for f in extra_fields:
            if f not in data:
                raise KeyError(f"FrameData has no field '{f}'")
            fields.append(f)
    # Create a deterministic bytes representation
    parts: list[bytes] = []
    for name in fields:
        val = data.get(name)
        parts.append(name.encode("utf-8"))
        if isinstance(val, list):
            parts.append(str(val).encode("utf-8"))
        else:
            parts.append(str(val).encode("utf-8"))
    h = blake2b(b"|".join(parts), digest_size=hash_len)
    return h.hexdigest()

__all__ = ["spatial_convergence_index", "frame_hash"]
