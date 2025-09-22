"""Snapshot/export scaffolding.

Provides a minimal Snapshotter that writes:
 - JSON metadata per frame (always)
 - PNG image if a pygame Surface is provided (GUI path)

Design goals:
 - Zero heavy dependencies beyond pygame (optional) and stdlib
 - Purely additive; does not alter simulation semantics
 - Safe to call in headless mode: gracefully skips PNG if no surface

Future extensions:
 - Video assembly (ffmpeg) after run
 - Batch summary export (Parquet/CSV)
 - Hashing & integrity metadata
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from .frame_data import FrameData

try:  # optional pygame import
    import pygame  # type: ignore

    HAS_PYGAME = True
except Exception:  # pragma: no cover
    HAS_PYGAME = False


class Snapshotter:
    def __init__(self, directory: Path, prefix: str = "frame") -> None:
        self.directory = directory
        self.prefix = prefix
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        frame: FrameData,
        *,
        surface: Optional["pygame.Surface"] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist snapshot.

        Returns path to JSON metadata file (PNG shares same stem if written).
        """
        ts = int(time.time() * 1000)
        stem = f"{self.prefix}_r{frame.round}_{ts}"
        json_path = self.directory / f"{stem}.json"

        data = asdict(frame)
        if extra_meta:
            data.update(extra_meta)
        data["snapshot_timestamp_ms"] = ts

        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

        # Attempt PNG if surface + pygame available
        if surface is not None and HAS_PYGAME:
            try:  # pragma: no cover - IO heavy
                png_path = self.directory / f"{stem}.png"
                pygame.image.save(surface, str(png_path))  # type: ignore[attr-defined]
            except Exception:
                pass
        return json_path


__all__ = ["Snapshotter"]
