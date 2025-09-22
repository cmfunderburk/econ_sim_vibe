"""Geometry sidecar helpers.

Writes and loads the geometry sidecar JSON file that contains static spatial
context required for replay distance reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

GEOMETRY_SIDECAR_SCHEMA = "geometry.sidecar.v1"


@dataclass
class GeometrySpec:
    run_name: str
    grid_width: int
    grid_height: int
    market_x_min: int
    market_x_max: int
    market_y_min: int
    market_y_max: int
    movement_policy: str
    random_seed: int
    notes: Optional[str] = None
    generated_at_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": GEOMETRY_SIDECAR_SCHEMA,
            "run_name": self.run_name,
            "generated_at_ns": self.generated_at_ns,
            "grid": {"width": self.grid_width, "height": self.grid_height},
            "marketplace": {
                "x_min": self.market_x_min,
                "x_max": self.market_x_max,
                "y_min": self.market_y_min,
                "y_max": self.market_y_max,
                "width": self.market_x_max - self.market_x_min + 1,
                "height": self.market_y_max - self.market_y_min + 1,
            },
            "movement_policy": self.movement_policy,
            "random_seed": self.random_seed,
            **({"notes": self.notes} if self.notes else {}),
        }


def write_geometry_sidecar(output_dir: Path, spec: GeometrySpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec.generated_at_ns = spec.generated_at_ns or time.time_ns()
    path = output_dir / f"{spec.run_name}_geometry.json"
    with path.open("w") as f:
        json.dump(spec.to_dict(), f, indent=2)
    return path


def load_geometry_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != GEOMETRY_SIDECAR_SCHEMA:
            return None
        return data
    except Exception:
        return None


def manhattan_distance_to_market(
    x: int, y: int, gx_min: int, gx_max: int, gy_min: int, gy_max: int
) -> int:
    if gx_min <= x <= gx_max and gy_min <= y <= gy_max:
        return 0
    dx = 0 if gx_min <= x <= gx_max else min(abs(x - gx_min), abs(x - gx_max))
    dy = 0 if gy_min <= y <= gy_max else min(abs(y - gy_min), abs(y - gy_max))
    return dx + dy
