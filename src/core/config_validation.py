"""Configuration validation utilities.

Provides a single entry point `validate_simulation_config` that performs
sanity checks on `SimulationConfig` instances and raises ValueError with
an aggregated, human-readable message if any issues are detected.

Design goals:
- Fail fast for clearly impossible or dangerous configurations.
- Aggregate all detected issues to reduce iteration cycles.
- Keep logic dependency-light (pure Python) to allow reuse in CLI & tests.
"""

from __future__ import annotations

from typing import List
from .simulation import SimulationConfig


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate a SimulationConfig instance.

    Raises:
        ValueError: if any validation rule fails. The exception message
            contains all detected issues (one per line) prefixed with '*'.
    """
    issues: List[str] = []

    # Basic numeric bounds
    if config.n_agents <= 0:
        issues.append("n_agents must be positive")
    if config.n_goods < 2:
        issues.append("n_goods must be at least 2 (need a numéraire + another good)")
    if config.grid_width <= 0 or config.grid_height <= 0:
        issues.append("grid dimensions must be positive")
    if config.max_rounds <= 0:
        issues.append("max_rounds must be positive")
    if config.movement_cost < 0:
        issues.append("movement_cost cannot be negative")

    # Marketplace geometry
    if config.marketplace_width <= 0 or config.marketplace_height <= 0:
        issues.append("marketplace dimensions must be positive")
    if (
        config.marketplace_width > config.grid_width
        or config.marketplace_height > config.grid_height
    ):
        issues.append("marketplace dimensions cannot exceed grid dimensions")

    # Relative scale heuristics (warnings escalated to errors for now)
    # Prevent degenerate case where marketplace == entire grid (unless very small simulation)
    if (
        config.marketplace_width == config.grid_width
        and config.marketplace_height == config.grid_height
        and (config.grid_width * config.grid_height) > 9
    ):
        issues.append(
            "marketplace spans entire grid (disable spatial friction) — adjust dimensions"
        )

    # Agent density heuristic: ensure grid is not trivially overcrowded
    cells = config.grid_width * config.grid_height
    if cells > 0 and config.n_agents / cells > 0.75:
        issues.append(
            f"agent density {config.n_agents}/{cells} (>75%) may cause movement degeneracy"
        )

    if issues:
        raise ValueError(
            "Invalid simulation configuration:\n"
            + "\n".join(f"* {msg}" for msg in issues)
        )


__all__ = ["validate_simulation_config"]
