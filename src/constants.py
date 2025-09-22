"""Core numerical and economic constants for the simulation.

These values are sourced from project specification documents and are
treated as authoritative. Do NOT modify without updating specification
and associated validation tests that rely on exact magnitudes.
"""

from __future__ import annotations

# Primary convergence tolerance for rest-goods excess demand system
SOLVER_TOL: float = 1e-8

# Feasibility / conservation tolerance (value feasibility, goods conservation)
FEASIBILITY_TOL: float = 1e-10

# Small epsilon to prevent division by zero in proportional rationing
RATIONING_EPS: float = 1e-10

# Index of the numéraire good (p[numéraire] ≡ 1.0)
NUMERAIRE_GOOD: int = 0

# Minimum Cobb-Douglas preference weight (interiority enforcement)
MIN_ALPHA: float = 0.05

__all__ = [
    "SOLVER_TOL",
    "FEASIBILITY_TOL",
    "RATIONING_EPS",
    "NUMERAIRE_GOOD",
    "MIN_ALPHA",
]
"""
Economic simulation constants - Single source of truth.

This module defines all numerical constants used throughout the economic simulation
to ensure consistency and prevent duplication across modules.

All constants are derived from SPECIFICATION.md and serve as the authoritative
source for numerical tolerances, economic parameters, and system constraints.
"""

# Numerical tolerances from SPECIFICATION.md
SOLVER_TOL = 1e-8  # Primary convergence: ||Z_rest||_∞ < SOLVER_TOL
FEASIBILITY_TOL = 1e-10  # Conservation and feasibility checks
RATIONING_EPS = 1e-10  # Prevent division by zero in rationing

# Economic parameters
NUMERAIRE_GOOD = 0  # Good 1 is numéraire (p[0] ≡ 1.0)
MIN_ALPHA = 0.05  # Minimum preference weight (ensures interior solutions)

# Spatial parameters (Phase 2)
DEFAULT_GRID_SIZE = 15  # Default grid size for spatial simulations
DEFAULT_MARKET_WIDTH = 2  # Default marketplace width
DEFAULT_MARKET_HEIGHT = 2  # Default marketplace height

# Performance parameters
MAX_SOLVER_ITERATIONS = 100  # Maximum iterations for equilibrium solver
DEFAULT_AGENT_COUNT = 50  # Default number of agents for simulations
