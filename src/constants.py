"""
Economic simulation constants - Single source of truth.

This module defines all numerical constants used throughout the economic simulation
to ensure consistency and prevent duplication across modules.

All constants are derived from SPECIFICATION.md and serve as the authoritative
source for numerical tolerances, economic parameters, and system constraints.
"""

# Numerical tolerances from SPECIFICATION.md
SOLVER_TOL = 1e-8        # Primary convergence: ||Z_rest||_∞ < SOLVER_TOL  
FEASIBILITY_TOL = 1e-10  # Conservation and feasibility checks
RATIONING_EPS = 1e-10    # Prevent division by zero in rationing

# Economic parameters
NUMERAIRE_GOOD = 0       # Good 1 is numéraire (p[0] ≡ 1.0)
MIN_ALPHA = 0.05         # Minimum preference weight (ensures interior solutions)

# Spatial parameters (Phase 2)
DEFAULT_GRID_SIZE = 15   # Default grid size for spatial simulations
DEFAULT_MARKET_WIDTH = 2 # Default marketplace width
DEFAULT_MARKET_HEIGHT = 2 # Default marketplace height

# Performance parameters
MAX_SOLVER_ITERATIONS = 100  # Maximum iterations for equilibrium solver
DEFAULT_AGENT_COUNT = 50     # Default number of agents for simulations