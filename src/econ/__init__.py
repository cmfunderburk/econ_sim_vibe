"""Economic computation modules for the simulation.

This package contains the core economic algorithms including equilibrium solving,
market clearing, and welfare analysis.
"""

from .equilibrium import (
    solve_walrasian_equilibrium,
    compute_excess_demand,
    validate_equilibrium_invariants,
    solve_equilibrium
)

from .market import (
    execute_constrained_clearing,
    apply_trades_to_agents
)

__all__ = [
    'solve_walrasian_equilibrium',
    'compute_excess_demand', 
    'validate_equilibrium_invariants',
    'solve_equilibrium',
    'execute_constrained_clearing',
    'apply_trades_to_agents'
]