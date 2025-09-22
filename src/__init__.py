"""
Economic Simulation Package

A research-grade economic simulation platform for studying spatial frictions
in market economies with agent-based modeling.
"""

__version__ = "0.1.0"
__author__ = "Economic Simulation Project"

# Re-export commonly used symbols for convenience
try:  # pragma: no cover - convenience import
	from .constants import (
		SOLVER_TOL,
		FEASIBILITY_TOL,
		RATIONING_EPS,
		NUMERAIRE_GOOD,
		MIN_ALPHA,
	)
except Exception:  # pragma: no cover
	# During partial initialization some modules may be absent
	pass

__all__ = [
	"SOLVER_TOL",
	"FEASIBILITY_TOL",
	"RATIONING_EPS",
	"NUMERAIRE_GOOD",
	"MIN_ALPHA",
]
