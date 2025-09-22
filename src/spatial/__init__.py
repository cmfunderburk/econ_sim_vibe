"""Spatial public API re-exports.

This package exposes the implemented Grid and Position types from grid.py.
All legacy stubs removed; see src/spatial/grid.py for full implementation.
"""

from .grid import Grid, Position  # noqa: F401

__all__ = ["Grid", "Position"]
