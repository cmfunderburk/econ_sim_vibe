"""Shape guard tests for Walrasian equilibrium solver.

Ensures solver returns price and excess demand arrays with consistent
dimensions and numéraire normalization. Acts as an early warning for
any accidental dimension drift when refactoring the solver.
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from src.econ.equilibrium import (
    solve_walrasian_equilibrium,
    compute_excess_demand,
)
from src.constants import SOLVER_TOL


@pytest.mark.economic_core
def test_equilibrium_shape_and_normalization_basic():
    # Three-good minimal scenario with interior preferences
    a1 = type("Agent", (), {})()
    a1.alpha = np.array([0.4, 0.3, 0.3])
    a1.home_endowment = np.array([1.0, 0.0, 0.5])
    a1.personal_endowment = np.array([0.0, 0.5, 0.0])
    a1.total_endowment = a1.home_endowment + a1.personal_endowment

    a2 = type("Agent", (), {})()
    a2.alpha = np.array([0.2, 0.5, 0.3])
    a2.home_endowment = np.array([0.0, 1.0, 0.0])
    a2.personal_endowment = np.array([0.5, 0.0, 0.5])
    a2.total_endowment = a2.home_endowment + a2.personal_endowment

    a3 = type("Agent", (), {})()
    a3.alpha = np.array([0.3, 0.2, 0.5])
    a3.home_endowment = np.array([0.2, 0.3, 0.6])
    a3.personal_endowment = np.array([0.3, 0.2, 0.0])
    a3.total_endowment = a3.home_endowment + a3.personal_endowment

    agents = [a1, a2, a3]

    prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)

    # Status can be 'converged' or 'poor_convergence' but prices must exist
    assert prices is not None, f"Solver returned no prices: status={status}"

    n_goods = agents[0].alpha.size
    assert prices.shape == (n_goods,), "Price vector dimension mismatch"
    assert prices[0] == pytest.approx(1.0), "Numéraire not normalized to 1.0"
    assert np.all(prices > 0), "Prices must be strictly positive"

    # Excess demand full dimensionality
    Z = compute_excess_demand(prices, agents)
    assert Z.shape == (n_goods,), "Excess demand dimension mismatch"

    # Rest-goods norm should be finite
    assert np.isfinite(z_rest_norm)
    assert np.isfinite(walras_dot)

    # Basic Walras' Law sanity (not asserting convergence here)
    walras_check = float(abs(np.dot(prices, Z)))
    assert walras_check < 1e-6, f"Walras' Law residual too large: {walras_check:.2e}"


@pytest.mark.economic_core
def test_equilibrium_shape_guard_failure_mode(tmp_path):
    """Inject a deliberate mismatch (by tampering with agent endowment shape) and
    assert the solver raises or returns failure status. This documents expected
    failure behavior for malformed inputs (defensive programming contract).
    """
    a = type("Agent", (), {})()
    a.alpha = np.array([0.5, 0.5])
    a.home_endowment = np.array([1.0, 0.0])
    a.personal_endowment = np.array([0.0, 1.0, 2.0])  # Intentional length mismatch
    # Provide total_endowment attribute to mimic interface; will misalign
    a.total_endowment = np.array([1.0, 1.0, 2.0])

    with pytest.raises(Exception):
        solve_walrasian_equilibrium([a])
