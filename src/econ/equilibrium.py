"""Walrasian equilibrium solver (minimal implementation for tests).

The solver targets price vectors p with p[0] = 1 (numéraire) such that
aggregate excess demand (excluding the numéraire good) is ~0 within
SOLVER_TOL. Handles edge cases (no participants, insufficient goods).
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

try:
    from ..constants import SOLVER_TOL, NUMERAIRE_GOOD  # type: ignore
except Exception:
    from constants import SOLVER_TOL, NUMERAIRE_GOOD  # type: ignore

try:  # SciPy optional; tests rely on its presence but guard anyway
    from scipy.optimize import fsolve
except Exception:  # pragma: no cover - fallback only if SciPy missing
    fsolve = None  # type: ignore


def _aggregate_excess_demand(prices: np.ndarray, agents: List["Agent"]) -> np.ndarray:
    n_goods = prices.size
    z = np.zeros(n_goods)
    for a in agents:
        z += a.excess_demand(prices)
    return z


def _initial_prices(n_goods: int) -> np.ndarray:
    p = np.ones(n_goods)
    p[NUMERAIRE_GOOD] = 1.0
    return p


def solve_walrasian_equilibrium(
    agents: List["Agent"],
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """Compute Walrasian equilibrium prices for Cobb-Douglas agents.

    Returns
    -------
    (prices, z_rest_norm, walras_dot, status)
      prices : np.ndarray or None
        Price vector with p[0]=1 on success, else None for no participants.
      z_rest_norm : float
        Infinity norm of excess demand excluding the numéraire good.
      walras_dot : float
        |p·Z(p)| sanity check value.
      status : str
        'converged', 'no_participants', 'poor_convergence', 'failed',
        or edge-case labels used in tests.
    """
    if not agents:
        return None, 0.0, 0.0, "no_participants"

    n_goods = agents[0].alpha.size
    if n_goods < 2:
        # Degenerate single-good economy (prices not meaningful)
        p = _initial_prices(n_goods)
        return p, 0.0, 0.0, "insufficient_goods"

    # Filter out zero wealth agents to avoid singular systems
    viable = []
    for a in agents:
        if np.dot(np.ones(n_goods), a.total_endowment) > 0.0:
            viable.append(a)
    if len(viable) < 1:
        return None, 0.0, 0.0, "insufficient_viable_agents"

    agents = viable

    p0 = _initial_prices(n_goods)

    def system(p_rest: np.ndarray) -> np.ndarray:
        prices = p0.copy()
        prices[1:] = p_rest
        z = _aggregate_excess_demand(prices, agents)
        return z[1:]  # Rest-goods excess demand

    if fsolve is None:
        # Very naive tâtonnement fallback (should not trigger in CI)
        p_rest = np.ones(n_goods - 1)
        for _ in range(500):
            F = system(p_rest)
            if np.linalg.norm(F, np.inf) < SOLVER_TOL:
                break
            p_rest = np.maximum(p_rest * (1 + 0.1 * F), 1e-8)
        prices = p0.copy()
        prices[1:] = p_rest
    else:
        p_rest_guess = np.ones(n_goods - 1)
        try:
            p_rest, info, ier, msg = fsolve(system, p_rest_guess, full_output=True)
            prices = p0.copy()
            prices[1:] = p_rest
        except Exception:  # pragma: no cover - defensive
            return None, 0.0, 0.0, "failed"

    # Normalization enforcement (numéraire)
    prices[0] = 1.0

    Z_full = _aggregate_excess_demand(prices, agents)
    z_rest_norm = float(np.linalg.norm(Z_full[1:], ord=np.inf))
    walras_dot = float(abs(np.dot(prices, Z_full)))

    if z_rest_norm < SOLVER_TOL:
        status = "converged"
    else:
        status = "poor_convergence"

    return prices, z_rest_norm, walras_dot, status

__all__ = ["solve_walrasian_equilibrium"]
"""
Walrasian equilibrium solver for the economic simulation.

This module implements the core equilibrium computation using Cobb-Douglas utility
functions with numéraire normalization. The solver uses closed-form demand functions
and scipy optimization for market-clearing prices.

Key Features:
- Numéraire normalization (p₁ ≡ 1) for price vector identification
- Closed-form Cobb-Douglas demand computation
- Rest-goods convergence criterion ||Z_{2:n}||_∞ < SOLVER_TOL
- Economic invariant validation (Walras' Law, conservation)
- Numerical stability with price floors and error handling

Mathematical Foundation:
For agent i with Cobb-Douglas utility U_i(x) = ∏_j x_j^{α_{ij}} where ∑_j α_{ij} = 1:
- Demand: x_{ij}(p, ω_i) = α_{ij} * (p·ω_i) / p_j
- Excess demand: Z_i(p) = x_i(p, ω_i) - ω_i
- Market clearing: ∑_i Z_i(p) = 0 for all goods

The solver normalizes good 1 as numéraire (p₁ ≡ 1) and solves for rest prices
p₂, p₃, ..., p_n using the excess demand system Z_{2:n}(p) = 0.
"""

import numpy as np
import scipy.optimize
from typing import List, Tuple, Optional, Callable, Dict, Any
import logging

# Import constants from centralized source
try:
    from constants import (  # noqa: F401
        SOLVER_TOL,
        FEASIBILITY_TOL,
        NUMERAIRE_GOOD,
        MIN_ALPHA,
    )
except ImportError:
    # Fallback for different execution contexts
    from src.constants import (  # noqa: F401
        SOLVER_TOL,
        FEASIBILITY_TOL,
        NUMERAIRE_GOOD,
        MIN_ALPHA,
    )

logger = logging.getLogger(__name__)


def compute_excess_demand(prices: np.ndarray, agents: List) -> np.ndarray:
    """
    Compute aggregate excess demand for marketplace participants.

    Uses Cobb-Douglas closed forms: x_{ij} = α_{ij} * wealth_i / p_j
    where wealth_i = p · ω_i^{total} for agent i.

    Args:
        prices: Price vector with p[0] = 1.0 (numéraire)
        agents: List of marketplace participants only

    Returns:
        Aggregate excess demand vector Z(p) = ∑_i [x_i(p) - ω_i^{total}]

    Raises:
        AssertionError: If no participants or invalid price vector

    Notes:
        - Only uses agents currently in marketplace (local-participants principle)
        - Uses total endowments (home + personal) for theoretical clearing
        - Guards against division by zero and negative prices
    """
    assert agents, "No participants in market this round"

    # Guard against numerical issues
    eps = 1e-10
    prices = np.maximum(prices, eps)  # Floor prices to prevent division by zero
    n_goods = agents[0].alpha.size

    # Validate numéraire constraint
    assert abs(prices[0] - 1.0) < 1e-12, f"Numéraire violated: p[0]={prices[0]}"

    total_demand = np.zeros(n_goods)
    total_endowment = np.zeros(n_goods)

    for agent in agents:
        # Use total endowment (home + personal) for LTE computation
        omega_total = agent.home_endowment + agent.personal_endowment
        wealth = float(np.dot(prices, omega_total))

        # Skip agents with zero or negative wealth to avoid singular system
        if wealth <= FEASIBILITY_TOL:
            logger.warning(
                f"Agent {agent.agent_id} has zero wealth {wealth:.2e}, excluding from demand computation"
            )
            continue

        # Cobb-Douglas demand: x_ij = alpha_ij * wealth / p_j
        demand = agent.alpha * wealth / prices

        total_demand += demand
        total_endowment += omega_total

    # Excess demand: Z(p) = demand - endowment
    excess_demand = total_demand - total_endowment

    return excess_demand


def _tatonnement(
    excess_fn: Callable[[np.ndarray], np.ndarray],
    p_rest_initial: np.ndarray,
    max_iterations: int = 500,
    step_size: float = 0.2,
    damping: float = 0.9,
    min_step: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simple adaptive tâtonnement in rest-goods space.

    Price update in rest-goods parameterization (p1 ≡ 1):
        p_rest^{t+1} = max(ε, p_rest^t * (1 + step_size * Z_rest(p)))

    Multiplicative update preserves positivity; adaptive damping reduces
    step_size if residual norm increases.

    Args:
        excess_fn: Function returning Z_rest(p_rest)
        p_rest_initial: Initial rest-goods prices (positive)
        max_iterations: Hard iteration cap
        step_size: Initial multiplicative step scale
        damping: Factor to scale step_size when residual worsens
        min_step: Minimum allowed step size before giving up

    Returns:
        (p_rest, diagnostics) where diagnostics includes iterations, final_norm,
        path list of norms, and final step_size.
    """
    eps = 1e-10
    p = np.maximum(p_rest_initial.copy(), eps)
    norms = []
    current_z = excess_fn(p)
    current_norm = float(np.linalg.norm(current_z, ord=np.inf))
    norms.append(current_norm)
    it = 0

    while it < max_iterations and current_norm > SOLVER_TOL and step_size >= min_step:
        proposal = np.maximum(p * (1.0 + step_size * current_z), eps)
        z_new = excess_fn(proposal)
        norm_new = float(np.linalg.norm(z_new, ord=np.inf))
        if norm_new <= current_norm:  # Accept improvement
            p = proposal
            current_z = z_new
            current_norm = norm_new
            norms.append(current_norm)
            # Mild acceleration when consistently improving
            if len(norms) >= 3 and norms[-1] < norms[-2] < norms[-3]:
                step_size = min(step_size * 1.05, 1.0)
        else:
            # Backtrack (dampen step) and retry
            step_size *= damping
        it += 1

    diagnostics = {
        "iterations": it,
        "final_norm": current_norm,
        "norm_path": norms,
        "final_step_size": step_size,
    }
    return p, diagnostics


def solve_walrasian_equilibrium(
    agents: List[Any],
    initial_guess: Optional[np.ndarray] = None,
    enable_fallback: bool = True,
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """
    Solve for market-clearing prices with numéraire normalization.

    Finds price vector p such that aggregate excess demand equals zero:
    ∑_i Z_i(p) = 0 where Z_i(p) = x_i(p, ω_i) - ω_i

    Uses numéraire normalization p₁ ≡ 1 and solves for rest goods p₂, ..., p_n.
    Primary convergence criterion: ||Z_{2:n}(p)||_∞ < SOLVER_TOL

    Args:
        agents: Marketplace participants (post-move, inside marketplace)
        initial_guess: Optional initial price vector for rest goods (p₂, ..., p_n)

    Returns:
        Tuple of (prices, z_rest_norm, walras_dot, status):
        - prices: Equilibrium price vector with p[0] = 1.0
        - z_rest_norm: ||Z_{2:n}(p)||_∞ (primary convergence metric)
        - walras_dot: |p·Z(p)| (Walras' Law sanity check)
        - status: 'converged', 'no_participants', 'failed', 'max_iterations'

    Notes:
        - Returns None for prices if insufficient participants for equilibrium
        - Uses scipy.optimize.fsolve for numerical root finding
        - Validates economic invariants and logs convergence diagnostics
    """
    # Edge case: insufficient participants for meaningful equilibrium
    if not agents:
        return None, 0.0, 0.0, "no_participants"

    n_goods = agents[0].alpha.size

    # Pre-filter agents by wealth to check viable participants
    # Use uniform prices for initial wealth screening
    uniform_prices = np.ones(n_goods)
    viable_agents = []
    for agent in agents:
        omega_total = agent.home_endowment + agent.personal_endowment
        wealth = np.dot(uniform_prices, omega_total)
        if wealth > FEASIBILITY_TOL:
            viable_agents.append(agent)

    # Edge case: need at least 2 viable agents and 2 goods for relative prices
    if len(viable_agents) < 2 or n_goods < 2:
        logger.warning(
            f"Insufficient viable participants: {len(viable_agents)}/{len(agents)} agents, {n_goods} goods"
        )
        if len(viable_agents) < len(agents):
            return None, 0.0, 0.0, "insufficient_viable_agents"
        else:
            return None, 0.0, 0.0, "insufficient_participants"

    def excess_demand_rest_goods(p_rest: np.ndarray) -> np.ndarray:
        """
        Excess demand for goods 2,...,n with numéraire constraint.

        Concatenates p₁ = 1.0 with rest prices and returns excess demand
        for goods 2 through n only (excludes numéraire).

        Filters agents dynamically based on current price iterate to prevent
        misclassification in heterogeneous-price economies.
        """
        prices = np.concatenate([[1.0], p_rest])  # p₁ ≡ 1

        try:
            # Filter agents based on current price iterate for economic consistency
            current_viable_agents = []
            for agent in agents:
                omega_total = agent.home_endowment + agent.personal_endowment
                wealth = np.dot(
                    prices, omega_total
                )  # Use current prices, not uniform/guess

                if wealth > FEASIBILITY_TOL:
                    current_viable_agents.append(agent)
                # Note: Don't log warnings here as this runs every iteration

            # Require minimum viable agents for meaningful equilibrium
            if len(current_viable_agents) < 2:
                # Return large residual to signal infeasible system to optimizer
                return np.full(n_goods - 1, 1e6)

            excess_demand = compute_excess_demand(prices, current_viable_agents)
            return excess_demand[1:]  # Return only rest goods (exclude numéraire)
        except Exception as e:
            logger.error(f"Excess demand computation failed: {e}")
            # Return large residual to signal failure to optimizer
            return np.full(n_goods - 1, 1e6)

    # Initial guess for rest-goods prices
    if initial_guess is None:
        p_rest_initial = np.ones(n_goods - 1)  # Uniform relative prices
    else:
        p_rest_initial = initial_guess.copy()

    # Solve using scipy
    solver_method = "fsolve"
    tatonnement_info: Dict[str, Any] = {}
    try:
        # Request full_output for potential future diagnostics; take first element (solution vector)
        p_rest_solution_full = scipy.optimize.fsolve(
            excess_demand_rest_goods,
            p_rest_initial,
            xtol=SOLVER_TOL,
            maxfev=1000,  # Prevent infinite loops
            full_output=False,
        )
        # Ensure ndarray form
        p_rest_solution = np.asarray(p_rest_solution_full, dtype=float)
        prices = np.concatenate([[1.0], p_rest_solution])

        # Price positivity guarantees - project prices to ensure positive floor
        # Economic interpretation: Negative prices are never meaningful in Walrasian systems
        # Apply floor to all non-numéraire prices while preserving numéraire constraint
        MIN_PRICE_FLOOR = 1e-6  # Small positive floor

        if np.any(prices[1:] <= 0):
            logger.warning(
                f"Negative/zero prices detected: {prices}, applying positivity projection"
            )
            # Floor projection for non-numéraire goods
            prices[1:] = np.maximum(prices[1:], MIN_PRICE_FLOOR)

            # Renormalize to maintain proper relative price structure if needed
            # For now, simple floor projection is sufficient

        # Compute convergence metrics
        z_rest_residual = excess_demand_rest_goods(p_rest_solution)
        z_rest_norm = float(np.linalg.norm(z_rest_residual, ord=np.inf))

        # Walras' Law validation (sanity check) - recompute viable agents for final prices
        final_viable_agents = []
        for agent in agents:
            omega_total = agent.home_endowment + agent.personal_endowment
            wealth = np.dot(prices, omega_total)
            if wealth > FEASIBILITY_TOL:
                final_viable_agents.append(agent)

        excess_demand_full = compute_excess_demand(prices, final_viable_agents)
        walras_dot = abs(np.dot(prices, excess_demand_full))

        # Validate convergence
        if z_rest_norm < SOLVER_TOL:
            status = "converged"
            logger.info(
                f"Solver converged ({solver_method}): ||Z_rest||_∞={z_rest_norm:.2e}, Walras={walras_dot:.2e}"
            )
        else:
            status = "poor_convergence"
            logger.warning(
                f"Poor convergence ({solver_method}): ||Z_rest||_∞={z_rest_norm:.2e} >= {SOLVER_TOL}"
            )

            # Attempt fallback tatonnement if enabled
            if enable_fallback:
                logger.info("Attempting tâtonnement fallback after poor convergence")

                def _rest_only_excess(p_rest_inner: np.ndarray) -> np.ndarray:
                    return excess_demand_rest_goods(p_rest_inner)

                p_tat, tatonnement_info = _tatonnement(
                    _rest_only_excess, p_rest_solution, max_iterations=800
                )
                tat_residual = _rest_only_excess(p_tat)
                tat_norm = float(np.linalg.norm(tat_residual, ord=np.inf))
                if tat_norm < z_rest_norm:  # Use improved solution
                    prices = np.concatenate([[1.0], p_tat])
                    z_rest_residual = tat_residual
                    z_rest_norm = tat_norm
                    # Reuse existing status labels to maintain test compatibility
                    status = (
                        "converged" if z_rest_norm < SOLVER_TOL else "poor_convergence"
                    )
                    logger.warning(
                        "Fallback tâtonnement %s: ||Z_rest||_∞=%.2e (initial %.2e) after %d iters (final step %.3g)",
                        status,
                        z_rest_norm,
                        tatonnement_info.get("norm_path", [np.nan])[0],
                        tatonnement_info.get("iterations", -1),
                        tatonnement_info.get("final_step_size", float("nan")),
                    )
                else:
                    logger.warning(
                        "Fallback tâtonnement provided no improvement (%.2e >= %.2e)",
                        tat_norm,
                        z_rest_norm,
                    )

        # Sanity check: Walras' Law should hold regardless of convergence
        if walras_dot > SOLVER_TOL:
            logger.warning(
                f"Walras' Law violation: |p·Z|={walras_dot:.2e} > {SOLVER_TOL}"
            )

        return prices, z_rest_norm, walras_dot, status
    except Exception as e:  # Broad catch ensures failure surfaces cleanly
        logger.error(f"Solver failed: {e}")
        return None, np.inf, np.inf, "failed"


def validate_equilibrium_invariants(
    prices: np.ndarray, agents: List[Any], excess_demand: np.ndarray
) -> bool:
    """
    Validate critical economic invariants for equilibrium solution.

    Tests all required invariants from SPECIFICATION.md:
    1. Numéraire constraint: p₁ ≡ 1
    2. Walras' Law: p·Z(p) ≈ 0
    3. Non-negativity: p ≥ 0
    4. Market clearing: ||Z_{2:n}||_∞ < SOLVER_TOL

    Args:
        prices: Equilibrium price vector
        agents: Marketplace participants
        excess_demand: Aggregate excess demand vector

    Returns:
        True if all invariants satisfied, False otherwise

    Raises:
        AssertionError: If critical invariants violated (configurable)
    """
    try:
        # 1. Numéraire constraint
        assert abs(prices[0] - 1.0) < 1e-12, f"Numéraire violated: p[0]={prices[0]}"

        # 2. Walras' Law
        walras_dot = np.dot(prices, excess_demand)
        assert abs(walras_dot) < SOLVER_TOL, f"Walras' Law violated: {walras_dot:.2e}"

        # 3. Non-negativity
        assert np.all(prices >= -FEASIBILITY_TOL), f"Negative prices: {prices}"

        # 4. Market clearing (primary convergence test)
        z_rest_norm = np.linalg.norm(excess_demand[1:], ord=np.inf)
        assert z_rest_norm < SOLVER_TOL, (
            f"Poor convergence: ||Z_rest||_∞={z_rest_norm:.2e}"
        )

        logger.info("All equilibrium invariants validated successfully")
        return True

    except AssertionError as e:
        logger.error(f"Equilibrium invariant violation: {e}")
        return False


def solve_equilibrium(
    agents: List[Any], normalization: str = "good_1", endowment_scope: str = "total"
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """
    High-level interface for equilibrium solving with configurable options.

    This function provides the main interface used by the simulation engine,
    with support for different normalization schemes and endowment scopes.

    Args:
        agents: Marketplace participants (post-move, inside marketplace)
        normalization: Price normalization method ('good_1' sets p[0] = 1.0)
        endowment_scope: 'total' (home+personal) or 'personal' (personal only)

    Returns:
        Tuple of (prices, z_rest_norm, walras_dot, status)
        Same as solve_walrasian_equilibrium but with interface compatibility

    Notes:
        - Currently only supports 'good_1' normalization and 'total' endowment scope
        - Future versions may add other normalization schemes
        - Validates parameters and delegates to core solver
    """
    # Parameter validation
    if normalization != "good_1":
        raise NotImplementedError(f"Normalization '{normalization}' not yet supported")

    if endowment_scope != "total":
        raise NotImplementedError(
            f"Endowment scope '{endowment_scope}' not yet supported"
        )

    # Delegate to core solver
    return solve_walrasian_equilibrium(agents)
