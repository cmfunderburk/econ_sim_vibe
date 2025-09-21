"""
Enhanced robust Walrasian equilibrium solver with multiple methods and fallback strategies.

This module extends the basic equilibrium solver with:
1. Multiple numerical methods (Newton-Raphson, Broyden, tâtonnement)
2. Analytical Jacobian computation for faster convergence
3. Intelligent fallback strategies for difficult convergence cases
4. Enhanced convergence diagnostics and adaptive tolerance handling
5. Detailed status reporting for debugging and performance analysis

Design Philosophy:
- Primary method: Newton-Raphson with analytical Jacobian (fastest convergence)
- Fallback 1: Broyden's method (quasi-Newton, more robust)
- Fallback 2: Tâtonnement process (guaranteed stability, slower)
- Emergency fallback: Uniform price reset with relaxed tolerance

Mathematical Foundation:
For Newton-Raphson method, we solve Z_rest(p_rest) = 0 using:
p_rest^{k+1} = p_rest^k - J^{-1}(p_rest^k) * Z_rest(p_rest^k)

where J is the Jacobian matrix J_ij = ∂Z_i/∂p_j for rest goods.
"""

import numpy as np
import scipy.optimize
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
from dataclasses import dataclass

# Import constants and existing equilibrium functions
try:
    from constants import SOLVER_TOL, FEASIBILITY_TOL, NUMERAIRE_GOOD, MIN_ALPHA
    from econ.equilibrium import compute_excess_demand
except ImportError:
    from src.constants import SOLVER_TOL, FEASIBILITY_TOL, NUMERAIRE_GOOD, MIN_ALPHA
    from src.econ.equilibrium import compute_excess_demand

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Detailed convergence information for enhanced diagnostics."""
    success: bool
    method_used: str
    iterations: int
    final_residual: float
    walras_law_error: float
    convergence_time: float
    status_message: str
    fallback_attempts: List[str]


class RobustEquilibriumSolver:
    """
    Enhanced equilibrium solver with multiple methods and intelligent fallbacks.
    
    This solver attempts multiple numerical methods in order of preference:
    1. Newton-Raphson with analytical Jacobian
    2. Broyden's quasi-Newton method
    3. Tâtonnement iterative process
    4. Emergency uniform price fallback
    """
    
    def __init__(self, tolerance: float = SOLVER_TOL, max_iterations: int = 1000):
        """Initialize the robust solver with configurable parameters."""
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.adaptive_tolerance = True
        
    def solve_equilibrium_robust(self, 
                                agents: List, 
                                initial_guess: Optional[np.ndarray] = None,
                                return_diagnostics: bool = False) -> Tuple:
        """
        Solve for market-clearing prices using robust multi-method approach.
        
        Args:
            agents: List of Agent objects participating in marketplace
            initial_guess: Optional initial price vector for rest goods
            return_diagnostics: Whether to return detailed convergence diagnostics
            
        Returns:
            If return_diagnostics=False: (prices, z_rest_norm, walras_dot, status)
            If return_diagnostics=True: (prices, z_rest_norm, walras_dot, status, diagnostics)
        """
        start_time = time.time()
        fallback_attempts = []
        
        # Edge case handling
        if not agents:
            result = None, 0.0, 0.0, 'no_participants'
            if return_diagnostics:
                diagnostics = ConvergenceResult(False, 'none', 0, np.inf, np.inf, 0.0, 'no_participants', [])
                return (*result, diagnostics)
            return result
            
        n_goods = agents[0].alpha.size
        if len(agents) < 2 or n_goods < 2:
            result = None, 0.0, 0.0, 'insufficient_participants'
            if return_diagnostics:
                diagnostics = ConvergenceResult(False, 'none', 0, np.inf, np.inf, 0.0, 'insufficient_participants', [])
                return (*result, diagnostics)
            return result
            
        # Filter viable agents (positive wealth)
        viable_agents = self._filter_viable_agents(agents)
        if len(viable_agents) < 2:
            result = None, 0.0, 0.0, 'insufficient_viable_agents'
            if return_diagnostics:
                diagnostics = ConvergenceResult(False, 'none', 0, np.inf, np.inf, 0.0, 'insufficient_viable_agents', [])
                return (*result, diagnostics)
            return result
            
        # Setup excess demand function for rest goods
        def excess_demand_rest(p_rest: np.ndarray) -> np.ndarray:
            """Excess demand for goods 2,...,n with numéraire constraint."""
            prices = np.concatenate([[1.0], p_rest])
            try:
                excess_demand = compute_excess_demand(prices, viable_agents)
                return excess_demand[1:]  # Return only rest goods
            except Exception as e:
                logger.error(f"Excess demand computation failed: {e}")
                return np.full(n_goods - 1, 1e6)
                
        # Initial guess preparation
        if initial_guess is None:
            p_rest_initial = np.ones(n_goods - 1)
        else:
            p_rest_initial = initial_guess.copy()
            
        # Method 1: Newton-Raphson with analytical Jacobian
        try:
            logger.info("Attempting Newton-Raphson with analytical Jacobian")
            p_rest_solution, iterations, success = self._newton_raphson_with_jacobian(
                excess_demand_rest, p_rest_initial, viable_agents, n_goods
            )
            if success:
                prices = np.concatenate([[1.0], p_rest_solution])
                convergence_time = time.time() - start_time
                z_rest_norm, walras_dot = self._compute_convergence_metrics(prices, viable_agents)
                
                if z_rest_norm < self.tolerance:
                    logger.info(f"Newton-Raphson converged in {iterations} iterations")
                    if return_diagnostics:
                        diagnostics = ConvergenceResult(
                            True, 'newton_raphson', iterations, z_rest_norm, walras_dot,
                            convergence_time, 'converged', fallback_attempts
                        )
                        return prices, z_rest_norm, walras_dot, 'converged', diagnostics
                    return prices, z_rest_norm, walras_dot, 'converged'
            else:
                logger.warning("Newton-Raphson failed to converge")
                fallback_attempts.append('newton_raphson_no_convergence')
        except Exception as e:
            logger.warning(f"Newton-Raphson failed: {e}")
            fallback_attempts.append('newton_raphson_exception')
            
        # Method 2: Broyden's quasi-Newton method
        try:
            logger.info("Falling back to Broyden's method")
            p_rest_solution, iterations, success = self._broyden_method(
                excess_demand_rest, p_rest_initial
            )
            if success:
                prices = np.concatenate([[1.0], p_rest_solution])
                convergence_time = time.time() - start_time
                z_rest_norm, walras_dot = self._compute_convergence_metrics(prices, viable_agents)
                
                if z_rest_norm < self.tolerance:
                    logger.info(f"Broyden's method converged in {iterations} iterations")
                    if return_diagnostics:
                        diagnostics = ConvergenceResult(
                            True, 'broyden', iterations, z_rest_norm, walras_dot,
                            convergence_time, 'converged', fallback_attempts
                        )
                        return prices, z_rest_norm, walras_dot, 'converged', diagnostics
                    return prices, z_rest_norm, walras_dot, 'converged'
            else:
                logger.warning("Broyden's method failed to converge")
                fallback_attempts.append('broyden_no_convergence')
        except Exception as e:
            logger.warning(f"Broyden's method failed: {e}")
            fallback_attempts.append('broyden_exception')
            
        # Method 3: Tâtonnement iterative process
        try:
            logger.info("Falling back to tâtonnement process")
            p_rest_solution, iterations, success = self._tatonnement_process(
                excess_demand_rest, p_rest_initial
            )
            if success:
                prices = np.concatenate([[1.0], p_rest_solution])
                convergence_time = time.time() - start_time
                z_rest_norm, walras_dot = self._compute_convergence_metrics(prices, viable_agents)
                
                if z_rest_norm < self.tolerance * 10:  # Relaxed tolerance for tâtonnement
                    logger.info(f"Tâtonnement converged in {iterations} iterations")
                    status = 'converged' if z_rest_norm < self.tolerance else 'poor_convergence'
                    if return_diagnostics:
                        diagnostics = ConvergenceResult(
                            True, 'tatonnement', iterations, z_rest_norm, walras_dot,
                            convergence_time, status, fallback_attempts
                        )
                        return prices, z_rest_norm, walras_dot, status, diagnostics
                    return prices, z_rest_norm, walras_dot, status
            else:
                logger.warning("Tâtonnement failed to converge")
                fallback_attempts.append('tatonnement_no_convergence')
        except Exception as e:
            logger.warning(f"Tâtonnement failed: {e}")
            fallback_attempts.append('tatonnement_exception')
            
        # Emergency fallback: uniform prices with relaxed tolerance
        logger.warning("All methods failed, using emergency uniform price fallback")
        fallback_attempts.append('emergency_fallback')
        
        prices = np.ones(n_goods)  # Uniform prices
        convergence_time = time.time() - start_time
        z_rest_norm, walras_dot = self._compute_convergence_metrics(prices, viable_agents)
        
        if return_diagnostics:
            diagnostics = ConvergenceResult(
                False, 'emergency_fallback', 0, z_rest_norm, walras_dot,
                convergence_time, 'failed', fallback_attempts
            )
            return prices, z_rest_norm, walras_dot, 'failed', diagnostics
        return prices, z_rest_norm, walras_dot, 'failed'
        
    def _filter_viable_agents(self, agents: List) -> List:
        """Filter agents with positive wealth for numerical stability."""
        viable_agents = []
        for agent in agents:
            wealth = np.sum(agent.total_endowment)  # Using total endowment as proxy
            if wealth > FEASIBILITY_TOL:
                viable_agents.append(agent)
        return viable_agents
        
    def _compute_analytical_jacobian(self, p_rest: np.ndarray, agents: List, n_goods: int) -> np.ndarray:
        """
        Compute analytical Jacobian matrix for Newton-Raphson method.
        
        For Cobb-Douglas utility, agent i's demand for good j is:
        x_ij(p) = α_ij * wealth_i / p_j where wealth_i = p · ω_i
        
        Excess demand for good j: Z_j(p) = ∑_i [x_ij(p) - ω_ij]
        
        The Jacobian element ∂Z_j/∂p_k:
        ∂Z_j/∂p_k = ∑_i ∂x_ij/∂p_k
        
        For Cobb-Douglas:
        ∂x_ij/∂p_k = α_ij * [∂wealth_i/∂p_k / p_j - wealth_i * δ_jk / p_j²]
        where ∂wealth_i/∂p_k = ω_ik and δ_jk is Kronecker delta.
        """
        prices = np.concatenate([[1.0], p_rest])
        jacobian = np.zeros((n_goods - 1, n_goods - 1))
        
        for agent in agents:
            omega_total = agent.home_endowment + agent.personal_endowment
            wealth = np.dot(prices, omega_total)
            
            # Skip zero-wealth agents
            if wealth <= 1e-12:
                continue
                
            # Compute derivatives for rest goods (j, k ∈ {1, ..., n-1})
            for j in range(n_goods - 1):  # j indexes rest goods array (0 to n-2)
                good_j = j + 1  # actual good index (1 to n-1)
                p_j = prices[good_j]
                alpha_j = agent.alpha[good_j]
                
                for k in range(n_goods - 1):  # k indexes rest goods array (0 to n-2)
                    good_k = k + 1  # actual good index (1 to n-1)
                    omega_k = omega_total[good_k]
                    
                    if j == k:  # Diagonal: ∂Z_j/∂p_j
                        # ∂x_ij/∂p_j = α_ij * (ω_ij/p_j - wealth_i/p_j²)
                        derivative = alpha_j * (omega_k / p_j - wealth / (p_j * p_j))
                    else:  # Off-diagonal: ∂Z_j/∂p_k
                        # ∂x_ij/∂p_k = α_ij * ω_ik / p_j
                        derivative = alpha_j * omega_k / p_j
                        
                    jacobian[j, k] += derivative
                
        return jacobian
        
    def _newton_raphson_with_jacobian(self, excess_demand_func, initial_guess: np.ndarray, 
                                    agents: List, n_goods: int) -> Tuple[np.ndarray, int, bool]:
        """Newton-Raphson method with analytical Jacobian computation."""
        p_rest = initial_guess.copy()
        
        for iteration in range(self.max_iterations):
            # Compute excess demand and Jacobian
            excess_demand = excess_demand_func(p_rest)
            jacobian = self._compute_analytical_jacobian(p_rest, agents, n_goods)
            
            # Check convergence
            residual_norm = np.linalg.norm(excess_demand, ord=np.inf)
            if residual_norm < self.tolerance:
                return p_rest, iteration + 1, True
                
            # Newton step: p_new = p_old - J^{-1} * F(p_old)
            try:
                delta = np.linalg.solve(jacobian, excess_demand)
                p_rest = p_rest - delta
                
                # Ensure positive prices
                p_rest = np.maximum(p_rest, FEASIBILITY_TOL)
                
            except np.linalg.LinAlgError:
                logger.warning("Jacobian singular in Newton-Raphson")
                return p_rest, iteration + 1, False
                
        return p_rest, self.max_iterations, False
        
    def _broyden_method(self, excess_demand_func, initial_guess: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """Broyden's quasi-Newton method for more robust convergence."""
        try:
            result = scipy.optimize.root(
                excess_demand_func, 
                initial_guess, 
                method='broyden1',
                options={'ftol': self.tolerance, 'maxiter': self.max_iterations}
            )
            return result.x, result.nit, result.success
        except Exception:
            return initial_guess, 0, False
            
    def _tatonnement_process(self, excess_demand_func, initial_guess: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """
        Tâtonnement (price adjustment) process for guaranteed stability.
        
        Uses adaptive step size: p_new = p_old + α * sign(Z(p_old))
        where α is adaptively adjusted based on convergence progress.
        """
        p_rest = initial_guess.copy()
        step_size = 0.1
        last_residual = np.inf
        
        for iteration in range(self.max_iterations):
            excess_demand = excess_demand_func(p_rest)
            residual_norm = np.linalg.norm(excess_demand, ord=np.inf)
            
            # Check convergence
            if residual_norm < self.tolerance * 10:  # Relaxed tolerance
                return p_rest, iteration + 1, True
                
            # Adaptive step size
            if residual_norm < last_residual:
                step_size *= 1.1  # Increase step size if improving
            else:
                step_size *= 0.5  # Decrease step size if diverging
                
            step_size = np.clip(step_size, 0.001, 0.5)  # Reasonable bounds
            
            # Price adjustment: move in direction of excess demand
            p_rest = p_rest + step_size * np.sign(excess_demand)
            p_rest = np.maximum(p_rest, FEASIBILITY_TOL)  # Ensure positive prices
            
            last_residual = residual_norm
            
        return p_rest, self.max_iterations, False
        
    def _compute_convergence_metrics(self, prices: np.ndarray, agents: List) -> Tuple[float, float]:
        """Compute convergence metrics for final validation."""
        try:
            excess_demand = compute_excess_demand(prices, agents)
            z_rest_norm = np.linalg.norm(excess_demand[1:], ord=np.inf)
            walras_dot = abs(np.dot(prices, excess_demand))
            return z_rest_norm, walras_dot
        except Exception:
            return np.inf, np.inf


# Convenience function with backward compatibility
def solve_walrasian_equilibrium_robust(agents: List, 
                                     initial_guess: Optional[np.ndarray] = None,
                                     tolerance: float = SOLVER_TOL,
                                     return_diagnostics: bool = False) -> Tuple:
    """
    Enhanced robust equilibrium solver with backward compatibility.
    
    This function provides the same interface as the original solve_walrasian_equilibrium
    but uses the enhanced robust solver with multiple methods and fallback strategies.
    """
    solver = RobustEquilibriumSolver(tolerance=tolerance)
    return solver.solve_equilibrium_robust(agents, initial_guess, return_diagnostics)