"""
Integration layer for robust equilibrium solver.

This module provides backward-compatible integration with the existing equilibrium
interface while adding robust solving capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

from .equilibrium_robust import solve_walrasian_equilibrium_robust
from .equilibrium import solve_walrasian_equilibrium

logger = logging.getLogger(__name__)


def solve_equilibrium_enhanced(
    agents: List, 
    initial_guess: Optional[np.ndarray] = None,
    use_robust_solver: bool = True,
    fallback_to_original: bool = True
) -> Tuple[np.ndarray, float, float, str]:
    """
    Enhanced equilibrium solver with automatic fallback.
    
    Attempts robust solving first, then falls back to original solver if needed.
    
    Args:
        agents: List of market participants
        initial_guess: Initial price guess (optional)
        use_robust_solver: Whether to try robust solver first
        fallback_to_original: Whether to fallback to original solver
        
    Returns:
        Tuple of (prices, z_norm, walras_dot, status)
        
    Notes:
        - Maintains full backward compatibility
        - Provides enhanced robustness when requested
        - Automatic intelligent fallback strategy
    """
    if not use_robust_solver:
        logger.info("Using original solver as requested")
        return solve_walrasian_equilibrium(agents, initial_guess)
    
    try:
        logger.info("Attempting robust equilibrium solver")
        result = solve_walrasian_equilibrium_robust(agents, initial_guess)
        
        # Check if robust solver succeeded
        if result[3] in ['converged', 'poor_convergence']:
            logger.info(f"Robust solver succeeded with status: {result[3]}")
            return result
        else:
            logger.warning(f"Robust solver failed with status: {result[3]}")
            if not fallback_to_original:
                return result
                
    except Exception as e:
        logger.warning(f"Robust solver encountered error: {e}")
        if not fallback_to_original:
            raise
    
    # Fallback to original solver
    if fallback_to_original:
        logger.info("Falling back to original equilibrium solver")
        try:
            result = solve_walrasian_equilibrium(agents, initial_guess)
            logger.info(f"Original solver completed with status: {result[3]}")
            return result
        except Exception as e:
            logger.error(f"Both solvers failed. Original solver error: {e}")
            raise
    
    # If we get here, robust solver failed and no fallback requested
    return result


def solve_equilibrium_with_diagnostics(
    agents: List, 
    initial_guess: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float, str, dict]:
    """
    Solve equilibrium with detailed diagnostic information.
    
    Args:
        agents: List of market participants
        initial_guess: Initial price guess (optional)
        
    Returns:
        Tuple of (prices, z_norm, walras_dot, status, diagnostics)
        where diagnostics contains detailed convergence information
    """
    try:
        # Use robust solver with diagnostics
        result = solve_walrasian_equilibrium_robust(
            agents, initial_guess, return_diagnostics=True
        )
        
        if len(result) == 5:  # Has diagnostics
            prices, z_norm, walras_dot, status, diagnostics = result
            
            # Convert ConvergenceResult to dict for easier consumption
            diag_dict = {
                'converged': diagnostics.success,
                'method_used': diagnostics.method_used,
                'iterations': diagnostics.iterations,
                'final_residual': diagnostics.final_residual,
                'walras_dot': diagnostics.walras_law_error,
                'solve_time': diagnostics.convergence_time,
                'status': diagnostics.status_message,
                'fallback_attempts': diagnostics.fallback_attempts
            }
            
            return prices, z_norm, walras_dot, status, diag_dict
        else:
            # Fallback if diagnostics not available
            prices, z_norm, walras_dot, status = result
            diag_dict = {
                'converged': status == 'converged',
                'method_used': 'robust_solver',
                'iterations': None,
                'final_residual': z_norm,
                'walras_dot': walras_dot,
                'solve_time': None,
                'status': status,
                'fallback_attempts': []
            }
            return prices, z_norm, walras_dot, status, diag_dict
            
    except Exception as e:
        logger.error(f"Diagnostic equilibrium solve failed: {e}")
        # Return minimal diagnostic info on failure
        raise


def compare_solver_performance(
    agents: List,
    initial_guess: Optional[np.ndarray] = None,
    num_trials: int = 5
) -> dict:
    """
    Compare performance between original and robust solvers.
    
    Args:
        agents: List of market participants
        initial_guess: Initial price guess (optional)
        num_trials: Number of trials for performance comparison
        
    Returns:
        Dictionary with performance comparison results
    """
    import time
    
    results = {
        'original_solver': {'times': [], 'successes': 0, 'errors': []},
        'robust_solver': {'times': [], 'successes': 0, 'errors': []}
    }
    
    # Test original solver
    for trial in range(num_trials):
        try:
            start_time = time.time()
            prices, z_norm, walras_dot, status = solve_walrasian_equilibrium(agents, initial_guess)
            solve_time = time.time() - start_time
            
            results['original_solver']['times'].append(solve_time)
            if status == 'converged':
                results['original_solver']['successes'] += 1
                
        except Exception as e:
            results['original_solver']['errors'].append(str(e))
    
    # Test robust solver
    for trial in range(num_trials):
        try:
            start_time = time.time()
            prices, z_norm, walras_dot, status = solve_walrasian_equilibrium_robust(agents, initial_guess)
            solve_time = time.time() - start_time
            
            results['robust_solver']['times'].append(solve_time)
            if status in ['converged', 'poor_convergence']:
                results['robust_solver']['successes'] += 1
                
        except Exception as e:
            results['robust_solver']['errors'].append(str(e))
    
    # Compute summary statistics
    for solver in ['original_solver', 'robust_solver']:
        times = results[solver]['times']
        if times:
            results[solver]['avg_time'] = np.mean(times)
            results[solver]['std_time'] = np.std(times)
            results[solver]['min_time'] = np.min(times)
            results[solver]['max_time'] = np.max(times)
        else:
            results[solver]['avg_time'] = None
            results[solver]['std_time'] = None
            results[solver]['min_time'] = None
            results[solver]['max_time'] = None
            
        results[solver]['success_rate'] = results[solver]['successes'] / num_trials
    
    return results