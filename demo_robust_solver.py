#!/usr/bin/env python3
"""
Demonstration of the enhanced robust equilibrium solver.

This script shows the improvements provided by Option B (Enhanced Numerical Robustness)
including analytical Jacobian computation, multiple solver methods, and intelligent fallbacks.
"""

import numpy as np
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent import Agent
from src.econ.equilibrium import solve_walrasian_equilibrium
from src.econ.equilibrium_robust import solve_walrasian_equilibrium_robust
from src.econ.equilibrium_integration import (
    solve_equilibrium_enhanced, 
    solve_equilibrium_with_diagnostics,
    compare_solver_performance
)


def create_standard_edgeworth_box():
    """Create a standard 2-agent, 2-good Edgeworth box setup."""
    agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    return [agent1, agent2]


def create_challenging_case():
    """Create a more challenging convergence case with extreme preferences."""
    agent1 = Agent(1, np.array([0.99, 0.01]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    agent2 = Agent(2, np.array([0.01, 0.99]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    return [agent1, agent2]


def create_larger_economy():
    """Create a larger 5-agent, 3-good economy."""
    agents = []
    
    # Agent 1: Prefers good 1
    agents.append(Agent(1, np.array([0.7, 0.2, 0.1]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))
    
    # Agent 2: Prefers good 2
    agents.append(Agent(2, np.array([0.2, 0.7, 0.1]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])))
    
    # Agent 3: Prefers good 3
    agents.append(Agent(3, np.array([0.1, 0.2, 0.7]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])))
    
    # Agent 4: Balanced preferences
    agents.append(Agent(4, np.array([0.33, 0.33, 0.34]), np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.0])))
    
    # Agent 5: Different preferences
    agents.append(Agent(5, np.array([0.4, 0.3, 0.3]), np.array([0.0, 0.0, 0.5]), np.array([0.5, 0.0, 0.0])))
    
    return agents


def demonstrate_basic_functionality():
    """Demonstrate basic enhanced solver functionality."""
    print("=" * 60)
    print("ENHANCED ROBUST EQUILIBRIUM SOLVER DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("1. STANDARD EDGEWORTH BOX")
    print("-" * 30)
    agents = create_standard_edgeworth_box()
    
    # Solve with enhanced solver
    start_time = time.time()
    prices, z_norm, walras_dot, status, diagnostics = solve_equilibrium_with_diagnostics(agents)
    solve_time = time.time() - start_time
    
    print(f"Status: {status}")
    print(f"Method used: {diagnostics['method_used']}")
    print(f"Iterations: {diagnostics['iterations']}")
    print(f"Solve time: {solve_time:.6f} seconds")
    print(f"Final residual: {diagnostics['final_residual']:.2e}")
    print(f"Equilibrium prices: {prices}")
    print(f"Walras' Law error: {diagnostics['walras_dot']:.2e}")
    print()


def demonstrate_challenging_case():
    """Demonstrate performance on challenging convergence case."""
    print("2. CHALLENGING CONVERGENCE CASE")
    print("-" * 35)
    agents = create_challenging_case()
    
    # Solve with enhanced solver
    start_time = time.time()
    prices, z_norm, walras_dot, status, diagnostics = solve_equilibrium_with_diagnostics(agents)
    solve_time = time.time() - start_time
    
    print(f"Status: {status}")
    print(f"Method used: {diagnostics['method_used']}")
    print(f"Iterations: {diagnostics['iterations']}")
    print(f"Solve time: {solve_time:.6f} seconds")
    print(f"Final residual: {diagnostics['final_residual']:.2e}")
    print(f"Equilibrium prices: {prices}")
    print(f"Fallback attempts: {diagnostics['fallback_attempts']}")
    print()


def demonstrate_larger_economy():
    """Demonstrate scalability with larger economy."""
    print("3. LARGER ECONOMY (5 agents, 3 goods)")
    print("-" * 40)
    agents = create_larger_economy()
    
    # Solve with enhanced solver
    start_time = time.time()
    prices, z_norm, walras_dot, status, diagnostics = solve_equilibrium_with_diagnostics(agents)
    solve_time = time.time() - start_time
    
    print(f"Status: {status}")
    print(f"Method used: {diagnostics['method_used']}")
    print(f"Iterations: {diagnostics['iterations']}")
    print(f"Solve time: {solve_time:.6f} seconds")
    print(f"Final residual: {diagnostics['final_residual']:.2e}")
    print(f"Equilibrium prices: {prices}")
    print()


def demonstrate_performance_comparison():
    """Compare performance between original and robust solvers."""
    print("4. PERFORMANCE COMPARISON")
    print("-" * 30)
    agents = create_standard_edgeworth_box()
    
    print("Comparing original vs robust solver (5 trials each)...")
    results = compare_solver_performance(agents, num_trials=5)
    
    print("\nOriginal Solver:")
    orig = results['original_solver']
    print(f"  Success rate: {orig['success_rate']:.1%}")
    if orig['avg_time'] is not None:
        print(f"  Average time: {orig['avg_time']:.6f} ± {orig['std_time']:.6f} seconds")
        print(f"  Time range: {orig['min_time']:.6f} - {orig['max_time']:.6f} seconds")
    else:
        print("  Average time: N/A (all trials failed)")
    
    print("\nRobust Solver:")
    robust = results['robust_solver']
    print(f"  Success rate: {robust['success_rate']:.1%}")
    if robust['avg_time'] is not None:
        print(f"  Average time: {robust['avg_time']:.6f} ± {robust['std_time']:.6f} seconds")
        print(f"  Time range: {robust['min_time']:.6f} - {robust['max_time']:.6f} seconds")
    else:
        print("  Average time: N/A (all trials failed)")
    
    if orig['avg_time'] and robust['avg_time']:
        if robust['avg_time'] < orig['avg_time']:
            speedup = orig['avg_time'] / robust['avg_time']
            print(f"\nRobust solver is {speedup:.2f}x faster on average")
        else:
            slowdown = robust['avg_time'] / orig['avg_time']
            print(f"\nRobust solver is {slowdown:.2f}x slower on average")
    print()


def demonstrate_integration_features():
    """Demonstrate integration layer features."""
    print("5. INTEGRATION LAYER FEATURES")
    print("-" * 35)
    agents = create_standard_edgeworth_box()
    
    # Test fallback behavior
    print("Testing automatic fallback capability...")
    prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
        agents, 
        use_robust_solver=True,
        fallback_to_original=True
    )
    print(f"Enhanced solver with fallback: {status}")
    
    # Test original solver preference
    prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
        agents, 
        use_robust_solver=False
    )
    print(f"Original solver preference: {status}")
    
    # Test robust-only mode
    prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
        agents, 
        use_robust_solver=True,
        fallback_to_original=False
    )
    print(f"Robust-only mode: {status}")
    print()


def main():
    """Run the complete demonstration."""
    try:
        demonstrate_basic_functionality()
        demonstrate_challenging_case()
        demonstrate_larger_economy()
        demonstrate_performance_comparison()
        demonstrate_integration_features()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Key improvements in Option B (Enhanced Numerical Robustness):")
        print("• Analytical Jacobian computation for faster Newton-Raphson convergence")
        print("• Multiple solver methods with intelligent fallback strategies")
        print("• Enhanced diagnostic information for debugging and analysis")
        print("• Backward-compatible integration with existing solver interface")
        print("• Robust handling of challenging convergence cases")
        print("• Scalable performance for larger economies")
        
    except Exception as e:
        print(f"ERROR: Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())