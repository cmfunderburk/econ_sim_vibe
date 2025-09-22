"""
Comprehensive tests for the robust equilibrium solver.

This module tests the enhanced equilibrium solver with multiple methods and fallback strategies.
Key focus areas:
1. Method selection and fallback behavior
2. Convergence performance under different conditions
3. Analytical Jacobian accuracy
4. Robustness under extreme scenarios
5. Diagnostic information quality
6. Backward compatibility with existing interface

Test Categories:
- Basic functionality and convergence
- Method-specific testing (Newton-Raphson, Broyden, Tâtonnement)
- Fallback scenario testing
- Performance and diagnostics
- Stress testing with difficult convergence cases
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.agent import Agent
from econ.equilibrium_robust import (
    RobustEquilibriumSolver,
    solve_walrasian_equilibrium_robust,
    ConvergenceResult,
)
from econ.equilibrium import solve_walrasian_equilibrium, compute_excess_demand
from constants import SOLVER_TOL, FEASIBILITY_TOL


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestRobustEquilibriumBasics:
    """Test basic functionality and convergence of robust solver."""

    def test_backward_compatibility_edgeworth_box(self):
        """Test that robust solver gives same results as original for standard cases."""
        # Standard Edgeworth box setup
        agent1 = Agent(
            1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        # Solve with original solver
        prices_orig, z_norm_orig, walras_orig, status_orig = (
            solve_walrasian_equilibrium([agent1, agent2])
        )

        # Solve with robust solver
        prices_robust, z_norm_robust, walras_robust, status_robust = (
            solve_walrasian_equilibrium_robust([agent1, agent2])
        )

        # Results should be essentially identical
        assert status_orig == "converged"
        assert status_robust == "converged"
        assert np.allclose(prices_orig, prices_robust, atol=1e-8)
        assert (
            abs(z_norm_orig - z_norm_robust) < 1e-8
        )  # Allow for slight numerical differences
        assert abs(walras_orig - walras_robust) < 1e-8

    def test_enhanced_diagnostics(self):
        """Test detailed convergence diagnostics."""
        agent1 = Agent(
            1, np.array([0.7, 0.3]), np.array([2.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.4, 0.6]), np.array([0.0, 1.5]), np.array([0.0, 0.0])
        )

        # Request diagnostics
        prices, z_norm, walras, status, diagnostics = (
            solve_walrasian_equilibrium_robust(
                [agent1, agent2], return_diagnostics=True
            )
        )

        assert status == "converged"
        assert isinstance(diagnostics, ConvergenceResult)
        assert diagnostics.success
        assert diagnostics.method_used in ["newton_raphson", "broyden", "tatonnement"]
        assert diagnostics.iterations >= 1
        assert diagnostics.final_residual < SOLVER_TOL
        assert diagnostics.convergence_time > 0
        assert isinstance(diagnostics.fallback_attempts, list)

    def test_multiple_agent_convergence(self):
        """Test convergence with multiple agents (3-agent economy)."""
        agents = [
            Agent(
                1,
                np.array([0.5, 0.3, 0.2]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            ),
            Agent(
                2,
                np.array([0.2, 0.5, 0.3]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            ),
            Agent(
                3,
                np.array([0.3, 0.2, 0.5]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, 0.0]),
            ),
        ]

        prices, z_norm, walras, status = solve_walrasian_equilibrium_robust(agents)

        assert status == "converged"
        assert z_norm < SOLVER_TOL
        assert abs(walras) < SOLVER_TOL
        assert prices[0] == 1.0  # Numéraire constraint
        assert np.all(prices > 0)  # All prices positive

    def test_different_initial_guesses(self):
        """Test robustness to different initial price guesses."""
        agent1 = Agent(
            1, np.array([0.8, 0.2]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.2, 0.8]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        initial_guesses = [
            None,  # Default
            np.array([0.5]),  # Low price
            np.array([5.0]),  # High price
            np.array([0.1]),  # Very low price
            np.array([50.0]),  # Very high price
        ]

        results = []
        for guess in initial_guesses:
            prices, z_norm, walras, status = solve_walrasian_equilibrium_robust(
                [agent1, agent2], guess
            )
            assert status == "converged"
            results.append(prices)

        # All should converge to the same solution
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], atol=1e-6)


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestAnalyticalJacobian:
    """Test the analytical Jacobian computation for Newton-Raphson method."""

    def test_jacobian_accuracy_two_goods(self):
        """Test analytical Jacobian against numerical approximation for 2-good case."""
        agent1 = Agent(
            1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )
        agents = [agent1, agent2]

        solver = RobustEquilibriumSolver()
        p_rest = np.array([1.5])  # Test point

        # Compute analytical Jacobian
        jacobian_analytical = solver._compute_analytical_jacobian(p_rest, agents, 2)

        # Compute numerical Jacobian
        def excess_demand_func(p):
            prices = np.concatenate([[1.0], p])
            excess = compute_excess_demand(prices, agents)
            return excess[1:]  # Rest goods only

        epsilon = 1e-8
        jacobian_numerical = np.zeros((1, 1))

        f0 = excess_demand_func(p_rest)
        for j in range(len(p_rest)):
            p_perturb = p_rest.copy()
            p_perturb[j] += epsilon
            f_perturb = excess_demand_func(p_perturb)
            jacobian_numerical[:, j] = (f_perturb - f0) / epsilon

        # Should match within numerical precision
        assert np.allclose(jacobian_analytical, jacobian_numerical, atol=1e-6)

    def test_jacobian_accuracy_three_goods(self):
        """Test analytical Jacobian for 3-good case."""
        agent1 = Agent(
            1,
            np.array([0.5, 0.3, 0.2]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        agent2 = Agent(
            2,
            np.array([0.2, 0.5, 0.3]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        agents = [agent1, agent2]

        solver = RobustEquilibriumSolver()
        p_rest = np.array([1.2, 0.8])  # Test point

        # Compute analytical Jacobian
        jacobian_analytical = solver._compute_analytical_jacobian(p_rest, agents, 3)

        # Compute numerical Jacobian
        def excess_demand_func(p):
            prices = np.concatenate([[1.0], p])
            excess = compute_excess_demand(prices, agents)
            return excess[1:]  # Rest goods only

        epsilon = 1e-8
        jacobian_numerical = np.zeros((2, 2))

        f0 = excess_demand_func(p_rest)
        for j in range(len(p_rest)):
            p_perturb = p_rest.copy()
            p_perturb[j] += epsilon
            f_perturb = excess_demand_func(p_perturb)
            jacobian_numerical[:, j] = (f_perturb - f0) / epsilon

        # Should match within numerical precision
        assert np.allclose(jacobian_analytical, jacobian_numerical, atol=1e-6)


@pytest.mark.robustness
@pytest.mark.real_functions
class TestFallbackMechanisms:
    """Test fallback behavior under challenging convergence scenarios."""

    def test_extreme_preferences_convergence(self):
        """Test convergence with extreme preferences that might challenge Newton-Raphson."""
        # Very skewed preferences (close to corner solutions)
        agent1 = Agent(
            1, np.array([0.99, 0.01]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.01, 0.99]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        prices, z_norm, walras, status, diagnostics = (
            solve_walrasian_equilibrium_robust(
                [agent1, agent2], return_diagnostics=True
            )
        )

        # Should converge, possibly with fallback methods
        assert status in ["converged", "poor_convergence"]
        assert z_norm < SOLVER_TOL * 10  # Allow relaxed tolerance

        # Should have attempted fallback if Newton-Raphson struggled
        if not diagnostics.success or diagnostics.method_used != "newton_raphson":
            assert len(diagnostics.fallback_attempts) > 0

    def test_method_selection_with_difficult_case(self):
        """Test that solver selects appropriate method for difficult cases."""
        # Create a case that might be challenging for Newton-Raphson
        agent1 = Agent(
            1, np.array([0.95, 0.05]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.05, 0.95]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        # Use poor initial guess to increase difficulty
        poor_guess = np.array([10.0])

        prices, z_norm, walras, status, diagnostics = (
            solve_walrasian_equilibrium_robust(
                [agent1, agent2], initial_guess=poor_guess, return_diagnostics=True
            )
        )

        # Should eventually converge
        assert status in ["converged", "poor_convergence"]

        # Should provide useful diagnostic information
        assert diagnostics.method_used in [
            "newton_raphson",
            "broyden",
            "tatonnement",
            "emergency_fallback",
        ]
        assert diagnostics.iterations >= 1

    def test_emergency_fallback_behavior(self):
        """Test emergency fallback when all methods fail."""
        # Create pathological case that should trigger emergency fallback
        # Note: This is difficult to construct reliably, so we test the mechanism

        solver = RobustEquilibriumSolver(
            tolerance=1e-15, max_iterations=5
        )  # Very strict tolerance, few iterations

        agent1 = Agent(
            1, np.array([0.999, 0.001]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.001, 0.999]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        result = solver.solve_equilibrium_robust(
            [agent1, agent2], return_diagnostics=True
        )
        prices, z_norm, walras, status, diagnostics = result

        # Should not crash even if convergence is poor
        assert prices is not None
        assert len(prices) == 2
        assert prices[0] == 1.0  # Numéraire constraint maintained
        assert status in ["converged", "poor_convergence", "failed"]


@pytest.mark.robustness
@pytest.mark.real_functions
class TestPerformanceAndScalability:
    """Test performance characteristics of the robust solver."""

    def test_convergence_speed_comparison(self):
        """Compare convergence speed of different methods."""
        # Standard case that should converge quickly
        agent1 = Agent(
            1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.4, 0.6]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        # Test with diagnostics to get iteration counts
        _, _, _, _, diagnostics = solve_walrasian_equilibrium_robust(
            [agent1, agent2], return_diagnostics=True
        )

        # Newton-Raphson should typically converge quickly for well-behaved cases
        if diagnostics.method_used == "newton_raphson":
            assert diagnostics.iterations <= 10  # Should be fast

        assert diagnostics.convergence_time < 1.0  # Should be fast (< 1 second)

    def test_larger_economy_scalability(self):
        """Test solver performance with larger number of agents."""
        # Set seed for deterministic behavior
        np.random.seed(42)

        # Create 5-agent, 3-good economy
        agents = []
        for i in range(5):
            alpha = np.random.dirichlet(np.ones(3))  # Random preferences summing to 1
            endowment = np.zeros(3)
            endowment[i % 3] = 1.0  # Each agent has one unit of different good
            agents.append(Agent(i + 1, alpha, endowment, np.array([0.0, 0.0, 0.0])))

        prices, z_norm, walras, status, diagnostics = (
            solve_walrasian_equilibrium_robust(agents, return_diagnostics=True)
        )

        assert status == "converged"
        assert z_norm < SOLVER_TOL
        assert diagnostics.convergence_time < 5.0  # Should complete in reasonable time


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestEconomicProperties:
    """Test that robust solver maintains economic properties."""

    def test_walras_law_satisfaction(self):
        """Test that Walras' Law is satisfied in equilibrium."""
        agent1 = Agent(
            1, np.array([0.7, 0.3]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        prices, z_norm, walras, status = solve_walrasian_equilibrium_robust(
            [agent1, agent2]
        )

        assert status == "converged"

        # Walras' Law: p·Z(p) = 0
        assert abs(walras) < SOLVER_TOL

        # Double-check by recomputing
        excess_demand = compute_excess_demand(prices, [agent1, agent2])
        walras_check = abs(np.dot(prices, excess_demand))
        assert walras_check < SOLVER_TOL

    def test_price_positivity(self):
        """Test that all equilibrium prices are positive."""
        agent1 = Agent(
            1, np.array([0.8, 0.2]), np.array([2.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.2, 0.8]), np.array([0.0, 3.0]), np.array([0.0, 0.0])
        )

        prices, z_norm, walras, status = solve_walrasian_equilibrium_robust(
            [agent1, agent2]
        )

        assert status == "converged"
        assert np.all(prices > 0)
        assert prices[0] == 1.0  # Numéraire constraint

    def test_market_clearing_consistency(self):
        """Test that computed prices actually clear the market."""
        agent1 = Agent(
            1, np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
        )
        agent2 = Agent(
            2, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )

        prices, z_norm, walras, status = solve_walrasian_equilibrium_robust(
            [agent1, agent2]
        )

        assert status == "converged"

        # Compute individual demands at equilibrium prices
        demand1 = agent1.demand(prices)
        demand2 = agent2.demand(prices)
        total_demand = demand1 + demand2

        # Compute total supply
        total_supply = agent1.total_endowment + agent2.total_endowment

        # Market clearing: total demand ≈ total supply
        assert np.allclose(total_demand, total_supply, atol=SOLVER_TOL)


if __name__ == "__main__":
    # Run robust solver tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
