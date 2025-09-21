"""
Enhanced unit tests for the Walrasian equilibrium solver with focus on economic validation.

This module extends the basic equilibrium tests with rigorous economic content validation:
- Analytical demand function verification against Cobb-Douglas theory
- Excess demand analytical validation (not just shape checking)
- Walras' Law numerical verification with theoretical computation
- Economic invariant enforcement throughout solving process
- Theoretical benchmark comparisons

Test Categories:
1. Analytical Economic Content Validation
2. Theoretical Benchmark Verification  
3. Economic Invariant Enforcement
4. Deep Economic Property Testing
5. Theoretical Consistency Checks
"""

import pytest
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from econ.equilibrium import (
    compute_excess_demand,
    solve_walrasian_equilibrium,
    validate_equilibrium_invariants,
    solve_equilibrium,
    SOLVER_TOL,
    FEASIBILITY_TOL,
    NUMERAIRE_GOOD,
)
from core.agent import Agent

# Import test categorization markers
from tests.test_categorization import economic_core, real_functions


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestAnalyticalDemandValidation:
    """Test that demand functions match analytical Cobb-Douglas formulas exactly."""

    def test_single_agent_demand_analytical_verification(self):
        """Test single agent demand against exact Cobb-Douglas formula."""
        # Agent with α = [0.7, 0.3], ω = [2, 1] at prices [1, 1.5]
        agent = Agent(
            agent_id=1,
            alpha=np.array([0.7, 0.3]),
            home_endowment=np.array([1.0, 0.5]),
            personal_endowment=np.array([1.0, 0.5]),
            position=(0, 0),
        )
        prices = np.array([1.0, 1.5])
        
        # Compute analytical Cobb-Douglas demand
        total_endowment = agent.total_endowment  # [2, 1] 
        wealth = np.dot(prices, total_endowment)  # 1*2 + 1.5*1 = 3.5
        
        # Theoretical demand: x_j = α_j * wealth / p_j
        analytical_demand = agent.alpha * wealth / prices
        expected_demand = np.array([0.7 * 3.5 / 1.0, 0.3 * 3.5 / 1.5])  # [2.45, 0.7]
        
        # Verify our analytical calculation
        assert np.allclose(analytical_demand, expected_demand, atol=1e-12)
        
        # Test agent.demand() method matches theory
        computed_demand = agent.demand(prices)
        assert np.allclose(computed_demand, analytical_demand, atol=1e-12), (
            f"Agent demand {computed_demand} != analytical {analytical_demand}"
        )
        
        # Test compute_excess_demand matches theory
        excess_demand = compute_excess_demand(prices, [agent])
        analytical_excess = analytical_demand - total_endowment
        assert np.allclose(excess_demand, analytical_excess, atol=1e-12), (
            f"Excess demand {excess_demand} != analytical {analytical_excess}"
        )

    def test_multi_agent_demand_aggregation(self):
        """Test that multi-agent excess demand aggregates correctly."""
        # Agent 1: α₁ = [0.6, 0.4], ω₁ = [3, 0]
        agent1 = Agent(
            1, np.array([0.6, 0.4]), np.array([1.5, 0.0]), np.array([1.5, 0.0])
        )
        
        # Agent 2: α₂ = [0.4, 0.6], ω₂ = [0, 2]  
        agent2 = Agent(
            2, np.array([0.4, 0.6]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
        
        prices = np.array([1.0, 2.0])
        agents = [agent1, agent2]
        
        # Compute analytical individual demands
        wealth1 = np.dot(prices, agent1.total_endowment)  # 1*3 + 2*0 = 3
        wealth2 = np.dot(prices, agent2.total_endowment)  # 1*0 + 2*2 = 4
        
        demand1 = agent1.alpha * wealth1 / prices  # [0.6*3/1, 0.4*3/2] = [1.8, 0.6]
        demand2 = agent2.alpha * wealth2 / prices  # [0.4*4/1, 0.6*4/2] = [1.6, 1.2]
        
        total_analytical_demand = demand1 + demand2  # [3.4, 1.8]
        total_endowments = agent1.total_endowment + agent2.total_endowment  # [3, 2]
        analytical_excess = total_analytical_demand - total_endowments  # [0.4, -0.2]
        
        # Test computed excess demand matches analytical
        computed_excess = compute_excess_demand(prices, agents)
        assert np.allclose(computed_excess, analytical_excess, atol=1e-12), (
            f"Computed excess {computed_excess} != analytical {analytical_excess}"
        )

    def test_wealth_calculation_consistency(self):
        """Test that wealth calculations are consistent across all functions."""
        agent = Agent(
            1, np.array([0.5, 0.5]), np.array([2.0, 1.5]), np.array([1.0, 0.5])
        )
        prices = np.array([1.0, 3.0])
        
        # Wealth should be computed from total endowment
        expected_wealth = np.dot(prices, agent.total_endowment)  # 1*3 + 3*2 = 9
        
        # Test agent.demand uses correct wealth
        demand = agent.demand(prices)
        analytical_demand = agent.alpha * expected_wealth / prices
        assert np.allclose(demand, analytical_demand, atol=1e-12)
        
        # Test compute_excess_demand uses same wealth calculation
        excess = compute_excess_demand(prices, [agent])
        expected_excess = analytical_demand - agent.total_endowment
        assert np.allclose(excess, expected_excess, atol=1e-12)


class TestWalrasLawAnalyticalValidation:
    """Test Walras' Law compliance through analytical computation."""

    def test_walras_law_theoretical_verification(self):
        """Test Walras' Law holds analytically for any price vector."""
        # Create 3 agents with different preferences
        agents = [
            Agent(1, np.array([0.7, 0.3]), np.array([2.0, 0.5]), np.array([1.0, 0.5])),
            Agent(2, np.array([0.3, 0.7]), np.array([0.5, 2.0]), np.array([0.5, 1.0])),
            Agent(3, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        ]
        
        # Test multiple price vectors
        price_vectors = [
            np.array([1.0, 1.0]),
            np.array([1.0, 2.0]), 
            np.array([1.0, 0.5]),
            np.array([1.0, 5.0]),
        ]
        
        for prices in price_vectors:
            # Compute excess demand
            excess_demand = compute_excess_demand(prices, agents)
            
            # Walras' Law: p · z = 0 should hold exactly
            walras_law_residual = np.dot(prices, excess_demand)
            
            assert abs(walras_law_residual) < 1e-12, (
                f"Walras' Law violated at prices {prices}: "
                f"p·z = {walras_law_residual}, should be ~0"
            )

    def test_equilibrium_walras_law_enforcement(self):
        """Test that equilibrium solutions strictly satisfy Walras' Law."""
        # Standard Edgeworth box setup
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([0.5, 0.0]), np.array([0.5, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 0.5]), np.array([0.0, 0.5]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        assert status == "converged"
        
        # Recompute excess demand at equilibrium prices
        excess_demand = compute_excess_demand(prices, [agent1, agent2])
        manual_walras_residual = np.dot(prices, excess_demand)
        
        # Both Walras residuals should be tiny
        assert abs(walras_dot) < SOLVER_TOL
        assert abs(manual_walras_residual) < SOLVER_TOL
        
        # They should also be nearly identical
        assert abs(walras_dot - manual_walras_residual) < 1e-12, (
            f"Walras residual mismatch: solver={walras_dot}, manual={manual_walras_residual}"
        )


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestEquilibriumAnalyticalBenchmarks:
    """Test equilibrium solutions against known analytical benchmarks."""

    def test_edgeworth_box_demand_verification(self):
        """Test that equilibrium demands match analytical Edgeworth solution."""
        # Standard Edgeworth box: Agent 1 = [0.6, 0.4], [1, 0]; Agent 2 = [0.3, 0.7], [0, 1]
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([0.5, 0.0]), np.array([0.5, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 0.5]), np.array([0.0, 0.5]))
        
        prices, _, _, status = solve_walrasian_equilibrium([agent1, agent2])
        assert status == "converged"
        
        # Analytical solution: p* = [1, 4/3], demands x₁* = [0.6, 0.3], x₂* = [0.4, 0.7]
        expected_prices = np.array([1.0, 4.0/3.0])
        assert np.allclose(prices, expected_prices, atol=1e-8)
        
        # Verify demands at equilibrium prices
        demand1 = agent1.demand(prices)
        demand2 = agent2.demand(prices)
        
        expected_demand1 = np.array([0.6, 0.3])
        expected_demand2 = np.array([0.4, 0.7])
        
        assert np.allclose(demand1, expected_demand1, atol=1e-8), (
            f"Agent 1 demand {demand1} != expected {expected_demand1}"
        )
        assert np.allclose(demand2, expected_demand2, atol=1e-8), (
            f"Agent 2 demand {demand2} != expected {expected_demand2}"
        )
        
        # Verify market clearing: total demand = total endowment
        total_demand = demand1 + demand2
        total_endowment = agent1.total_endowment + agent2.total_endowment
        assert np.allclose(total_demand, total_endowment, atol=1e-12)

    def test_symmetric_agents_equal_prices(self):
        """Test that symmetric agents in symmetric economy produce equal prices."""
        # Two identical agents with identical endowments → prices should be [1, 1]
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, _, _, status = solve_walrasian_equilibrium([agent1, agent2])
        assert status == "converged"
        
        # With symmetric preferences and endowments, relative price should be 1
        expected_prices = np.array([1.0, 1.0])
        assert np.allclose(prices, expected_prices, atol=1e-8)


@pytest.mark.robustness
@pytest.mark.real_functions  
class TestEquilibriumRobustness:
    """Test equilibrium solver robustness under extreme conditions."""

    def test_corner_solution_handling(self):
        """Test equilibrium computation with extreme preferences (near corner solutions)."""
        # Agent 1 heavily prefers good 1, Agent 2 heavily prefers good 2
        agent1 = Agent(1, np.array([0.95, 0.05]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.05, 0.95]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        
        prices, _, _, status = solve_walrasian_equilibrium([agent1, agent2])
        assert status in ["converged", "poor_convergence"]  # May struggle with extreme case
        
        if status == "converged":
            # Verify demands are economically sensible
            demand1 = agent1.demand(prices)
            demand2 = agent2.demand(prices)
            
            # Agent 1 should demand mostly good 1
            assert demand1[0] > demand1[1]
            # Agent 2 should demand mostly good 2  
            assert demand2[1] > demand2[0]


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestNumeraireConstraintValidation:
    """Test rigorous enforcement of numéraire constraint throughout solving."""

    def test_numeraire_preserved_across_solver_iterations(self):
        """Test that numéraire constraint p[0] = 1 is maintained precisely."""
        agents = [
            Agent(1, np.array([0.7, 0.3]), np.array([2.0, 0.0]), np.array([0.0, 0.0])),
            Agent(2, np.array([0.4, 0.6]), np.array([0.0, 1.5]), np.array([0.0, 0.0])),
        ]
        
        # Test with different initial guesses
        initial_guesses = [None, np.array([0.5]), np.array([2.0]), np.array([10.0])]
        
        for initial_guess in initial_guesses:
            prices, _, _, status = solve_walrasian_equilibrium(agents, initial_guess)
            
            if status == "converged":
                # Numéraire should be exactly 1.0 (not just approximately)
                assert abs(prices[0] - 1.0) < 1e-15, (
                    f"Numéraire not preserved: p[0] = {prices[0]} != 1.0"
                )

    def test_numeraire_validation_in_excess_demand(self):
        """Test that compute_excess_demand enforces numéraire validation."""
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        # Valid numéraire price
        excess = compute_excess_demand(np.array([1.0, 2.0]), [agent])
        assert isinstance(excess, np.ndarray)
        
        # Invalid numéraire should raise AssertionError
        with pytest.raises(AssertionError, match="Numéraire violated"):
            compute_excess_demand(np.array([0.5, 2.0]), [agent])
        
        with pytest.raises(AssertionError, match="Numéraire violated"):
            compute_excess_demand(np.array([2.0, 2.0]), [agent])


class TestBudgetConstraintEnforcement:
    """Test that budget constraints are correctly enforced in all computations."""

    def test_agent_demand_budget_feasibility(self):
        """Test that agent demands satisfy budget constraint exactly."""
        agent = Agent(1, np.array([0.6, 0.4]), np.array([2.0, 1.0]), np.array([1.0, 0.5]))
        prices = np.array([1.0, 3.0])
        
        demand = agent.demand(prices)
        wealth = np.dot(prices, agent.total_endowment)
        expenditure = np.dot(prices, demand)
        
        # Budget constraint: p · x = wealth (exactly, not approximately)
        assert abs(expenditure - wealth) < 1e-12, (
            f"Budget constraint violated: expenditure={expenditure}, wealth={wealth}"
        )

    def test_market_value_conservation(self):
        """Test that total market value is conserved in equilibrium."""
        agents = [
            Agent(1, np.array([0.7, 0.3]), np.array([2.0, 0.5]), np.array([1.0, 0.5])),
            Agent(2, np.array([0.3, 0.7]), np.array([0.5, 2.0]), np.array([0.5, 1.0])),
        ]
        
        prices, _, _, status = solve_walrasian_equilibrium(agents)
        assert status == "converged"
        
        # Total value of initial endowments
        total_initial_value = sum(np.dot(prices, agent.total_endowment) for agent in agents)
        
        # Total value of equilibrium demands
        total_demand_value = sum(np.dot(prices, agent.demand(prices)) for agent in agents)
        
        # Should be exactly equal (conservation of value)
        assert abs(total_initial_value - total_demand_value) < 1e-12, (
            f"Value not conserved: initial={total_initial_value}, demand={total_demand_value}"
        )


class TestConvergenceCriteriaValidation:
    """Test that convergence criteria have economic meaning and are properly enforced."""

    def test_convergence_implies_small_excess_demand(self):
        """Test that convergence status guarantees economically meaningful residuals."""
        agents = [
            Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0])),
            Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0])),
        ]
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
        
        if status == "converged":
            # Compute excess demand at solution
            excess_demand = compute_excess_demand(prices, agents)
            
            # All excess demands should be economically negligible
            max_excess = np.max(np.abs(excess_demand))
            assert max_excess < SOLVER_TOL, (
                f"Convergence claimed but large excess demand: max |z| = {max_excess}"
            )
            
            # Specifically for rest goods (non-numéraire)
            assert z_rest_norm < SOLVER_TOL
            assert np.max(np.abs(excess_demand[1:])) <= z_rest_norm + 1e-15

    def test_poor_convergence_detection(self):
        """Test that poor convergence is properly detected and flagged."""
        # Create a challenging case that might not converge well
        agent1 = Agent(1, np.array([0.999, 0.001]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.001, 0.999]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        # If convergence is poor, it should be detected
        if z_rest_norm >= SOLVER_TOL or abs(walras_dot) >= SOLVER_TOL:
            assert status == "poor_convergence", (
                f"Poor convergence not detected: status={status}, z_norm={z_rest_norm}, walras={walras_dot}"
            )


if __name__ == "__main__":
    # Run enhanced tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])