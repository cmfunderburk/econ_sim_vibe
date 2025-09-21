"""
Unit tests for the Walrasian equilibrium solver.

This module tests the core economic engine with focus on:
- Economic invariant validation (Walras' Law, conservation, numéraire)
- Numerical convergence and stability
- Edge cases and error handling
- V1 Edgeworth box analytical verification
- Performance and scalability targets

Test Categories:
1. Basic functionality and API contracts
2. Economic invariants and theoretical properties
3. Numerical stability and convergence
4. Edge cases and error conditions
5. Integration with Agent framework
6. Performance benchmarks
"""

import pytest
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from econ.equilibrium import (
    compute_excess_demand, 
    solve_walrasian_equilibrium,
    validate_equilibrium_invariants,
    solve_equilibrium,
    SOLVER_TOL,
    FEASIBILITY_TOL,
    NUMERAIRE_GOOD
)
from core.agent import Agent


class TestComputeExcessDemand:
    """Test the excess demand computation function."""
    
    def test_basic_excess_demand_computation(self):
        """Test basic excess demand calculation with valid inputs."""
        # Create simple 2-good agents
        agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.6, 0.4]),
            home_endowment=np.array([0.5, 0.0]),
            personal_endowment=np.array([0.5, 0.0]),
            position=(0, 0)
        )
        agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.3, 0.7]),
            home_endowment=np.array([0.0, 0.5]),
            personal_endowment=np.array([0.0, 0.5]),
            position=(0, 0)
        )
        
        agents = [agent1, agent2]
        prices = np.array([1.0, 1.0])  # Equal prices
        
        excess_demand = compute_excess_demand(prices, agents)
        
        # Basic sanity checks
        assert isinstance(excess_demand, np.ndarray)
        assert excess_demand.shape == (2,)
        assert np.isfinite(excess_demand).all()
        
    def test_numeraire_constraint_validation(self):
        """Test that excess demand function validates numéraire constraint."""
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        # Should work with p[0] = 1.0
        excess_demand = compute_excess_demand(np.array([1.0, 2.0]), [agent])
        assert isinstance(excess_demand, np.ndarray)
        
        # Should fail with p[0] != 1.0
        with pytest.raises(AssertionError, match="Numéraire violated"):
            compute_excess_demand(np.array([2.0, 1.0]), [agent])
    
    def test_empty_agents_list(self):
        """Test that empty agents list raises appropriate error."""
        with pytest.raises(AssertionError, match="No participants in market"):
            compute_excess_demand(np.array([1.0, 1.0]), [])
    
    def test_zero_wealth_agent_handling(self):
        """Test that agents with zero wealth are handled gracefully."""
        # Agent with zero endowment (zero wealth)
        zero_agent = Agent(1, np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        normal_agent = Agent(2, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        # Should work with warning logged (zero wealth agent excluded)
        excess_demand = compute_excess_demand(np.array([1.0, 1.0]), [zero_agent, normal_agent])
        assert isinstance(excess_demand, np.ndarray)
    
    def test_price_floor_protection(self):
        """Test that price floors prevent division by zero."""
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        # Very small prices should be floored
        excess_demand = compute_excess_demand(np.array([1.0, 1e-12]), [agent])
        assert np.isfinite(excess_demand).all()
        
    def test_cobb_douglas_demand_formula(self):
        """Test that Cobb-Douglas demand formula is implemented correctly."""
        # Agent with α = [0.6, 0.4], ω = [1, 1], wealth = 3 at prices [1, 2]
        agent = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        prices = np.array([1.0, 2.0])
        
        excess_demand = compute_excess_demand(prices, [agent])
        
        # Expected demand: x1 = 0.6 * 3 / 1 = 1.8, x2 = 0.4 * 3 / 2 = 0.6
        # Expected excess demand: [1.8 - 1, 0.6 - 1] = [0.8, -0.4]
        expected_excess = np.array([0.8, -0.4])
        
        assert np.allclose(excess_demand, expected_excess, atol=1e-10)


class TestSolveWalrasianEquilibrium:
    """Test the main equilibrium solver function."""
    
    def test_edgeworth_box_analytical_solution(self):
        """Test V1: Verify against known Edgeworth box analytical solution."""
        # Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]  
        # Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]
        # Correct analytical equilibrium: p* = [1, 4/3], x₁* = [0.6, 0.3], x₂* = [0.4, 0.7]
        
        agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.6, 0.4]),
            home_endowment=np.array([0.5, 0.0]),  # Split total endowment
            personal_endowment=np.array([0.5, 0.0]),
            position=(0, 0)
        )
        agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.3, 0.7]),
            home_endowment=np.array([0.0, 0.5]),
            personal_endowment=np.array([0.0, 0.5]),
            position=(0, 0)
        )
        
        agents = [agent1, agent2]
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
        
        # Test convergence
        assert status == 'converged'
        assert z_rest_norm < SOLVER_TOL
        assert abs(walras_dot) < SOLVER_TOL
        
        # Test analytical solution: p* = [1, 4/3]
        expected_prices = np.array([1.0, 4.0/3.0])
        assert np.allclose(prices, expected_prices, atol=1e-8)
        
    def test_numerical_convergence_criteria(self):
        """Test that convergence criteria are properly enforced."""
        # Simple 2-agent, 2-good case that should converge
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        assert status == 'converged'
        assert z_rest_norm < SOLVER_TOL
        assert abs(walras_dot) < SOLVER_TOL
        assert prices[0] == 1.0  # Numéraire constraint
        
    def test_insufficient_participants_handling(self):
        """Test handling of edge cases with insufficient participants."""
        # No agents
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([])
        assert prices is None
        assert status == 'no_participants'
        
        # Single agent
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent])
        assert prices is None
        assert status == 'insufficient_participants'
        
    def test_single_good_edge_case(self):
        """Test handling of single good case (no relative prices)."""
        # Agents with single good
        agent1 = Agent(1, np.array([1.0]), np.array([1.0]), np.array([0.0]))
        agent2 = Agent(2, np.array([1.0]), np.array([1.0]), np.array([0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        assert prices is None
        assert status == 'insufficient_participants'
        
    def test_zero_wealth_agent_filtering(self):
        """Test that zero-wealth agents are filtered out."""
        # One viable agent, one zero-wealth agent
        viable_agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        zero_agent = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([viable_agent, zero_agent])
        assert prices is None  # Only one viable agent left
        assert status == 'insufficient_viable_agents'
        
    def test_initial_guess_handling(self):
        """Test that initial guess parameter works correctly."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        # Solve without initial guess
        prices1, _, _, status1 = solve_walrasian_equilibrium([agent1, agent2])
        assert status1 == 'converged'
        
        # Solve with initial guess close to solution
        initial_guess = np.array([0.8])  # For second good price
        prices2, _, _, status2 = solve_walrasian_equilibrium([agent1, agent2], initial_guess)
        assert status2 == 'converged'
        
        # Should get same result regardless of initial guess
        assert np.allclose(prices1, prices2, atol=1e-8)


class TestValidateEquilibriumInvariants:
    """Test the economic invariant validation function."""
    
    def test_all_invariants_satisfied(self):
        """Test validation when all invariants are satisfied."""
        prices = np.array([1.0, 2.0])
        agents = []  # Not used in this test
        excess_demand = np.array([1e-10, -1e-10])  # Near-zero excess demand
        
        result = validate_equilibrium_invariants(prices, agents, excess_demand)
        assert result is True
        
    def test_numeraire_violation(self):
        """Test detection of numéraire constraint violation."""
        prices = np.array([0.5, 2.0])  # p[0] != 1.0
        agents = []
        excess_demand = np.array([0.0, 0.0])
        
        result = validate_equilibrium_invariants(prices, agents, excess_demand)
        assert result is False
        
    def test_walras_law_violation(self):
        """Test detection of Walras' Law violation."""
        prices = np.array([1.0, 2.0])
        agents = []
        excess_demand = np.array([1.0, 1.0])  # Large excess demand
        
        result = validate_equilibrium_invariants(prices, agents, excess_demand)
        assert result is False
        
    def test_negative_price_detection(self):
        """Test detection of negative prices."""
        prices = np.array([1.0, -0.5])  # Negative price
        agents = []
        excess_demand = np.array([0.0, 0.0])
        
        result = validate_equilibrium_invariants(prices, agents, excess_demand)
        assert result is False
        
    def test_poor_convergence_detection(self):
        """Test detection of poor convergence."""
        prices = np.array([1.0, 2.0])
        agents = []
        excess_demand = np.array([0.0, 1e-6])  # Large rest-goods residual
        
        result = validate_equilibrium_invariants(prices, agents, excess_demand)
        assert result is False


class TestSolveEquilibriumInterface:
    """Test the high-level solve_equilibrium interface."""
    
    def test_default_parameters(self):
        """Test solve_equilibrium with default parameters."""
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_equilibrium([agent1, agent2])
        
        assert status == 'converged'
        assert prices is not None
        assert prices[0] == 1.0
        
    def test_unsupported_normalization(self):
        """Test error handling for unsupported normalization schemes."""
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        with pytest.raises(NotImplementedError, match="Normalization 'unsupported'"):
            solve_equilibrium([agent], normalization='unsupported')
            
    def test_unsupported_endowment_scope(self):
        """Test error handling for unsupported endowment scopes."""
        agent = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        
        with pytest.raises(NotImplementedError, match="Endowment scope 'personal'"):
            solve_equilibrium([agent], endowment_scope='personal')


class TestEconomicInvariants:
    """Test critical economic invariants that must always hold."""
    
    def test_walras_law_always_holds(self):
        """Test that Walras' Law holds for any equilibrium."""
        # Multiple test cases with different agent configurations
        test_cases = [
            # Case 1: Symmetric agents
            [
                Agent(1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0])),
                Agent(2, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
            ],
            # Case 2: Asymmetric preferences
            [
                Agent(1, np.array([0.8, 0.2]), np.array([2.0, 0.0]), np.array([0.0, 0.0])),
                Agent(2, np.array([0.2, 0.8]), np.array([0.0, 2.0]), np.array([0.0, 0.0]))
            ],
            # Case 3: Three agents
            [
                Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0])),
                Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0])),
                Agent(3, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
            ]
        ]
        
        for agents in test_cases:
            prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
            
            if status == 'converged':
                # Walras' Law must hold
                assert abs(walras_dot) < SOLVER_TOL
                
                # Recompute excess demand to double-check
                excess_demand = compute_excess_demand(prices, agents)
                walras_check = abs(np.dot(prices, excess_demand))
                assert walras_check < SOLVER_TOL
                
    def test_conservation_of_total_endowments(self):
        """Test that total endowments are conserved in demand computation."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([0.8, 0.1]), np.array([0.2, 0.1]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.1, 0.8]), np.array([0.1, 0.2]))
        
        agents = [agent1, agent2]
        prices = np.array([1.0, 1.5])
        
        # Total initial endowments
        total_initial = sum(agent.home_endowment + agent.personal_endowment for agent in agents)
        
        # Compute demands
        total_demand = np.zeros(2)
        for agent in agents:
            omega_total = agent.home_endowment + agent.personal_endowment
            wealth = np.dot(prices, omega_total)
            demand = agent.alpha * wealth / prices
            total_demand += demand
        
        # In equilibrium, total demand should equal total endowments
        # (This test checks the demand computation logic)
        excess_demand = compute_excess_demand(prices, agents)
        computed_total_demand = total_initial + excess_demand
        
        assert np.allclose(total_demand, computed_total_demand, atol=FEASIBILITY_TOL)
        
    def test_numeraire_constraint_preserved(self):
        """Test that numéraire constraint is always preserved."""
        agent1 = Agent(1, np.array([0.7, 0.3]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.4, 0.6]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        # Multiple solver runs with different initial guesses
        initial_guesses = [None, np.array([0.5]), np.array([2.0]), np.array([10.0])]
        
        for initial_guess in initial_guesses:
            prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(
                [agent1, agent2], initial_guess
            )
            
            if status == 'converged':
                assert abs(prices[0] - 1.0) < 1e-12


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    def test_small_scale_performance(self):
        """Test performance with small number of agents (target: <1 second)."""
        import time
        
        # Create 10 agents with random preferences
        np.random.seed(42)  # Reproducible
        agents = []
        for i in range(10):
            alpha = np.random.dirichlet([1, 1, 1])  # 3 goods
            home_endow = np.random.exponential(1.0, 3)
            personal_endow = np.random.exponential(0.1, 3)
            agents.append(Agent(i, alpha, home_endow, personal_endow))
        
        start_time = time.time()
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
        elapsed = time.time() - start_time
        
        assert status == 'converged'
        assert elapsed < 1.0  # Should solve quickly
        
    def test_convergence_iteration_bounds(self):
        """Test that solver converges within reasonable iterations."""
        # This is implicitly tested by scipy.optimize.fsolve maxfev parameter
        # If solver takes too many iterations, it will return 'failed' status
        
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        # Should converge without hitting iteration limit
        assert status == 'converged'
        assert z_rest_norm < SOLVER_TOL


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness to edge cases."""
    
    def test_extreme_preference_values(self):
        """Test handling of extreme preference parameter values."""
        # Very skewed preferences (close to corner solutions)
        agent1 = Agent(1, np.array([0.99, 0.01]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.01, 0.99]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        # Should still converge despite extreme preferences
        assert status in ['converged', 'poor_convergence']
        if status == 'converged':
            assert z_rest_norm < SOLVER_TOL
            
    def test_large_endowment_disparities(self):
        """Test handling of large disparities in endowments."""
        # One agent with large endowment, one with small
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1000.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 0.001]), np.array([0.0, 0.0]))
        
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        # Should handle large disparities
        assert status in ['converged', 'poor_convergence']
        if status == 'converged':
            assert np.isfinite(prices).all()
            assert np.all(prices > 0)
            
    def test_numerical_precision_edge_cases(self):
        """Test handling of numerical precision edge cases."""
        # Very small endowments (near machine precision)
        agent1 = Agent(1, np.array([0.5, 0.5]), np.array([1e-10, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.5, 0.5]), np.array([0.0, 1e-10]), np.array([0.0, 0.0]))
        
        # Should either converge or gracefully handle the edge case
        prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium([agent1, agent2])
        
        # Status should be one of the expected outcomes
        assert status in ['converged', 'poor_convergence', 'insufficient_viable_agents']


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])