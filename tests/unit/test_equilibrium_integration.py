"""
Test the integration layer for robust equilibrium solving.
"""

import pytest
import numpy as np
from src.core.agent import Agent
from src.econ.equilibrium_integration import (
    solve_equilibrium_enhanced,
    solve_equilibrium_with_diagnostics,
    compare_solver_performance
)


@pytest.mark.economic_core
@pytest.mark.real_functions
class TestEquilibriumIntegration:
    """Test the integration between original and robust solvers."""
    
    def test_enhanced_solver_with_robust_preference(self):
        """Test enhanced solver defaults to robust method."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
            [agent1, agent2], use_robust_solver=True
        )
        
        assert status in ['converged', 'poor_convergence']
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 2
        assert prices[0] == pytest.approx(1.0, abs=1e-10)  # Numéraire
        assert np.all(prices > 0)
    
    def test_enhanced_solver_original_preference(self):
        """Test enhanced solver can use original method."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
            [agent1, agent2], use_robust_solver=False
        )
        
        assert status == 'converged'
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 2
        assert prices[0] == pytest.approx(1.0, abs=1e-10)  # Numéraire
    
    def test_solver_with_diagnostics(self):
        """Test diagnostic information collection."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        prices, z_norm, walras_dot, status, diagnostics = solve_equilibrium_with_diagnostics(
            [agent1, agent2]
        )
        
        assert status in ['converged', 'poor_convergence']
        assert isinstance(diagnostics, dict)
        
        # Check required diagnostic fields
        required_fields = ['converged', 'method_used', 'final_residual', 'status']
        for field in required_fields:
            assert field in diagnostics
        
        assert isinstance(diagnostics['converged'], bool)
        assert isinstance(diagnostics['method_used'], str)
        assert isinstance(diagnostics['final_residual'], (int, float))
        assert isinstance(diagnostics['status'], str)
    
    def test_performance_comparison(self):
        """Test performance comparison between solvers."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        results = compare_solver_performance([agent1, agent2], num_trials=3)
        
        assert isinstance(results, dict)
        assert 'original_solver' in results
        assert 'robust_solver' in results
        
        for solver in ['original_solver', 'robust_solver']:
            solver_results = results[solver]
            assert 'times' in solver_results
            assert 'successes' in solver_results
            assert 'success_rate' in solver_results
            assert isinstance(solver_results['success_rate'], float)
            assert 0.0 <= solver_results['success_rate'] <= 1.0
    
    def test_fallback_behavior(self):
        """Test fallback from robust to original solver."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        # This should work with either solver
        prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
            [agent1, agent2], 
            use_robust_solver=True, 
            fallback_to_original=True
        )
        
        assert status in ['converged', 'poor_convergence']
        assert isinstance(prices, np.ndarray)
        assert np.all(prices > 0)
        
    def test_no_fallback_behavior(self):
        """Test behavior when fallback is disabled."""
        agent1 = Agent(1, np.array([0.6, 0.4]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        agent2 = Agent(2, np.array([0.3, 0.7]), np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        
        # This should still work for a well-behaved case
        prices, z_norm, walras_dot, status = solve_equilibrium_enhanced(
            [agent1, agent2], 
            use_robust_solver=True, 
            fallback_to_original=False
        )
        
        assert status in ['converged', 'poor_convergence', 'failed']
        assert isinstance(prices, np.ndarray)