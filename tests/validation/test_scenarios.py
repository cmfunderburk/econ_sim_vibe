"""
Validation test suite for economic scenarios V1-V10.

These tests validate the core economic properties and theoretical correctness
of the simulation against known analytical results and economic theory.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.agent import Agent
from econ.equilibrium import solve_walrasian_equilibrium
from econ.market import execute_constrained_clearing
from core.types import MarketResult, Trade
from tests import load_config, SOLVER_TOL, FEASIBILITY_TOL


def test_v1_edgeworth_2x2():
    """
    V1: Edgeworth Box 2×2 Analytical Verification
    
    Tests our Economic Engine against the known analytical solution for the classic
    2-agent, 2-good Edgeworth box economy.
    
    Setup:
    - Agent 1: preferences α₁ = [0.6, 0.4], endowment ω₁ = [1, 0]
    - Agent 2: preferences α₂ = [0.3, 0.7], endowment ω₂ = [0, 1]
    
    Analytical Solution:
    For Cobb-Douglas agents, the equilibrium price ratio is determined by:
    p₂/p₁ = (α₁₂/α₁₁) * (ω₁₁/ω₁₂) * (α₂₁/α₂₂) * (ω₂₂/ω₂₁)
    
    With our parameters:
    p₂/p₁ = (0.4/0.6) * (1/0) → Need to solve the full system
    
    Using market clearing conditions and Cobb-Douglas demands:
    x₁₁ = α₁₁ * (p₁*ω₁₁ + p₂*ω₁₂) / p₁ = 0.6 * p₁ / p₁ = 0.6
    x₁₂ = α₁₂ * (p₁*ω₁₁ + p₂*ω₁₂) / p₂ = 0.4 * p₁ / p₂
    x₂₁ = α₂₁ * (p₁*ω₂₁ + p₂*ω₂₂) / p₁ = 0.3 * p₂ / p₁
    x₂₂ = α₂₂ * (p₁*ω₂₁ + p₂*ω₂₂) / p₂ = 0.7 * p₂ / p₂ = 0.7
    
    Market clearing: x₁₁ + x₂₁ = 1, x₁₂ + x₂₂ = 1
    0.6 + 0.3 * p₂/p₁ = 1  →  p₂/p₁ = 4/3
    0.4 * p₁/p₂ + 0.7 = 1   →  p₁/p₂ = 3/4  →  p₂/p₁ = 4/3 ✓
    
    Therefore: p* = [1, 4/3] with numéraire p₁ ≡ 1
    
    Expected: ‖p_computed - p_analytic‖ < 1e-8
    """
    print("\n=== V1: Edgeworth Box 2×2 Analytical Verification ===")
    
    # Create agents with specified preferences and endowments
    alpha_1 = np.array([0.6, 0.4])
    alpha_2 = np.array([0.3, 0.7])
    
    # Agent endowments - all in personal inventory for trading
    endowment_1 = np.array([1.0, 0.0])
    endowment_2 = np.array([0.0, 1.0])
    
    agent_1 = Agent(
        agent_id=1,
        alpha=alpha_1,
        home_endowment=np.zeros(2),  # Empty home inventory
        personal_endowment=endowment_1,  # All goods available for trading
        position=(0, 0)  # Both agents at marketplace
    )
    
    agent_2 = Agent(
        agent_id=2, 
        alpha=alpha_2,
        home_endowment=np.zeros(2),  # Empty home inventory
        personal_endowment=endowment_2,  # All goods available for trading
        position=(0, 0)  # Both agents at marketplace
    )
    
    agents = [agent_1, agent_2]
    
    # Solve for equilibrium prices
    print("Computing equilibrium prices...")
    prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
    
    print(f"Solver status: {status}")
    print(f"Computed prices: {prices}")
    print(f"Rest-goods norm: {z_rest_norm}")
    print(f"Walras' Law residual: {walras_dot}")
    
    # Verify solver converged
    assert status == "converged", f"Solver failed with status: {status}"
    assert z_rest_norm < SOLVER_TOL, f"Poor convergence: ||Z_rest||_∞ = {z_rest_norm}"
    assert abs(walras_dot) < SOLVER_TOL, f"Walras' Law violated: |p·Z| = {abs(walras_dot)}"
    
    # Known analytical solution
    expected_prices = np.array([1.0, 4.0/3.0])
    
    print(f"Expected prices: {expected_prices}")
    print(f"Price difference: {np.linalg.norm(prices - expected_prices)}")
    
    # Primary validation: price accuracy
    price_error = np.linalg.norm(prices - expected_prices)
    assert price_error < SOLVER_TOL, f"Price error too large: {price_error} >= {SOLVER_TOL}"
    
    # Verify optimal consumption bundles
    wealth_1 = np.dot(prices, agent_1.total_endowment)
    wealth_2 = np.dot(prices, agent_2.total_endowment)
    
    consumption_1 = agent_1.demand(prices, wealth_1)
    consumption_2 = agent_2.demand(prices, wealth_2)
    
    print(f"Agent 1 wealth: {wealth_1:.6f}, consumption: {consumption_1}")
    print(f"Agent 2 wealth: {wealth_2:.6f}, consumption: {consumption_2}")
    
    # Expected consumption from analytical solution
    # Agent 1: x₁ = [0.6, 0.4 * 1 / (4/3)] = [0.6, 0.3]
    # Agent 2: x₂ = [0.3 * (4/3) / 1, 0.7] = [0.4, 0.7]
    expected_consumption_1 = np.array([0.6, 0.3])
    expected_consumption_2 = np.array([0.4, 0.7])
    
    consumption_error_1 = np.linalg.norm(consumption_1 - expected_consumption_1)
    consumption_error_2 = np.linalg.norm(consumption_2 - expected_consumption_2)
    
    print(f"Expected consumption 1: {expected_consumption_1}, error: {consumption_error_1:.2e}")
    print(f"Expected consumption 2: {expected_consumption_2}, error: {consumption_error_2:.2e}")
    
    assert consumption_error_1 < SOLVER_TOL, f"Agent 1 consumption error: {consumption_error_1}"
    assert consumption_error_2 < SOLVER_TOL, f"Agent 2 consumption error: {consumption_error_2}"
    
    # Verify market clearing (total consumption = total endowment)
    total_consumption = consumption_1 + consumption_2
    total_endowment = agent_1.total_endowment + agent_2.total_endowment
    
    clearing_error = np.linalg.norm(total_consumption - total_endowment)
    print(f"Market clearing error: {clearing_error:.2e}")
    assert clearing_error < FEASIBILITY_TOL, f"Market clearing violated: {clearing_error}"
    
    # Test complete trading pipeline with market clearing
    print("\nTesting complete trading pipeline...")
    
    market_result = execute_constrained_clearing(agents, prices)
    
    print(f"Market clearing efficiency: {market_result.clearing_efficiency:.6f}")
    print(f"Number of trades executed: {len(market_result.executed_trades)}")
    
    # For this perfect scenario, we should get very high efficiency
    assert market_result.clearing_efficiency > 0.99, f"Low clearing efficiency: {market_result.clearing_efficiency}"
    
    print("✅ V1 Edgeworth Box 2×2 validation PASSED")
    print(f"   Price accuracy: {price_error:.2e} < {SOLVER_TOL}")
    print(f"   Consumption accuracy: max({consumption_error_1:.2e}, {consumption_error_2:.2e}) < {SOLVER_TOL}")
    print(f"   Market clearing: {clearing_error:.2e} < {FEASIBILITY_TOL}")
    print(f"   Trading efficiency: {market_result.clearing_efficiency:.4f}")


def test_v2_spatial_null():
    """V2: Spatial Null - Phase 2 should equal Phase 1 exactly."""
    config = load_config("zero_movement_cost")
    # TODO: Implement efficiency comparison test
    # Expected: efficiency_loss < 1e-10
    pytest.skip("Implementation pending")

def test_v3_market_access():
    """V3: Market Access - Efficiency loss vs baseline."""
    config = load_config("small_market")
    # TODO: Implement efficiency loss test
    # Expected: efficiency_loss > 0.1 units of good 1
    pytest.skip("Implementation pending")

def test_v4_throughput_cap():
    """V4: Throughput Cap - Queue formation and carry-over orders."""
    config = load_config("rationed_market")
    # TODO: Implement queue formation test
    # Expected: uncleared_orders > 0
    pytest.skip("Implementation pending")

def test_v5_spatial_dominance():
    """V5: Spatial Dominance - Phase 2 efficiency ≤ Phase 1."""
    config = load_config("infinite_movement_cost")
    # TODO: Implement welfare dominance test
    # Expected: spatial_welfare ≤ walrasian_welfare
    pytest.skip("Implementation pending")

def test_v6_price_normalization():
    """V6: Price Normalization - p₁ ≡ 1 and rest-goods convergence."""
    config = load_config("price_validation")
    # TODO: Implement numerical stability test
    # Expected: p[0] == 1.0 and ||Z_market(p)[1:]||_∞ < 1e-8
    pytest.skip("Implementation pending")

def test_v7_empty_marketplace():
    """V7: Empty Marketplace - Edge case handling."""
    config = load_config("empty_market")
    # TODO: Implement edge case test
    # Expected: prices == None and trades == []
    pytest.skip("Implementation pending")

def test_v8_stop_conditions():
    """V8: Stop Conditions - Termination logic validation."""
    config = load_config("termination")
    # TODO: Implement termination logic test
    # Expected: Proper termination reasons
    pytest.skip("Implementation pending")

def test_v9_scale_invariance():
    """V9: Scale Invariance - Price scaling preserves allocation."""
    config = load_config("scale_test")
    # TODO: Implement scale invariance test
    # Expected: Identical demand after rescaling
    pytest.skip("Implementation pending")

def test_v10_spatial_null_unit():
    """V10: Spatial Null (Unit Test) - Perfect equivalence."""
    config = load_config("spatial_null_test")
    # TODO: Implement perfect equivalence test
    # Expected: phase2_allocation == phase1_allocation
    pytest.skip("Implementation pending")