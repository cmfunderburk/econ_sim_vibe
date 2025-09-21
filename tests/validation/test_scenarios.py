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
    """V2: Spatial Null - Phase 2 should equal Phase 1 exactly.
    
    When movement costs κ=0, Phase 2 spatial simulation should produce
    exactly the same welfare as Phase 1 pure Walrasian equilibrium.
    This validates that spatial extensions don't break baseline economics.
    """
    print("\n=== V2: Spatial Null Test (κ=0) ===")
    
    # Create test agents for Phase 1 and Phase 2 comparison
    n_agents = 10
    n_goods = 3
    np.random.seed(42)  # Deterministic test
    
    # Generate identical agent setup for both phases
    agents_phase1 = []
    agents_phase2 = []
    
    for i in range(n_agents):
        # Generate Cobb-Douglas preferences with min 0.05 and renormalization
        alpha_raw = np.random.dirichlet(np.ones(n_goods))
        alpha = np.maximum(alpha_raw, 0.05)
        alpha = alpha / np.sum(alpha)  # Renormalize to sum to 1
        
        # Generate random positive endowments
        total_endowment = np.random.uniform(0.1, 2.0, n_goods)
        
        # For Phase 1: assume all goods are "personal" (tradeable)
        # For Phase 2: split between home and personal for spatial modeling
        home_endowment = total_endowment * 0.0    # Start with all goods personal for null test
        personal_endowment = total_endowment.copy()
        
        # Create identical agents for both phases
        agent1 = Agent(
            agent_id=i+1, 
            alpha=alpha, 
            home_endowment=home_endowment, 
            personal_endowment=personal_endowment
        )
        agent2 = Agent(
            agent_id=i+1, 
            alpha=alpha, 
            home_endowment=home_endowment.copy(), 
            personal_endowment=personal_endowment.copy()
        )
        
        # Phase 2 agents have positions (assume all start at marketplace for null test)
        agent2.position = (5, 5)  # Center of 10x10 grid, inside 2x2 marketplace
        
        agents_phase1.append(agent1)
        agents_phase2.append(agent2)
    
    print(f"Created {n_agents} agents with {n_goods} goods each")
    
    # Phase 1: Pure Walrasian equilibrium computation
    print("\nPhase 1: Pure Walrasian equilibrium...")
    prices_phase1, z_rest_inf_1, walras_dot_1, status_1 = solve_walrasian_equilibrium(agents_phase1)
    
    print(f"Phase 1 prices: {prices_phase1}")
    print(f"Convergence: ||Z_rest||_∞ = {z_rest_inf_1:.2e}, Walras = {walras_dot_1:.2e}")
    assert status_1 == 'converged', f"Phase 1 solver failed: {status_1}"
    assert z_rest_inf_1 < SOLVER_TOL, f"Phase 1 poor convergence: {z_rest_inf_1}"
    
    # Execute Phase 1 clearing (no spatial constraints)
    market_result_1 = execute_constrained_clearing(agents_phase1, prices_phase1)
    
    # Phase 2: Spatial simulation with κ=0 (all agents at marketplace)
    print("\nPhase 2: Spatial simulation with κ=0...")
    
    # Since all agents are at marketplace and κ=0, this should be identical to Phase 1
    # Filter agents at marketplace (all agents in this case)
    marketplace_agents = [agent for agent in agents_phase2 if agent.position == (5, 5)]
    print(f"Agents at marketplace: {len(marketplace_agents)}")
    
    prices_phase2, z_rest_inf_2, walras_dot_2, status_2 = solve_walrasian_equilibrium(marketplace_agents)
    
    print(f"Phase 2 prices: {prices_phase2}")
    print(f"Convergence: ||Z_rest||_∞ = {z_rest_inf_2:.2e}, Walras = {walras_dot_2:.2e}")
    assert status_2 == 'converged', f"Phase 2 solver failed: {status_2}"
    assert z_rest_inf_2 < SOLVER_TOL, f"Phase 2 poor convergence: {z_rest_inf_2}"
    
    # Execute Phase 2 clearing (with spatial constraints, but κ=0)
    market_result_2 = execute_constrained_clearing(marketplace_agents, prices_phase2)
    
    # Compare results: prices should be identical
    price_difference = np.linalg.norm(prices_phase1 - prices_phase2)
    print(f"Price difference: {price_difference:.2e}")
    assert price_difference < FEASIBILITY_TOL, f"Prices differ: {price_difference}"
    
    # Compare clearing efficiency
    efficiency_diff = abs(market_result_1.clearing_efficiency - market_result_2.clearing_efficiency)
    print(f"Clearing efficiency difference: {efficiency_diff:.2e}")
    assert efficiency_diff < FEASIBILITY_TOL, f"Efficiency differs: {efficiency_diff}"
    
    # Compute welfare comparison using equivalent variation
    # For this null test, we expect zero welfare loss since κ=0 and all agents at market
    total_welfare_1 = 0.0
    total_welfare_2 = 0.0
    
    for i in range(n_agents):
        agent1 = agents_phase1[i]
        agent2 = agents_phase2[i]
        
        # Compute post-trade consumption (demand at equilibrium prices)
        wealth_1 = np.dot(prices_phase1, agent1.total_endowment)
        wealth_2 = np.dot(prices_phase2, agent2.total_endowment)  # κ=0, so no travel cost
        
        consumption_1 = agent1.demand(prices_phase1, wealth_1)
        consumption_2 = agent2.demand(prices_phase2, wealth_2)
        
        utility_1 = agent1.utility(consumption_1)
        utility_2 = agent2.utility(consumption_2)
        
        total_welfare_1 += utility_1
        total_welfare_2 += utility_2
    
    welfare_difference = abs(total_welfare_1 - total_welfare_2)
    print(f"Total welfare difference: {welfare_difference:.2e}")
    
    # For V2, welfare should be identical (efficiency_loss < 1e-10)
    efficiency_loss = total_welfare_1 - total_welfare_2  # Should be ~0
    print(f"Efficiency loss: {efficiency_loss:.2e}")
    
    assert abs(efficiency_loss) < 1e-10, f"Efficiency loss too high: {efficiency_loss}"
    
    print("✅ V2 Spatial Null Test PASSED")
    print(f"   Price difference: {price_difference:.2e} < {FEASIBILITY_TOL}")
    print(f"   Efficiency difference: {efficiency_diff:.2e} < {FEASIBILITY_TOL}")
    print(f"   Welfare loss: {abs(efficiency_loss):.2e} < 1e-10")
    print("   Phase 2 with κ=0 exactly equals Phase 1 Walrasian")

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