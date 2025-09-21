#!/usr/bin/env python3
"""
Comprehensive integration testing for Agent implementation
Tests multi-agent scenarios and complex interactions
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Economic constants
FEASIBILITY_TOL = 1e-10
MIN_ALPHA = 0.05


def test_integration_1_multi_agent_edgeworth():
    """Test multi-agent Edgeworth box with proper market clearing"""
    print("🔗 Integration Test 1: Multi-agent Edgeworth box")
    
    # Create the classic 2x2 Edgeworth box agents
    agent1 = Agent(
        agent_id=1,
        alpha=np.array([0.6, 0.4]),
        home_endowment=np.array([1.0, 0.0]),
        personal_endowment=np.array([0.0, 0.0]),
        position=(0, 0)
    )
    
    agent2 = Agent(
        agent_id=2,
        alpha=np.array([0.3, 0.7]),
        home_endowment=np.array([0.0, 1.0]),
        personal_endowment=np.array([0.0, 0.0]),
        position=(0, 0)
    )
    
    agents = [agent1, agent2]
    equilibrium_prices = np.array([1.0, 4.0/3.0])
    
    # Test all agents with equilibrium prices
    demands = []
    total_demand = np.zeros(2)
    total_endowment = np.zeros(2)
    
    for agent in agents:
        demand = agent.demand(equilibrium_prices)
        demands.append(demand)
        total_demand += demand
        total_endowment += agent.total_endowment
        
        # Verify individual rationality
        assert np.all(demand >= 0), f"Negative demand for agent {agent.agent_id}"
        
        # Verify budget constraint
        wealth = np.dot(equilibrium_prices, agent.total_endowment)
        expenditure = np.dot(equilibrium_prices, demand)
        assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth
    
    # Verify market clearing
    assert np.allclose(total_demand, total_endowment, atol=FEASIBILITY_TOL)
    
    print("   ✅ Multi-agent Edgeworth box integration successful")


def test_integration_2_three_agent_economy():
    """Test three-agent, three-good economy"""
    print("🔗 Integration Test 2: Three-agent, three-good economy")
    
    # Create diverse agents
    agents = [
        Agent(
            agent_id=1,
            alpha=np.array([0.5, 0.3, 0.2]),
            home_endowment=np.array([2.0, 0.0, 1.0]),
            personal_endowment=np.array([0.0, 0.0, 0.0]),
            position=(0, 0)
        ),
        Agent(
            agent_id=2,
            alpha=np.array([0.2, 0.6, 0.2]),
            home_endowment=np.array([0.0, 3.0, 0.0]),
            personal_endowment=np.array([0.5, 0.0, 0.5]),
            position=(1, 1)
        ),
        Agent(
            agent_id=3,
            alpha=np.array([0.3, 0.2, 0.5]),
            home_endowment=np.array([1.0, 0.5, 2.0]),
            personal_endowment=np.array([0.0, 1.0, 0.0]),
            position=(2, 2)
        )
    ]
    
    # Test with arbitrary prices (not necessarily equilibrium)
    test_prices = np.array([1.0, 1.5, 2.0])
    
    total_demand = np.zeros(3)
    total_endowment = np.zeros(3)
    total_wealth = 0.0
    total_expenditure = 0.0
    
    for agent in agents:
        demand = agent.demand(test_prices)
        wealth = np.dot(test_prices, agent.total_endowment)
        expenditure = np.dot(test_prices, demand)
        
        # Individual agent checks
        assert np.all(demand >= 0), f"Negative demand for agent {agent.agent_id}"
        assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth
        
        # Accumulate for aggregate checks
        total_demand += demand
        total_endowment += agent.total_endowment
        total_wealth += wealth
        total_expenditure += expenditure
    
    # Aggregate budget constraint (Walras' Law)
    assert abs(total_expenditure - total_wealth) < FEASIBILITY_TOL * total_wealth
    
    # Test utilities are positive and finite
    for agent in agents:
        demand = agent.demand(test_prices)
        utility = agent.utility(demand)
        assert utility > 0 and np.isfinite(utility)
    
    print("   ✅ Three-agent, three-good economy integration successful")


def test_integration_3_inventory_management():
    """Test complex inventory transfers across multiple agents"""
    print("🔗 Integration Test 3: Multi-agent inventory management")
    
    # Create agents with different inventory distributions
    agents = [
        Agent(
            agent_id=1,
            alpha=np.array([0.4, 0.3, 0.3]),
            home_endowment=np.array([5.0, 2.0, 1.0]),
            personal_endowment=np.array([1.0, 3.0, 2.0]),
            position=(0, 0)
        ),
        Agent(
            agent_id=2,
            alpha=np.array([0.3, 0.4, 0.3]),
            home_endowment=np.array([2.0, 4.0, 3.0]),
            personal_endowment=np.array([2.0, 1.0, 1.0]),
            position=(1, 1)
        )
    ]
    
    # Record initial state
    initial_totals = [agent.total_endowment.copy() for agent in agents]
    grand_total = sum(initial_totals)
    
    # Perform complex transfers
    transfers = [
        (agents[0], np.array([1.0, 0.5, 0.2]), True),   # Agent 0: home → personal
        (agents[1], np.array([0.5, 1.0, 0.8]), False),  # Agent 1: personal → home
        (agents[0], np.array([2.0, 1.0, 0.3]), True),   # Agent 0: more home → personal
        (agents[1], np.array([1.0, 0.5, 0.5]), True),   # Agent 1: home → personal
    ]
    
    for agent, transfer, to_personal in transfers:
        # Verify transfer is valid before executing
        if to_personal:
            assert np.all(agent.home_endowment >= transfer - FEASIBILITY_TOL)
        else:
            assert np.all(agent.personal_endowment >= transfer - FEASIBILITY_TOL)
        
        # Execute transfer
        agent.transfer_goods(transfer, to_personal)
        
        # Verify conservation for this agent
        current_total = agent.total_endowment
        expected_total = initial_totals[agent.agent_id - 1]
        assert np.allclose(current_total, expected_total, atol=FEASIBILITY_TOL)
    
    # Verify global conservation
    final_grand_total = sum(agent.total_endowment for agent in agents)
    assert np.allclose(final_grand_total, grand_total, atol=FEASIBILITY_TOL)
    
    print("   ✅ Multi-agent inventory management integration successful")


def test_integration_4_spatial_marketplace():
    """Test spatial positioning and marketplace interactions"""
    print("🔗 Integration Test 4: Spatial marketplace interactions")
    
    # Create agents at various positions
    marketplace_bounds = ((2, 4), (2, 4))  # 3x3 marketplace
    marketplace_center = (3, 3)
    
    agents = [
        Agent(  # At marketplace
            agent_id=1,
            alpha=np.array([0.5, 0.5]),
            home_endowment=np.array([1.0, 1.0]),
            personal_endowment=np.array([0.5, 0.5]),
            position=(3, 3)
        ),
        Agent(  # Near marketplace
            agent_id=2,
            alpha=np.array([0.4, 0.6]),
            home_endowment=np.array([2.0, 0.5]),
            personal_endowment=np.array([0.0, 1.0]),
            position=(1, 3)
        ),
        Agent(  # Far from marketplace
            agent_id=3,
            alpha=np.array([0.6, 0.4]),
            home_endowment=np.array([0.5, 2.0]),
            personal_endowment=np.array([1.0, 0.0]),
            position=(10, 10)
        ),
        Agent(  # Edge of marketplace
            agent_id=4,
            alpha=np.array([0.3, 0.7]),
            home_endowment=np.array([1.5, 1.5]),
            personal_endowment=np.array([0.2, 0.8]),
            position=(4, 2)
        )
    ]
    
    # Test marketplace detection
    expected_in_marketplace = [True, False, False, True]
    expected_distances = [0, 2, 14, 2]  # Fixed: (4,2) to (3,3) = |4-3| + |2-3| = 1 + 1 = 2
    
    for i, agent in enumerate(agents):
        in_market = agent.is_at_marketplace(marketplace_bounds)
        distance = agent.distance_to_marketplace(marketplace_center)
        
        assert in_market == expected_in_marketplace[i], \
            f"Agent {agent.agent_id} marketplace detection failed"
        assert distance == expected_distances[i], \
            f"Agent {agent.agent_id} distance calculation failed"
    
    # Test that spatial position doesn't affect economic calculations
    test_prices = np.array([1.0, 1.2])
    
    for agent in agents:
        demand = agent.demand(test_prices)
        utility = agent.utility(demand)
        
        # Economic functions should work regardless of position
        assert np.all(demand >= 0)
        assert utility > 0 and np.isfinite(utility)
        
        # Budget constraint should hold
        wealth = np.dot(test_prices, agent.total_endowment)
        expenditure = np.dot(test_prices, demand)
        assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth
    
    print("   ✅ Spatial marketplace interactions integration successful")


def test_integration_5_large_scale_economy():
    """Test economy with many agents and goods"""
    print("🔗 Integration Test 5: Large-scale economy")
    
    n_agents = 20
    n_goods = 5
    np.random.seed(42)  # Deterministic for testing
    
    agents = []
    for i in range(n_agents):
        # Generate random but valid preferences
        alpha_raw = np.random.exponential(scale=1.0, size=n_goods)
        alpha_raw = np.maximum(alpha_raw, MIN_ALPHA * 1.1)  # Ensure above minimum
        
        # Generate random endowments
        home_endowment = np.random.exponential(scale=2.0, size=n_goods)
        personal_endowment = np.random.exponential(scale=0.5, size=n_goods)
        
        # Random position
        position = (np.random.randint(-10, 11), np.random.randint(-10, 11))
        
        agent = Agent(
            agent_id=i,
            alpha=alpha_raw,
            home_endowment=home_endowment,
            personal_endowment=personal_endowment,
            position=position
        )
        agents.append(agent)
    
    # Test with random prices
    test_prices = np.concatenate([[1.0], np.random.exponential(scale=1.5, size=n_goods-1)])
    
    # Compute aggregate statistics
    total_demand = np.zeros(n_goods)
    total_endowment = np.zeros(n_goods)
    total_wealth = 0.0
    total_expenditure = 0.0
    utilities = []
    
    for agent in agents:
        demand = agent.demand(test_prices)
        wealth = np.dot(test_prices, agent.total_endowment)
        expenditure = np.dot(test_prices, demand)
        utility = agent.utility(demand)
        
        # Individual checks
        assert np.all(demand >= 0), f"Negative demand for agent {agent.agent_id}"
        assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth
        assert utility > 0 and np.isfinite(utility)
        assert np.all(agent.alpha >= MIN_ALPHA - FEASIBILITY_TOL)
        assert abs(np.sum(agent.alpha) - 1.0) < 1e-15
        
        # Accumulate
        total_demand += demand
        total_endowment += agent.total_endowment
        total_wealth += wealth
        total_expenditure += expenditure
        utilities.append(utility)
    
    # Aggregate checks
    assert abs(total_expenditure - total_wealth) < FEASIBILITY_TOL * total_wealth
    
    # Statistical checks
    assert len(utilities) == n_agents
    assert all(u > 0 for u in utilities)
    assert np.all(np.isfinite(utilities))
    
    print(f"   ✅ Large-scale economy ({n_agents} agents, {n_goods} goods) integration successful")


def test_integration_6_extreme_scenarios():
    """Test agents in extreme but valid scenarios"""
    print("🔗 Integration Test 6: Extreme scenarios")
    
    # Scenario 1: Agent with very unequal preferences
    extreme_agent = Agent(
        agent_id=1,
        alpha=np.array([0.9, 0.05, 0.05]),  # Will be adjusted for interiority
        home_endowment=np.array([0.1, 10.0, 5.0]),
        personal_endowment=np.array([5.0, 0.1, 0.1]),
        position=(0, 0)
    )
    
    # Scenario 2: Agent with very small endowments
    tiny_agent = Agent(
        agent_id=2,
        alpha=np.array([0.33, 0.33, 0.34]),
        home_endowment=np.array([1e-6, 1e-5, 1e-4]),
        personal_endowment=np.array([1e-7, 1e-6, 1e-5]),
        position=(100, -100)
    )
    
    # Scenario 3: Agent with large endowments
    wealthy_agent = Agent(
        agent_id=3,
        alpha=np.array([0.2, 0.3, 0.5]),
        home_endowment=np.array([1e6, 1e7, 1e8]),
        personal_endowment=np.array([1e5, 1e6, 1e7]),
        position=(-1000, 1000)
    )
    
    agents = [extreme_agent, tiny_agent, wealthy_agent]
    
    # Test with various price scenarios
    price_scenarios = [
        np.array([1.0, 1.0, 1.0]),           # Equal prices
        np.array([1.0, 1e-6, 1e6]),          # Extreme price ratios
        np.array([1.0, 100.0, 0.01]),        # Mixed extreme prices
    ]
    
    for i, prices in enumerate(price_scenarios):
        print(f"     Testing price scenario {i+1}")
        
        for agent in agents:
            demand = agent.demand(prices)
            wealth = np.dot(prices, agent.total_endowment)
            expenditure = np.dot(prices, demand)
            utility = agent.utility(demand)
            
            # All should work without numerical issues
            assert np.all(demand >= 0), f"Negative demand in scenario {i+1}"
            assert np.all(np.isfinite(demand)), f"Infinite demand in scenario {i+1}"
            assert utility > 0 and np.isfinite(utility), f"Invalid utility in scenario {i+1}"
            
            # Budget constraint (with relative tolerance for extreme values)
            rel_error = abs(expenditure - wealth) / max(wealth, 1e-10)
            assert rel_error < 1e-10, f"Budget violation in scenario {i+1}: {rel_error}"
    
    print("   ✅ Extreme scenarios integration successful")


def run_all_integration_tests():
    """Run all integration tests"""
    print("🔗 COMPREHENSIVE INTEGRATION TESTING")
    print("=" * 50)
    
    try:
        test_integration_1_multi_agent_edgeworth()
        test_integration_2_three_agent_economy()
        test_integration_3_inventory_management()
        test_integration_4_spatial_marketplace()
        test_integration_5_large_scale_economy()
        test_integration_6_extreme_scenarios()
        
        print("\n" + "=" * 50)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("Agent implementation handles complex multi-agent scenarios correctly.")
        print("Ready for production use in equilibrium solver and market simulation.")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)