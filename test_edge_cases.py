#!/usr/bin/env python3
"""
Comprehensive edge case testing for Agent implementation
Tests boundary conditions and robustness scenarios
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Economic constants
MIN_ALPHA = 0.05
FEASIBILITY_TOL = 1e-10

def test_edge_case_1_extreme_preferences():
    """Test agents with extreme preference distributions"""
    print("🔍 Test 1: Extreme preference distributions")
    
    # Test 1a: One very high preference, others minimal
    alpha_extreme = np.array([0.95, 0.025, 0.025])
    home_endowment = np.array([1.0, 2.0, 3.0])
    personal_endowment = np.array([0.5, 0.5, 0.5])
    agent = Agent(agent_id=1, alpha=alpha_extreme, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Verify normalization preserved
    assert abs(np.sum(agent.alpha) - 1.0) < FEASIBILITY_TOL
    
    # Verify minimum alpha constraint enforced
    assert np.all(agent.alpha >= MIN_ALPHA - FEASIBILITY_TOL)
    
    # Test utility computation
    x = np.array([1.0, 1.0, 1.0])
    utility = agent.utility(x)
    assert utility > 0, "Utility should be positive for positive consumption"
    
    print("   ✅ Extreme preferences handled correctly")

def test_edge_case_2_zero_endowments():
    """Test agents with zero or near-zero endowments"""
    print("🔍 Test 2: Zero and near-zero endowments")
    
    # Test 2a: Zero endowment in one good
    alpha = np.array([0.3, 0.3, 0.4])
    home_endowment_zero = np.array([0.0, 1.0, 2.0])
    personal_endowment_zero = np.array([0.0, 0.0, 0.0])
    agent = Agent(agent_id=2, alpha=alpha, home_endowment=home_endowment_zero, 
                  personal_endowment=personal_endowment_zero, position=(1, 1))
    
    # Verify agent created successfully
    assert np.allclose(agent.home_endowment, home_endowment_zero)
    assert np.allclose(agent.personal_endowment, personal_endowment_zero)
    
    # Test 2b: Very small endowments
    home_endowment_small = np.array([1e-12, 1e-10, 1e-8])
    personal_endowment_small = np.array([1e-14, 1e-12, 1e-10])
    agent_small = Agent(agent_id=3, alpha=alpha, home_endowment=home_endowment_small, 
                       personal_endowment=personal_endowment_small, position=(2, 2))
    
    # Should handle gracefully
    assert agent_small.total_endowment.sum() > 0
    
    print("   ✅ Zero and small endowments handled correctly")

def test_edge_case_3_price_extremes():
    """Test demand calculation with extreme prices"""
    print("🔍 Test 3: Extreme price scenarios")
    
    alpha = np.array([0.4, 0.3, 0.3])
    home_endowment = np.array([1.0, 1.0, 1.0])
    personal_endowment = np.array([0.5, 0.5, 0.5])
    agent = Agent(agent_id=4, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Test 3a: Very high prices
    prices_high = np.array([1.0, 1e6, 1e8])
    demand = agent.demand(prices_high)
    
    # Demand should be non-negative and finite
    assert np.all(demand >= 0), "Demand should be non-negative"
    assert np.all(np.isfinite(demand)), "Demand should be finite"
    
    # Budget constraint should hold
    wealth = np.dot(prices_high, agent.total_endowment)
    expenditure = np.dot(prices_high, demand)
    assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth
    
    # Test 3b: Very small prices (except numéraire)
    prices_small = np.array([1.0, 1e-8, 1e-10])
    demand_small = agent.demand(prices_small)
    
    assert np.all(demand_small >= 0)
    assert np.all(np.isfinite(demand_small))
    
    print("   ✅ Extreme prices handled correctly")

def test_edge_case_4_consumption_extremes():
    """Test utility calculation with extreme consumption bundles"""
    print("🔍 Test 4: Extreme consumption scenarios")
    
    alpha = np.array([0.5, 0.3, 0.2])
    home_endowment = np.array([1.0, 1.0, 1.0])
    personal_endowment = np.array([0.5, 0.5, 0.5])
    agent = Agent(agent_id=5, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Test 4a: Very large consumption
    x_large = np.array([1e6, 1e8, 1e10])
    utility_large = agent.utility(x_large)
    assert utility_large > 0
    assert np.isfinite(utility_large)
    
    # Test 4b: Very small consumption
    x_small = np.array([1e-10, 1e-12, 1e-8])
    utility_small = agent.utility(x_small)
    assert utility_small > 0  # Should handle log protection
    assert np.isfinite(utility_small)
    
    # Test 4c: Zero consumption (should be protected)
    x_zero = np.array([0.0, 0.0, 0.0])
    utility_zero = agent.utility(x_zero)
    assert utility_zero > 0  # Should use protection mechanism
    assert np.isfinite(utility_zero)
    
    print("   ✅ Extreme consumption bundles handled correctly")

def test_edge_case_5_inventory_transfers():
    """Test inventory transfer edge cases"""
    print("🔍 Test 5: Inventory transfer edge cases")
    
    alpha = np.array([0.4, 0.3, 0.3])
    home_endowment = np.array([5.0, 3.0, 2.0])
    personal_endowment = np.array([0.0, 0.0, 0.0])
    agent = Agent(agent_id=6, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Transfer all goods to personal
    all_goods = agent.home_endowment.copy()
    agent.transfer_goods(all_goods, to_personal=True)
    
    # Verify complete transfer
    assert np.allclose(agent.home_endowment, np.zeros(3), atol=FEASIBILITY_TOL)
    assert np.allclose(agent.personal_endowment, home_endowment, atol=FEASIBILITY_TOL)
    
    # Try to transfer more than available (should be prevented by validation)
    try:
        excess_goods = np.array([1.0, 1.0, 1.0])  # More than home has (now zero)
        agent.transfer_goods(excess_goods, to_personal=True)
        # Should raise an error due to insufficient inventory
        assert False, "Should have failed due to insufficient inventory"
    except ValueError as e:
        # Expected behavior - ValueError should be raised for insufficient inventory
        assert "Insufficient home inventory" in str(e)
        pass  # Expected behavior
    
    print("   ✅ Inventory transfer edge cases handled correctly")

def test_edge_case_6_spatial_boundaries():
    """Test spatial positioning edge cases"""
    print("🔍 Test 6: Spatial positioning boundaries")
    
    alpha = np.array([0.4, 0.3, 0.3])
    home_endowment = np.array([1.0, 1.0, 1.0])
    personal_endowment = np.array([0.5, 0.5, 0.5])
    
    # Test extreme positions
    positions_to_test = [
        (-1000, -1000),  # Very negative
        (1000, 1000),    # Very positive
        (0, 0),          # Origin
        (-1, 5),         # Mixed signs
        (100, -200)      # Large mixed
    ]
    
    for pos in positions_to_test:
        agent = Agent(agent_id=7, alpha=alpha, home_endowment=home_endowment, 
                      personal_endowment=personal_endowment, position=pos)
        assert agent.position == pos
        
        # Test marketplace detection (create reasonable bounds)
        # For a 10x10 grid with 2x2 marketplace, center would be around (4,4) to (5,5)
        marketplace_bounds = ((4, 5), (4, 5))  # ((min_x, max_x), (min_y, max_y))
        in_market = agent.is_at_marketplace(marketplace_bounds)
        assert isinstance(in_market, bool)
        
        # Test distance calculation
        marketplace_center = (4, 4)  # Center of the marketplace
        distance = agent.distance_to_marketplace(marketplace_center)
        assert distance >= 0
        assert np.isfinite(distance)
    
    print("   ✅ Spatial positioning boundaries handled correctly")

def test_edge_case_7_single_good_economy():
    """Test behavior with single good (degenerate case)"""
    print("🔍 Test 7: Single good economy (edge case)")
    
    # Single good case
    alpha_single = np.array([1.0])
    home_endowment_single = np.array([5.0])
    personal_endowment_single = np.array([2.0])
    agent = Agent(agent_id=8, alpha=alpha_single, home_endowment=home_endowment_single, 
                  personal_endowment=personal_endowment_single, position=(0, 0))
    
    # Verify initialization
    assert len(agent.alpha) == 1
    assert agent.alpha[0] == 1.0
    
    # Test utility
    x_single = np.array([3.0])
    utility = agent.utility(x_single)
    assert utility > 0
    
    # Test demand with single good
    prices_single = np.array([1.0])  # Numéraire
    demand = agent.demand(prices_single)
    
    # Should equal total endowment (no substitution possible)
    expected_demand = agent.total_endowment
    assert np.allclose(demand, expected_demand, atol=FEASIBILITY_TOL)
    
    print("   ✅ Single good economy handled correctly")

def test_edge_case_8_numerical_precision():
    """Test numerical precision and floating point edge cases"""
    print("🔍 Test 8: Numerical precision edge cases")
    
    # Use numbers that might cause floating point issues
    alpha = np.array([1/3, 1/3, 1/3])  # Should sum to 1.0 exactly after normalization
    home_endowment = np.array([0.1, 0.2, 0.3])
    personal_endowment = np.array([0.05, 0.1, 0.15])
    agent = Agent(agent_id=9, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Verify exact normalization
    assert abs(np.sum(agent.alpha) - 1.0) < 1e-15
    
    # Test with prices that sum to problematic values
    prices = np.array([1.0, np.pi, np.e])  # Irrational numbers
    demand = agent.demand(prices)
    
    # Verify budget constraint holds precisely
    wealth = np.dot(prices, agent.total_endowment)
    expenditure = np.dot(prices, demand)
    relative_error = abs(expenditure - wealth) / wealth
    assert relative_error < 1e-12, f"Budget constraint violated by {relative_error}"
    
    print("   ✅ Numerical precision edge cases handled correctly")

def run_all_edge_case_tests():
    """Run all edge case tests"""
    print("🧪 COMPREHENSIVE EDGE CASE TESTING")
    print("=" * 50)
    
    try:
        test_edge_case_1_extreme_preferences()
        test_edge_case_2_zero_endowments()
        test_edge_case_3_price_extremes()
        test_edge_case_4_consumption_extremes()
        test_edge_case_5_inventory_transfers()
        test_edge_case_6_spatial_boundaries()
        test_edge_case_7_single_good_economy()
        test_edge_case_8_numerical_precision()
        
        print("\n" + "=" * 50)
        print("🎉 ALL EDGE CASE TESTS PASSED!")
        print("Agent implementation is robust and handles boundary conditions correctly.")
        
    except Exception as e:
        print(f"\n❌ EDGE CASE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_edge_case_tests()
    sys.exit(0 if success else 1)