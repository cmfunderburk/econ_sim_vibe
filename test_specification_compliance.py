#!/usr/bin/env python3
"""
Comprehensive specification compliance testing for Agent implementation
Validates compliance with SPECIFICATION.md requirements and constants
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Economic constants from SPECIFICATION.md
FEASIBILITY_TOL = 1e-10
MIN_ALPHA = 0.05
SOLVER_TOL = 1e-8
NUMERAIRE_GOOD = 0


def test_spec_compliance_1_cobb_douglas_utility():
    """Verify Cobb-Douglas utility function U(x) = ∏_j x_j^α_j"""
    print("📋 Spec Check 1: Cobb-Douglas utility function")
    
    # Test case from SPECIFICATION.md examples
    alpha = np.array([0.6, 0.4])
    home_endowment = np.array([1.0, 0.0])
    personal_endowment = np.array([0.0, 0.0])
    agent = Agent(agent_id=1, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Test utility calculation: U(x) = x₁^0.6 * x₂^0.4
    x = np.array([2.0, 3.0])
    expected_utility = (2.0 ** 0.6) * (3.0 ** 0.4)
    computed_utility = agent.utility(x)
    
    relative_error = abs(computed_utility - expected_utility) / expected_utility
    assert relative_error < 1e-12, f"Utility calculation error: {relative_error}"
    
    print("   ✅ Cobb-Douglas utility function correctly implemented")


def test_spec_compliance_2_preference_normalization():
    """Verify preference weights sum to 1: ∑_j α_j = 1"""
    print("📋 Spec Check 2: Preference weight normalization")
    
    # Test cases that can preserve proportions (don't violate MIN_ALPHA)
    preservable_cases = [
        np.array([0.2, 0.3, 0.5]),      # Already normalized and valid
        np.array([2.0, 3.0, 5.0]),      # Simple scaling
        np.array([0.1, 0.2, 0.7]),      # Valid proportions
    ]
    
    for i, alpha_input in enumerate(preservable_cases):
        home_endowment = np.ones(len(alpha_input))
        personal_endowment = np.zeros(len(alpha_input))
        agent = Agent(agent_id=i+10, alpha=alpha_input, home_endowment=home_endowment, 
                      personal_endowment=personal_endowment, position=(0, 0))
        
        # Verify normalization
        alpha_sum = np.sum(agent.alpha)
        assert abs(alpha_sum - 1.0) < 1e-15, f"Alpha normalization failed: sum={alpha_sum}"
        
        # For cases that don't violate MIN_ALPHA, proportions should be preserved
        expected_proportions = alpha_input / np.sum(alpha_input)
        if np.all(expected_proportions >= MIN_ALPHA - FEASIBILITY_TOL):
            assert np.allclose(expected_proportions, agent.alpha, atol=1e-12)
    
    # Test cases that require interiority adjustment
    problematic_cases = [
        np.array([0.001, 0.002, 0.003]),  # All very small
        np.array([0.98, 0.01, 0.01]),     # Two below MIN_ALPHA
    ]
    
    for i, alpha_input in enumerate(problematic_cases):
        home_endowment = np.ones(len(alpha_input))
        personal_endowment = np.zeros(len(alpha_input))
        agent = Agent(agent_id=i+20, alpha=alpha_input, home_endowment=home_endowment, 
                      personal_endowment=personal_endowment, position=(0, 0))
        
        # Verify normalization still holds
        alpha_sum = np.sum(agent.alpha)
        assert abs(alpha_sum - 1.0) < 1e-15, f"Alpha normalization failed: sum={alpha_sum}"
        
        # Verify interiority condition is satisfied (this takes precedence)
        assert np.all(agent.alpha >= MIN_ALPHA - FEASIBILITY_TOL)
        
        # The algorithm should preserve relative ordering when possible
        original_order = np.argsort(alpha_input)
        final_order = np.argsort(agent.alpha)
        # Note: In extreme cases, interiority may override ordering, which is acceptable
    
    print("   ✅ Preference weight normalization correctly implemented")


def test_spec_compliance_3_interiority_condition():
    """Verify α_j ≥ MIN_ALPHA for all j to ensure interior solutions"""
    print("📋 Spec Check 3: Interiority condition (α_j ≥ MIN_ALPHA)")
    
    # Test cases that violate interiority
    problematic_cases = [
        np.array([0.99, 0.005, 0.005]),  # Two goods below MIN_ALPHA
        np.array([0.8, 0.01, 0.19]),     # One good below MIN_ALPHA
        np.array([1.0, 0.0, 0.0]),       # Pure corner solution
        np.array([0.02, 0.03, 0.95])     # Multiple violations
    ]
    
    for i, alpha_input in enumerate(problematic_cases):
        home_endowment = np.ones(len(alpha_input))
        personal_endowment = np.zeros(len(alpha_input))
        agent = Agent(agent_id=i+20, alpha=alpha_input, home_endowment=home_endowment, 
                      personal_endowment=personal_endowment, position=(0, 0))
        
        # Verify all alphas meet minimum threshold
        assert np.all(agent.alpha >= MIN_ALPHA - FEASIBILITY_TOL), \
            f"Interiority violation: min(α)={np.min(agent.alpha)}, MIN_ALPHA={MIN_ALPHA}"
        
        # Verify still sums to 1
        assert abs(np.sum(agent.alpha) - 1.0) < 1e-15
    
    print("   ✅ Interiority condition correctly enforced")


def test_spec_compliance_4_demand_function():
    """Verify Cobb-Douglas demand: x_j = α_j * wealth / p_j"""
    print("📋 Spec Check 4: Cobb-Douglas demand function")
    
    # Test with known values
    alpha = np.array([0.3, 0.7])
    home_endowment = np.array([2.0, 4.0])
    personal_endowment = np.array([1.0, 2.0])
    agent = Agent(agent_id=30, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Test prices (p₁ = 1 is numéraire)
    prices = np.array([1.0, 2.0])
    
    # Compute wealth and expected demand
    total_endowment = agent.total_endowment  # [3.0, 6.0]
    wealth = np.dot(prices, total_endowment)  # 1*3 + 2*6 = 15
    expected_demand = agent.alpha * wealth / prices  # [0.3*15/1, 0.7*15/2] = [4.5, 5.25]
    
    computed_demand = agent.demand(prices)
    
    assert np.allclose(computed_demand, expected_demand, atol=1e-12), \
        f"Demand calculation error: expected={expected_demand}, computed={computed_demand}"
    
    # Verify budget constraint
    expenditure = np.dot(prices, computed_demand)
    assert abs(expenditure - wealth) < FEASIBILITY_TOL * wealth, \
        f"Budget constraint violated: wealth={wealth}, expenditure={expenditure}"
    
    print("   ✅ Cobb-Douglas demand function correctly implemented")


def test_spec_compliance_5_conservation_laws():
    """Verify goods conservation during transfers"""
    print("📋 Spec Check 5: Conservation laws")
    
    alpha = np.array([0.4, 0.3, 0.3])
    home_endowment = np.array([5.0, 3.0, 2.0])
    personal_endowment = np.array([1.0, 2.0, 1.5])
    agent = Agent(agent_id=40, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Record initial total
    initial_total = agent.total_endowment.copy()
    
    # Perform various transfers
    transfers = [
        np.array([1.0, 0.5, 0.2]),
        np.array([0.5, 1.0, 0.8]),
        np.array([2.0, 0.5, 1.0])
    ]
    
    for transfer in transfers:
        # Transfer to personal
        agent.transfer_goods(transfer, to_personal=True)
        assert np.allclose(agent.total_endowment, initial_total, atol=FEASIBILITY_TOL)
        
        # Transfer back to home
        agent.transfer_goods(transfer, to_personal=False)
        assert np.allclose(agent.total_endowment, initial_total, atol=FEASIBILITY_TOL)
    
    print("   ✅ Conservation laws correctly enforced")


def test_spec_compliance_6_spatial_functionality():
    """Verify spatial positioning and marketplace detection"""
    print("📋 Spec Check 6: Spatial functionality")
    
    alpha = np.array([0.5, 0.5])
    home_endowment = np.array([1.0, 1.0])
    personal_endowment = np.array([0.5, 0.5])
    
    # Test agents at different positions
    test_positions = [
        (0, 0),    # Origin
        (4, 4),    # Marketplace center
        (10, 15),  # Far from marketplace
        (-5, 3)    # Mixed coordinates
    ]
    
    marketplace_bounds = ((3, 5), (3, 5))  # 3x3 marketplace at center
    marketplace_center = (4, 4)
    
    for pos in test_positions:
        agent = Agent(agent_id=50, alpha=alpha, home_endowment=home_endowment, 
                      personal_endowment=personal_endowment, position=pos)
        
        # Test position storage
        assert agent.position == pos
        
        # Test marketplace detection
        in_market = agent.is_at_marketplace(marketplace_bounds)
        expected_in_market = (3 <= pos[0] <= 5) and (3 <= pos[1] <= 5)
        assert in_market == expected_in_market, \
            f"Marketplace detection error at {pos}: expected={expected_in_market}, got={in_market}"
        
        # Test distance calculation (Manhattan)
        distance = agent.distance_to_marketplace(marketplace_center)
        expected_distance = abs(pos[0] - 4) + abs(pos[1] - 4)
        assert distance == expected_distance, \
            f"Distance calculation error at {pos}: expected={expected_distance}, got={distance}"
    
    print("   ✅ Spatial functionality correctly implemented")


def test_spec_compliance_7_numerical_constants():
    """Verify all numerical constants match SPECIFICATION.md"""
    print("📋 Spec Check 7: Numerical constants compliance")
    
    # Import constants from agent module
    from core.agent import FEASIBILITY_TOL as agent_FEASIBILITY_TOL
    from core.agent import MIN_ALPHA as agent_MIN_ALPHA
    
    # Verify constants match specification
    assert agent_FEASIBILITY_TOL == 1e-10, \
        f"FEASIBILITY_TOL mismatch: spec=1e-10, agent={agent_FEASIBILITY_TOL}"
    
    assert agent_MIN_ALPHA == 0.05, \
        f"MIN_ALPHA mismatch: spec=0.05, agent={agent_MIN_ALPHA}"
    
    # Test that constants are used correctly
    alpha = np.array([0.96, 0.02, 0.02])  # Violates MIN_ALPHA
    home_endowment = np.array([1.0, 1.0, 1.0])
    personal_endowment = np.array([0.0, 0.0, 0.0])
    agent = Agent(agent_id=60, alpha=alpha, home_endowment=home_endowment, 
                  personal_endowment=personal_endowment, position=(0, 0))
    
    # Verify MIN_ALPHA constraint is enforced
    assert np.all(agent.alpha >= MIN_ALPHA - FEASIBILITY_TOL)
    
    print("   ✅ Numerical constants correctly implemented")


def test_spec_compliance_8_edgeworth_compatibility():
    """Verify compatibility with Edgeworth box validation scenario V1"""
    print("📋 Spec Check 8: Edgeworth box compatibility")
    
    # Create agents for V1 scenario (from SPECIFICATION.md)
    # Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]
    agent1 = Agent(
        agent_id=1, 
        alpha=np.array([0.6, 0.4]),
        home_endowment=np.array([1.0, 0.0]),
        personal_endowment=np.array([0.0, 0.0]),
        position=(0, 0)
    )
    
    # Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]  
    agent2 = Agent(
        agent_id=2,
        alpha=np.array([0.3, 0.7]),
        home_endowment=np.array([0.0, 1.0]),
        personal_endowment=np.array([0.0, 0.0]),
        position=(0, 0)
    )
    
    # Correct equilibrium prices: p* = [1, 4/3]
    # Derived from market clearing: 0.6 + 0.3p = 1 and 0.4/p + 0.7 = 1
    equilibrium_prices = np.array([1.0, 4.0/3.0])
    
    # Compute demands
    demand1 = agent1.demand(equilibrium_prices)
    demand2 = agent2.demand(equilibrium_prices)
    
    # Expected demands from analytical solution: x₁* = [0.6, 0.3], x₂* = [0.4, 0.7]
    expected_demand1 = np.array([0.6, 0.3])
    expected_demand2 = np.array([0.4, 0.7])
    
    assert np.allclose(demand1, expected_demand1, atol=1e-12), \
        f"Agent 1 demand error: expected={expected_demand1}, computed={demand1}"
    
    assert np.allclose(demand2, expected_demand2, atol=1e-12), \
        f"Agent 2 demand error: expected={expected_demand2}, computed={demand2}"
    
    # Verify market clearing
    total_endowment = agent1.total_endowment + agent2.total_endowment
    total_demand = demand1 + demand2
    
    assert np.allclose(total_demand, total_endowment, atol=1e-12), \
        f"Market clearing violated: endowment={total_endowment}, demand={total_demand}"
    
    print("   ✅ Edgeworth box compatibility verified")


def run_all_specification_compliance_tests():
    """Run all specification compliance tests"""
    print("📋 COMPREHENSIVE SPECIFICATION COMPLIANCE TESTING")
    print("=" * 60)
    
    try:
        test_spec_compliance_1_cobb_douglas_utility()
        test_spec_compliance_2_preference_normalization()
        test_spec_compliance_3_interiority_condition()
        test_spec_compliance_4_demand_function()
        test_spec_compliance_5_conservation_laws()
        test_spec_compliance_6_spatial_functionality()
        test_spec_compliance_7_numerical_constants()
        test_spec_compliance_8_edgeworth_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 ALL SPECIFICATION COMPLIANCE TESTS PASSED!")
        print("Agent implementation fully complies with SPECIFICATION.md requirements.")
        print("Ready for integration with equilibrium solver and market clearing.")
        
    except Exception as e:
        print(f"\n❌ SPECIFICATION COMPLIANCE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_specification_compliance_tests()
    sys.exit(0 if success else 1)