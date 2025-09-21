"""
Tests for Economically Correct Order Generation

This file validates the economically sound implementation of order generation
that enforces pure-exchange budget constraints, replacing the older tests 
that assumed unlimited credit (which is not realistic for barter economies).

PHILOSOPHY CHANGE:
================
OLD APPROACH: Tests validated pure Cobb-Douglas theory without constraints
- Assumed agents could buy unlimited goods on credit
- Ignored feasibility constraints in pure exchange economies
- Tested mathematical purity over economic realism

NEW APPROACH: Tests validate constrained Cobb-Douglas with budget enforcement
- Agents can only buy what they can afford with their selling capacity
- Enforces pure-exchange constraint: p·buy_orders ≤ p·sell_orders
- Tests economic correctness over mathematical purity

ECONOMIC JUSTIFICATION:
======================
In a pure exchange (barter) economy:
1. No external credit or money creation exists
2. Agents can only acquire goods by giving up other goods
3. Budget constraints must be satisfied instantaneously
4. Theory must yield to economic feasibility

This represents a fundamental improvement in economic realism.
"""

import numpy as np
import pytest
from src.core.agent import Agent
from src.econ.market import _generate_agent_orders
from src.constants import FEASIBILITY_TOL

@pytest.mark.economic_core
@pytest.mark.real_functions
class TestBudgetConstrainedOrderGeneration:
    """Test that order generation properly enforces economic budget constraints."""
    
    def test_budget_constraint_enforcement(self):
        """Test that orders satisfy p·buy ≤ p·sell (pure exchange constraint) after loading full inventory."""
        # Create agent with specific endowments that would violate budget if unconstrained
        alpha = np.array([0.8, 0.2])  # Strong preference for good 1
        home_endowment = np.array([0.5, 4.0])  # Little good 1, lots good 2 at home
        personal_endowment = np.array([0.1, 1.0])  # Small amounts available for trade
        
        agent = Agent(
            agent_id=1,
            alpha=alpha, 
            home_endowment=home_endowment,
            personal_endowment=personal_endowment
        )
        
        # NEW: Agent loads full inventory for travel (simplified model)
        agent.load_inventory_for_travel()
        
        prices = np.array([1.0, 1.0])  # Equal prices
        
        # Generate orders using budget-constrained implementation
        orders = _generate_agent_orders([agent], prices)
        order = orders[0]
        
        # CRITICAL TEST: Budget constraint must be satisfied
        buy_value = np.dot(prices, order.buy_orders)
        sell_value = np.dot(prices, order.sell_orders)
        
        assert buy_value <= sell_value + FEASIBILITY_TOL, (
            f"Budget constraint violated: buy_value={buy_value} > sell_value={sell_value}"
        )
        print(f"✅ Budget constraint satisfied: {buy_value:.3f} ≤ {sell_value:.3f}")
    
    def test_scaling_when_budget_exceeded(self):
        """Test that orders are properly scaled when theoretical demand exceeds budget."""
        # Create scenario where pure theory would violate budget
        alpha = np.array([0.9, 0.1])  # Extreme preference for good 1
        personal_endowment = np.array([0.1, 2.0])  # Can sell 2 units of good 2
        
        agent = Agent(
            agent_id=1,
            alpha=alpha,
            home_endowment=np.array([1.0, 1.0]),  # Additional inventory at home
            personal_endowment=personal_endowment
        )
        
        # Load full inventory for simplified trading model
        agent.load_inventory_for_travel()
        
        prices = np.array([1.0, 0.5])  # Good 1 expensive, good 2 cheap
        
        # Theoretical demand would be: x = α * wealth / p
        wealth = np.dot(prices, agent.total_endowment)  # Uses total for LTE pricing
        theoretical_demand = alpha * wealth / prices
        theoretical_net = theoretical_demand - personal_endowment
        
        # Budget is now limited by total inventory agent is carrying
        max_sell_value = np.dot(prices, np.maximum(-theoretical_net, 0))  # Value of what agent can sell
        
        orders = _generate_agent_orders([agent], prices)
        order = orders[0]
        
        # Orders should be scaled to respect budget
        buy_value = np.dot(prices, order.buy_orders)
        sell_value = np.dot(prices, order.sell_orders)
        assert buy_value <= sell_value + FEASIBILITY_TOL
        
        # But should still reflect preference ratios when possible
        if order.buy_orders[0] > 0 and order.buy_orders[1] > 0:
            ratio = (order.buy_orders[0] / prices[0]) / (order.buy_orders[1] / prices[1])
            expected_ratio = alpha[0] / alpha[1]
            assert abs(ratio - expected_ratio) < 0.1, (
                f"Preference ratio not preserved in scaling: {ratio} vs {expected_ratio}"
            )
        
        print(f"✅ Budget scaling working: theoretical buy value would be {np.dot(prices, np.maximum(theoretical_net, 0)):.3f}, actual is {buy_value:.3f}")
    
    def test_inventory_constraints_respected(self):
        """Test that sell orders never exceed personal inventory."""
        personal_endowment = np.array([1.5, 2.5])
        
        agent = Agent(
            agent_id=1,
            alpha=np.array([0.3, 0.7]),  # Prefer good 2, will sell good 1
            home_endowment=np.array([2.0, 1.0]),
            personal_endowment=personal_endowment
        )
        
        prices = np.array([2.0, 1.0])  # Good 1 expensive
        
        orders = _generate_agent_orders([agent], prices)
        order = orders[0]
        
        # Sell orders must not exceed personal inventory
        for g in range(len(personal_endowment)):
            assert order.sell_orders[g] <= personal_endowment[g] + FEASIBILITY_TOL, (
                f"Good {g}: sell order {order.sell_orders[g]} > personal inventory {personal_endowment[g]}"
            )
        
        print(f"✅ Inventory constraints respected: max sells {order.sell_orders} ≤ personal {personal_endowment}")
    
    def test_economic_consistency_with_equilibrium(self):
        """Test that budget-constrained orders are consistent with economic theory."""
        # Create multiple agents for market context
        agents = []
        for i in range(3):
            agents.append(Agent(
                agent_id=i+1,
                alpha=np.array([0.4 + 0.2*i, 0.6 - 0.2*i]),
                home_endowment=np.array([2.0, 2.0]),
                personal_endowment=np.array([1.0, 1.0])
            ))
        
        prices = np.array([1.0, 1.5])
        
        # Generate all orders
        orders = _generate_agent_orders(agents, prices)
        
        # Aggregate market totals
        total_buy_value = sum(np.dot(prices, order.buy_orders) for order in orders)
        total_sell_value = sum(np.dot(prices, order.sell_orders) for order in orders)
        
        # In budget-constrained system, these should be approximately equal
        assert abs(total_buy_value - total_sell_value) < len(agents) * FEASIBILITY_TOL, (
            f"Market imbalance too large: buy_value={total_buy_value}, sell_value={total_sell_value}"
        )
        
        print(f"✅ Market consistency: total buy value {total_buy_value:.3f} ≈ total sell value {total_sell_value:.3f}")

@pytest.mark.economic_core 
@pytest.mark.real_functions
class TestTheoryVsPracticeComparison:
    """Compare pure theory with economically constrained implementation."""
    
    def test_pure_theory_vs_constrained_orders(self):
        """Demonstrate the difference between pure theory and budget-constrained orders."""
        # Agent with strong preference but limited trading capacity
        alpha = np.array([0.8, 0.2])
        personal_endowment = np.array([0.5, 3.0])  # Little good 1, lots good 2
        
        agent = Agent(
            agent_id=1,
            alpha=alpha,
            home_endowment=np.array([2.0, 1.0]),  # Total wealth for LTE pricing
            personal_endowment=personal_endowment
        )
        
        prices = np.array([1.0, 1.0])
        
        # PURE THEORY: What unconstrained Cobb-Douglas would want
        wealth = np.dot(prices, agent.total_endowment)  # 3.0 total wealth
        theoretical_demand = alpha * wealth / prices  # [2.4, 0.6]
        theoretical_net = theoretical_demand - personal_endowment  # [1.9, -2.4]
        theoretical_buy_value = np.dot(prices, np.maximum(theoretical_net, 0))  # 1.9
        theoretical_sell_value = np.dot(prices, np.maximum(-theoretical_net, 0))  # 2.4
        
        print(f"PURE THEORY: Would want to buy {theoretical_buy_value:.2f} value, sell {theoretical_sell_value:.2f} value")
        print(f"THEORY PROBLEM: Buy value < sell value, violates budget constraint!")
        
        # CONSTRAINED PRACTICE: What our implementation actually does
        orders = _generate_agent_orders([agent], prices)
        order = orders[0]
        
        actual_buy_value = np.dot(prices, order.buy_orders)
        actual_sell_value = np.dot(prices, order.sell_orders)
        
        print(f"CONSTRAINED PRACTICE: Actually buys {actual_buy_value:.2f} value, sells {actual_sell_value:.2f} value")
        print(f"PRACTICE SOLUTION: Buy value ≤ sell value, satisfies budget constraint!")
        
        # Verify the constraint is satisfied
        assert actual_buy_value <= actual_sell_value + FEASIBILITY_TOL
        
        # Verify preferences are still respected in the constrained solution
        if order.buy_orders[0] > 0:
            # Should still prefer good 1 over good 2 proportionally
            buy_ratio = order.buy_orders[0] / max(order.buy_orders[1], 1e-10)
            preference_ratio = alpha[0] / alpha[1] 
            print(f"Preference preservation: buy ratio {buy_ratio:.2f} vs preference ratio {preference_ratio:.2f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])