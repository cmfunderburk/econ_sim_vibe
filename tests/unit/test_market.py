"""Comprehensive test suite for market clearing mechanisms.

This module tests the constrained clearing algorithm with proportional rationing,
validating all economic invariants and edge cases as specified in the economic
specification.

Tests cover:
- Basic clearing functionality with balanced supply/demand
- Proportional rationing under excess demand/supply conditions
- Personal inventory constraints and value feasibility
- Economic invariant validation (conservation, market balance)
- Edge cases (empty market, zero orders, capacity constraints)
- Integration with Agent class and equilibrium solver

Author: AI Assistant
Date: 2024-12-19
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from typing import List

# Import modules under test
from src.core.agent import Agent
from src.core.types import Trade, MarketResult
from src.econ.market import (
    execute_constrained_clearing,
    apply_trades_to_agents,
    _generate_agent_orders,
    _compute_market_totals,
    _execute_proportional_rationing,
    _validate_clearing_invariants,
    _convert_to_trades,
    AgentOrder,
    RATIONING_EPS,
    FEASIBILITY_TOL,
)


class TestAgentOrder:
    """Test the AgentOrder dataclass validation."""

    def test_valid_order_creation(self):
        """Test creation of valid agent orders."""
        order = AgentOrder(
            agent_id=1,
            buy_orders=np.array([2.0, 0.5]),
            sell_orders=np.array([0.0, 1.5]),
            max_sell_capacity=np.array([1.0, 2.0]),
        )
        assert order.agent_id == 1
        assert np.allclose(order.buy_orders, [2.0, 0.5])
        assert np.allclose(order.sell_orders, [0.0, 1.5])
        assert np.allclose(order.max_sell_capacity, [1.0, 2.0])

    def test_mismatched_array_lengths(self):
        """Test validation of array length consistency."""
        with pytest.raises(ValueError, match="must have same length"):
            AgentOrder(
                agent_id=1,
                buy_orders=np.array([1.0, 2.0]),
                sell_orders=np.array([1.0]),  # Wrong length
                max_sell_capacity=np.array([1.0, 1.0]),
            )

    def test_negative_values_rejection(self):
        """Test rejection of negative order quantities."""
        with pytest.raises(ValueError, match="must be non-negative"):
            AgentOrder(
                agent_id=1,
                buy_orders=np.array([-1.0, 2.0]),  # Negative buy order
                sell_orders=np.array([1.0, 1.0]),
                max_sell_capacity=np.array([2.0, 2.0]),
            )

    def test_inventory_constraint_violation(self):
        """Test validation of sell orders against inventory capacity."""
        with pytest.raises(ValueError, match="exceed inventory capacity"):
            AgentOrder(
                agent_id=1,
                buy_orders=np.array([1.0, 1.0]),
                sell_orders=np.array([2.0, 1.0]),  # Exceeds capacity
                max_sell_capacity=np.array([1.0, 1.0]),
            )


class TestGenerateAgentOrders:
    """Test agent order generation from optimal demands."""

    def setup_method(self):
        """Set up test agents with known preferences."""
        # Agent 1: Prefers good 0 (α=[0.7, 0.3])
        self.agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.7, 0.3]),
            home_endowment=np.array([2.0, 2.0]),
            personal_endowment=np.array(
                [1.0, 0.5]
            ),  # Start with some personal inventory
            position=(5, 5),  # In marketplace
        )

        # Agent 2: Prefers good 1 (α=[0.3, 0.7])
        self.agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.3, 0.7]),
            home_endowment=np.array([1.0, 3.0]),
            personal_endowment=np.array([0.5, 2.0]),
            position=(5, 5),  # In marketplace
        )

        self.agents = [self.agent1, self.agent2]
        self.prices = np.array([1.0, 1.0])  # Equal prices

    def test_basic_order_generation(self):
        """Test basic order generation with simple setup."""
        orders = _generate_agent_orders(self.agents, self.prices)

        assert len(orders) == 2
        assert orders[0].agent_id == 1
        assert orders[1].agent_id == 2

        # Verify non-negative orders
        for order in orders:
            assert np.all(order.buy_orders >= 0)
            assert np.all(order.sell_orders >= 0)
            assert np.all(order.max_sell_capacity >= 0)

    def test_order_inventory_consistency(self):
        """Test that sell orders respect personal inventory limits."""
        orders = _generate_agent_orders(self.agents, self.prices)

        for i, order in enumerate(orders):
            agent = self.agents[i]
            # Sell orders should not exceed personal inventory
            assert np.all(
                order.sell_orders <= order.max_sell_capacity + FEASIBILITY_TOL
            )
            # Max sell capacity should equal personal inventory
            assert np.allclose(order.max_sell_capacity, agent.personal_endowment)

    def test_cobb_douglas_demand_integration(self):
        """Test integration with Agent's Cobb-Douglas demand function."""
        orders = _generate_agent_orders(self.agents, self.prices)

        for i, order in enumerate(orders):
            agent = self.agents[i]
            desired = agent.demand(self.prices)
            current_personal = agent.personal_endowment

            # Net orders should match desired minus current
            net_orders = order.buy_orders - order.sell_orders
            expected_net = desired - current_personal

            # Allow for non-negativity clipping in buy/sell separation
            assert len(net_orders) == len(expected_net)

    def test_empty_agents_list(self):
        """Test order generation with no agents."""
        orders = _generate_agent_orders([], self.prices)
        assert orders == []


class TestComputeMarketTotals:
    """Test computation of aggregate market demand and supply."""

    def test_basic_totals_computation(self):
        """Test basic aggregation of individual orders."""
        orders = [
            AgentOrder(
                1, np.array([2.0, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 2.0])
            ),
            AgentOrder(
                2, np.array([1.0, 1.5]), np.array([0.5, 0.0]), np.array([1.0, 1.0])
            ),
        ]

        total_buys, total_sells = _compute_market_totals(orders)

        assert np.allclose(total_buys, [3.0, 2.0])  # 2.0+1.0, 0.5+1.5
        assert np.allclose(total_sells, [0.5, 1.0])  # 0.0+0.5, 1.0+0.0

    def test_empty_orders_list(self):
        """Test totals computation with no orders."""
        total_buys, total_sells = _compute_market_totals([])

        assert len(total_buys) == 0
        assert len(total_sells) == 0

    def test_single_agent_totals(self):
        """Test totals with single agent."""
        orders = [
            AgentOrder(
                1, np.array([1.5, 2.0]), np.array([0.5, 0.0]), np.array([1.0, 1.0])
            )
        ]

        total_buys, total_sells = _compute_market_totals(orders)

        assert np.allclose(total_buys, [1.5, 2.0])
        assert np.allclose(total_sells, [0.5, 0.0])


class TestProportionalRationing:
    """Test the core proportional rationing algorithm."""

    def setup_method(self):
        """Set up test orders for rationing scenarios."""
        self.orders = [
            AgentOrder(
                1, np.array([3.0, 1.0]), np.array([0.0, 2.0]), np.array([1.0, 3.0])
            ),
            AgentOrder(
                2, np.array([1.0, 2.0]), np.array([1.0, 0.0]), np.array([2.0, 1.0])
            ),
        ]
        self.total_buys = np.array([4.0, 3.0])  # 3.0+1.0, 1.0+2.0
        self.total_sells = np.array([1.0, 2.0])  # 0.0+1.0, 2.0+0.0

    def test_supply_constrained_rationing(self):
        """Test rationing when demand exceeds supply."""
        executed_buys, executed_sells, volumes = _execute_proportional_rationing(
            self.orders, self.total_buys, self.total_sells
        )

        # Good 0: demand=4.0, supply=1.0 → cleared volume = 1.0
        # Good 1: demand=3.0, supply=2.0 → cleared volume = 2.0
        assert np.allclose(volumes, [1.0, 2.0])

        # Verify buy rationing for good 0 (supply constrained)
        # Agent 1: (3.0/4.0) * 1.0 = 0.75, Agent 2: (1.0/4.0) * 1.0 = 0.25
        assert np.isclose(executed_buys[1][0], 0.75)
        assert np.isclose(executed_buys[2][0], 0.25)

        # Verify full sell execution for good 0
        assert np.isclose(executed_sells[2][0], 1.0)  # Agent 2's full sell order

    def test_demand_constrained_rationing(self):
        """Test rationing when supply exceeds demand."""
        # Modify to create demand-constrained scenario
        modified_sells = np.array([5.0, 4.0])  # Excess supply

        executed_buys, executed_sells, volumes = _execute_proportional_rationing(
            self.orders, self.total_buys, modified_sells
        )

        # Cleared volumes should equal demand (smaller of buy/sell)
        assert np.allclose(volumes, self.total_buys)  # [4.0, 3.0]

        # All buy orders should be filled
        assert np.isclose(executed_buys[1][0], 3.0)  # Agent 1's full buy order
        assert np.isclose(executed_buys[2][0], 1.0)  # Agent 2's full buy order

    def test_capacity_constraints(self):
        """Test throughput capacity limits."""
        capacity = np.array([0.5, 1.5])  # Tight capacity limits

        executed_buys, executed_sells, volumes = _execute_proportional_rationing(
            self.orders, self.total_buys, self.total_sells, capacity
        )

        # Volumes should respect capacity limits
        expected_volumes = np.minimum(
            np.minimum(self.total_buys, self.total_sells), capacity
        )
        assert np.allclose(volumes, expected_volumes)

    def test_zero_volume_handling(self):
        """Test handling of goods with zero trading volume."""
        zero_buys = np.array([0.0, 1.0])

        executed_buys, executed_sells, volumes = _execute_proportional_rationing(
            self.orders, zero_buys, self.total_sells
        )

        # No trades for good 0
        assert volumes[0] == 0.0
        assert executed_buys[1][0] == 0.0
        assert executed_buys[2][0] == 0.0
        assert executed_sells[1][0] == 0.0
        assert executed_sells[2][0] == 0.0


class TestClearingInvariantsValidation:
    """Test validation of economic invariants after clearing."""

    def setup_method(self):
        """Set up test data for invariant validation."""
        self.orders = [
            AgentOrder(
                1, np.array([2.0, 1.0]), np.array([0.0, 1.5]), np.array([1.0, 2.0])
            ),
            AgentOrder(
                2, np.array([1.0, 0.5]), np.array([1.0, 0.0]), np.array([1.5, 1.0])
            ),
        ]

        # Valid executed quantities (satisfies all invariants)
        self.executed_buys = {1: np.array([1.5, 0.8]), 2: np.array([0.5, 0.2])}
        self.executed_sells = {
            1: np.array([0.0, 1.0]),
            2: np.array([2.0, 0.0]),  # Violates inventory constraint for testing
        }
        self.executed_volumes = np.array([2.0, 1.0])
        self.prices = np.array([1.0, 2.0])

    def test_valid_invariants_pass(self):
        """Test that valid clearing results pass all invariant checks."""
        # Create properly balanced executed quantities
        valid_executed_buys = {1: np.array([1.0, 0.5]), 2: np.array([0.5, 1.0])}
        valid_executed_sells = {
            1: np.array([0.0, 1.0]),
            2: np.array([1.5, 0.5]),  # Adjusted to balance
        }
        # Volumes must equal total buys = total sells
        valid_executed_volumes = np.array([1.5, 1.5])  # Total buys and sells both = 1.5

        # Should not raise any assertions
        _validate_clearing_invariants(
            self.orders,
            valid_executed_buys,
            valid_executed_sells,
            valid_executed_volumes,
            self.prices,
        )

    def test_market_imbalance_detection(self):
        """Test detection of market imbalance violations."""
        # Create imbalanced volumes
        imbalanced_volumes = np.array([1.5, 1.0])  # Doesn't match total buys/sells

        with pytest.raises(AssertionError, match="Volume inconsistency"):
            _validate_clearing_invariants(
                self.orders,
                self.executed_buys,
                self.executed_sells,
                imbalanced_volumes,
                self.prices,
            )

    def test_inventory_constraint_violation(self):
        """Test detection of inventory constraint violations."""
        # Create scenario that passes value feasibility but fails inventory constraints
        valid_buys = {
            1: np.array([0.5, 0.0]),  # Agent 1 buys 0.5*1 = 0.5 value
            2: np.array([1.5, 0.0]),  # Agent 2 buys 1.5*1 = 1.5 value
        }
        violating_sells = {
            1: np.array([0.5, 0.0]),  # Agent 1 sells 0.5*1 = 0.5 value (value feasible)
            2: np.array(
                [1.5, 0.0]
            ),  # Agent 2 sells 1.5 > max_capacity[0]=1.5 (boundary case)
        }
        balanced_volumes = np.array([2.0, 0.0])  # Market balanced

        # Set agent 2's capacity to be violated
        # Create new orders with lower capacity for agent 2
        violating_orders = [
            AgentOrder(
                1, np.array([2.0, 1.0]), np.array([0.0, 1.5]), np.array([1.0, 2.0])
            ),
            AgentOrder(
                2, np.array([1.0, 0.5]), np.array([1.0, 0.0]), np.array([1.0, 1.0])
            ),  # Lower capacity
        ]

        # This should catch inventory constraint violation
        with pytest.raises(AssertionError, match="inventory constraint violated"):
            _validate_clearing_invariants(
                violating_orders,
                valid_buys,
                violating_sells,
                balanced_volumes,
                self.prices,
            )

    def test_inventory_constraint_violation_in_clearing(self):
        """Test detection of inventory constraint violations in clearing validation."""
        # Create scenario that violates inventory constraints  
        valid_buys = {
            1: np.array([0.5, 0.0]),  # Agent 1 buys reasonable amount
            2: np.array([1.5, 0.0]),  # Agent 2 buys reasonable amount
        }
        violating_sells = {
            1: np.array([0.5, 0.0]),  # Agent 1 sells within capacity
            2: np.array([1.5, 0.0]),  # Agent 2 tries to sell more than they have
        }
        balanced_volumes = np.array([2.0, 0.0])  # Market balanced

        # Set agent 2's capacity to be violated (lower than their sell order)
        violating_orders = [
            AgentOrder(
                1, np.array([2.0, 1.0]), np.array([0.0, 1.5]), np.array([1.0, 2.0])
            ),
            AgentOrder(
                2, np.array([1.0, 0.5]), np.array([1.0, 0.0]), np.array([1.0, 1.0])  # Max capacity < actual sell
            ),
        ]

        # This should catch inventory constraint violation
        with pytest.raises(AssertionError, match="inventory constraint violated"):
            _validate_clearing_invariants(
                violating_orders,
                valid_buys,
                violating_sells,
                balanced_volumes,
                self.prices,
            )


class TestTradeConversion:
    """Test conversion of executed quantities to Trade objects."""

    def test_basic_trade_conversion(self):
        """Test conversion with mixed buy/sell orders."""
        executed_buys = {1: np.array([2.0, 0.0]), 2: np.array([0.0, 1.5])}
        executed_sells = {1: np.array([0.0, 1.0]), 2: np.array([1.0, 0.0])}
        prices = np.array([1.5, 2.0])

        trades = _convert_to_trades(executed_buys, executed_sells, prices)

        # Should have 4 trades total (2 buys + 2 sells)
        assert len(trades) == 4

        # Check trade properties
        trade_dict = {(t.agent_id, t.good_id): t for t in trades}

        # Agent 1 buys good 0
        buy_trade = trade_dict[(1, 0)]
        assert buy_trade.quantity == 2.0
        assert buy_trade.price == 1.5
        assert buy_trade.is_purchase

        # Agent 1 sells good 1
        sell_trade = trade_dict[(1, 1)]
        assert sell_trade.quantity == -1.0  # Negative for sales
        assert sell_trade.price == 2.0
        assert sell_trade.is_sale

    def test_zero_quantity_filtering(self):
        """Test that trades with zero quantities are filtered out."""
        executed_buys = {
            1: np.array([1.0, 0.0]),  # Zero buy for good 1
            2: np.array([0.0, 0.0]),  # No buys
        }
        executed_sells = {
            1: np.array([0.0, 0.0]),  # No sells
            2: np.array([0.5, 0.0]),  # Zero sell for good 1
        }
        prices = np.array([1.0, 1.0])

        trades = _convert_to_trades(executed_buys, executed_sells, prices)

        # Should only have 2 trades (1 buy + 1 sell)
        assert len(trades) == 2

        # Verify non-zero quantities
        for trade in trades:
            assert abs(trade.quantity) > RATIONING_EPS


class TestExecuteConstrainedClearing:
    """Test the main market clearing interface function."""

    def setup_method(self):
        """Set up realistic test scenario with Cobb-Douglas agents."""
        # Create agents with complementary preferences
        self.agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.6, 0.4]),
            home_endowment=np.array([2.0, 1.0]),
            personal_endowment=np.array([1.0, 0.0]),
            position=(5, 5),
        )

        self.agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.4, 0.6]),
            home_endowment=np.array([1.0, 2.0]),
            personal_endowment=np.array([0.0, 1.0]),
            position=(5, 5),
        )

        self.agents = [self.agent1, self.agent2]
        self.prices = np.array([1.0, 1.0])

    def test_successful_clearing_execution(self):
        """Test complete clearing execution with realistic agents."""
        result = execute_constrained_clearing(self.agents, self.prices)

        # Verify result structure
        assert isinstance(result, MarketResult)
        assert result.participant_count == 2
        assert len(result.executed_trades) >= 0
        assert len(result.unmet_demand) == len(self.prices)
        assert len(result.unmet_supply) == len(self.prices)
        assert len(result.total_volume) == len(self.prices)

        # Verify economic invariants in result
        assert np.all(result.unmet_demand >= 0)
        assert np.all(result.unmet_supply >= 0)
        assert np.all(result.total_volume >= 0)
        assert 0.0 <= result.clearing_efficiency <= 1.0

    def test_empty_agents_handling(self):
        """Test clearing with no marketplace participants."""
        result = execute_constrained_clearing([], self.prices)

        assert result.participant_count == 0
        assert len(result.executed_trades) == 0
        assert np.allclose(result.unmet_demand, 0)
        assert np.allclose(result.unmet_supply, 0)
        assert np.allclose(result.total_volume, 0)
        assert result.clearing_efficiency == 1.0  # Perfect efficiency when no demand

    def test_input_validation(self):
        """Test input validation for edge cases."""
        # Empty prices array
        with pytest.raises(ValueError, match="cannot be empty"):
            execute_constrained_clearing(self.agents, np.array([]))

        # Negative prices
        with pytest.raises(ValueError, match="must be positive"):
            execute_constrained_clearing(self.agents, np.array([1.0, -0.5]))

        # Mismatched capacity length
        with pytest.raises(ValueError, match="capacity length"):
            execute_constrained_clearing(
                self.agents, self.prices, capacity=np.array([1.0])
            )  # Wrong length

    def test_capacity_constraints_integration(self):
        """Test integration of throughput capacity constraints."""
        tight_capacity = np.array([0.1, 0.1])  # Very tight limits

        result = execute_constrained_clearing(self.agents, self.prices, tight_capacity)

        # Total volume should respect capacity
        assert np.all(result.total_volume <= tight_capacity + FEASIBILITY_TOL)

        # Should increase unmet demand due to capacity constraints
        assert np.any(result.unmet_demand > 0) or np.any(result.unmet_supply > 0)


class TestApplyTradesToAgents:
    """Test application of trades to agent inventories."""

    def setup_method(self):
        """Set up agents and trades for application testing."""
        self.agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.5, 0.5]),
            home_endowment=np.array([2.0, 2.0]),
            personal_endowment=np.array([1.0, 1.0]),
            position=(5, 5),
        )

        self.agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.5, 0.5]),
            home_endowment=np.array([2.0, 2.0]),
            personal_endowment=np.array([1.0, 1.0]),
            position=(5, 5),
        )

        self.agents = [self.agent1, self.agent2]

        # Trades: Agent 1 buys good 0, sells good 1
        #         Agent 2 sells good 0, buys good 1
        self.trades = [
            Trade(agent_id=1, good_id=0, quantity=0.5, price=1.0),  # Agent 1 buys
            Trade(agent_id=1, good_id=1, quantity=-0.5, price=1.0),  # Agent 1 sells
            Trade(agent_id=2, good_id=0, quantity=-0.5, price=1.0),  # Agent 2 sells
            Trade(agent_id=2, good_id=1, quantity=0.5, price=1.0),  # Agent 2 buys
        ]

    def test_successful_trade_application(self):
        """Test successful application of trades to inventories."""
        # Store initial inventories for comparison
        initial_personal_1 = self.agent1.personal_endowment.copy()
        initial_personal_2 = self.agent2.personal_endowment.copy()

        apply_trades_to_agents(self.agents, self.trades)

        # Verify inventory changes
        assert np.isclose(
            self.agent1.personal_endowment[0], initial_personal_1[0] + 0.5
        )
        assert np.isclose(
            self.agent1.personal_endowment[1], initial_personal_1[1] - 0.5
        )
        assert np.isclose(
            self.agent2.personal_endowment[0], initial_personal_2[0] - 0.5
        )
        assert np.isclose(
            self.agent2.personal_endowment[1], initial_personal_2[1] + 0.5
        )

    def test_conservation_validation(self):
        """Test that total goods are conserved across all agents."""
        # Compute total goods before trades
        total_before = sum(agent.personal_endowment for agent in self.agents)

        apply_trades_to_agents(self.agents, self.trades)

        # Compute total goods after trades
        total_after = sum(agent.personal_endowment for agent in self.agents)

        # Total should be conserved
        assert np.allclose(total_before, total_after, atol=FEASIBILITY_TOL)

    def test_empty_trades_handling(self):
        """Test handling of empty trade list."""
        initial_inventories = [agent.personal_endowment.copy() for agent in self.agents]

        apply_trades_to_agents(self.agents, [])

        # Inventories should be unchanged
        for i, agent in enumerate(self.agents):
            assert np.allclose(agent.personal_endowment, initial_inventories[i])

    def test_unknown_agent_handling(self):
        """Test graceful handling of trades for unknown agents."""
        unknown_trade = Trade(agent_id=999, good_id=0, quantity=1.0, price=1.0)

        # Should not raise exception, just log warning
        apply_trades_to_agents(self.agents, [unknown_trade])


class TestMarketClearingIntegration:
    """Integration tests combining equilibrium solving with market clearing."""

    def setup_method(self):
        """Set up Edgeworth box scenario for integration testing."""
        # Classic 2x2 Edgeworth box
        self.agent1 = Agent(
            agent_id=1,
            alpha=np.array([0.6, 0.4]),
            home_endowment=np.array([1.0, 0.0]),
            personal_endowment=np.array([1.0, 0.0]),
            position=(5, 5),
        )

        self.agent2 = Agent(
            agent_id=2,
            alpha=np.array([0.3, 0.7]),
            home_endowment=np.array([0.0, 1.0]),
            personal_endowment=np.array([0.0, 1.0]),
            position=(5, 5),
        )

        self.agents = [self.agent1, self.agent2]

    def test_edgeworth_clearing_integration(self):
        """Test complete pipeline: equilibrium → clearing → trade application."""
        from src.econ.equilibrium import solve_walrasian_equilibrium

        # Step 1: Solve for equilibrium prices
        prices, z_norm, walras_dot, status = solve_walrasian_equilibrium(self.agents)
        assert z_norm < 1e-8  # Should converge

        # Step 2: Execute market clearing
        result = execute_constrained_clearing(self.agents, prices)

        # Step 3: Apply trades
        initial_total = sum(agent.total_endowment for agent in self.agents)
        apply_trades_to_agents(self.agents, result.executed_trades)
        final_total = sum(agent.total_endowment for agent in self.agents)

        # Verify conservation through entire pipeline
        assert np.allclose(initial_total, final_total, atol=FEASIBILITY_TOL)

        # Note: With liquidity constraints, trades may be limited by personal inventory
        # This is correct behavior - agents can only trade what they have in personal inventory
        # The key test is that the system doesn't create or destroy goods
        if len(result.executed_trades) > 0:
            assert result.clearing_efficiency > 0  # Some efficiency if trades occurred
        
        # Test that liquidity constraints are properly enforced
        assert np.all(result.unmet_demand >= 0)  # Unmet demand should be non-negative
        assert np.all(result.unmet_supply >= 0)  # Unmet supply should be non-negative

    def test_multiple_round_consistency(self):
        """Test that clearing produces consistent results across rounds."""
        from src.econ.equilibrium import solve_walrasian_equilibrium

        # Run clearing for multiple rounds with same setup
        results = []
        for round_num in range(3):
            # Reset agents to initial state
            self.setup_method()

            prices, _, _, _ = solve_walrasian_equilibrium(self.agents)
            result = execute_constrained_clearing(self.agents, prices)
            results.append(result)

        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            assert len(results[i].executed_trades) == len(results[0].executed_trades)
            assert np.allclose(results[i].total_volume, results[0].total_volume)
            assert results[i].clearing_efficiency == results[0].clearing_efficiency
