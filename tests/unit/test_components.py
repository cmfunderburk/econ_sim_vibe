"""
Unit tests for core simulation components.

Tests individual modules and functions for correctness independent
of full simulation scenarios.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import constants and modules
try:
    from constants import SOLVER_TOL, FEASIBILITY_TOL
except ImportError:
    from src.constants import SOLVER_TOL, FEASIBILITY_TOL

from core.agent import Agent
from core.types import Trade, SimulationState


class TestAgent:
    """Test suite for Agent class functionality."""

    def test_agent_initialization(self):
        """Test Agent class initialization."""
        # Test basic initialization
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
            position=(0, 0),
        )

        assert agent.agent_id == 1
        assert np.allclose(agent.alpha, [0.6, 0.4])
        assert np.allclose(agent.home_endowment, [10.0, 5.0])
        assert np.allclose(agent.personal_endowment, [2.0, 1.0])
        assert agent.position == (0, 0)
        assert agent.n_goods == 2

    def test_alpha_normalization(self):
        """Test that alpha preferences are normalized to sum to 1."""
        # Test normalization
        agent = Agent(
            agent_id=1,
            alpha=[3.0, 2.0],  # Should be normalized to [0.6, 0.4]
            home_endowment=[1.0, 1.0],
            personal_endowment=[0.0, 0.0],
        )

        assert np.allclose(agent.alpha, [0.6, 0.4])
        assert np.allclose(np.sum(agent.alpha), 1.0)

    def test_alpha_interiority(self):
        """Test that alpha values are clipped to ensure interior solutions."""
        # Test minimum alpha enforcement
        agent = Agent(
            agent_id=1,
            alpha=[0.99, 0.01],  # Second alpha below MIN_ALPHA
            home_endowment=[1.0, 1.0],
            personal_endowment=[0.0, 0.0],
        )

        # Should be clipped and renormalized
        assert np.all(agent.alpha >= 0.05)
        assert np.allclose(np.sum(agent.alpha), 1.0)

    def test_negative_endowment_validation(self):
        """Test that negative endowments are rejected."""
        with pytest.raises(ValueError, match="Negative home endowment"):
            Agent(
                agent_id=1,
                alpha=[0.6, 0.4],
                home_endowment=[-1.0, 5.0],
                personal_endowment=[2.0, 1.0],
            )

        with pytest.raises(ValueError, match="Negative personal endowment"):
            Agent(
                agent_id=1,
                alpha=[0.6, 0.4],
                home_endowment=[10.0, 5.0],
                personal_endowment=[2.0, -1.0],
            )

    def test_dimension_consistency(self):
        """Test that all arrays have consistent dimensions."""
        with pytest.raises(ValueError, match="Dimension mismatch"):
            Agent(
                agent_id=1,
                alpha=[0.6, 0.4],
                home_endowment=[10.0, 5.0, 3.0],  # Wrong dimension
                personal_endowment=[2.0, 1.0],
            )

    def test_cobb_douglas_utility(self):
        """Test Cobb-Douglas utility calculation."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
        )

        # Test utility calculation
        consumption = np.array([6.0, 4.0])
        utility = agent.utility(consumption)

        # Expected: U(x) = x1^0.6 * x2^0.4 = 6^0.6 * 4^0.4 ≈ 4.93
        expected_utility = (6.0**0.6) * (4.0**0.4)
        assert np.allclose(utility, expected_utility, rtol=1e-10)

        # Test edge case: zero consumption (should be handled gracefully)
        zero_consumption = np.array([0.0, 4.0])
        utility_zero = agent.utility(zero_consumption)
        assert np.isfinite(utility_zero)
        assert utility_zero > 0

    def test_demand_function(self):
        """Test Cobb-Douglas demand function."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
        )

        prices = np.array([1.0, 0.8])
        demand = agent.demand(prices)

        # Expected demand: x_j = alpha_j * wealth / p_j
        wealth = np.dot(prices, agent.total_endowment)  # 1.0*12 + 0.8*6 = 16.8
        expected_demand = agent.alpha * wealth / prices

        assert np.allclose(demand, expected_demand, rtol=1e-10)

        # Test with custom wealth
        custom_wealth = 20.0
        demand_custom = agent.demand(prices, wealth=custom_wealth)
        expected_custom = agent.alpha * custom_wealth / prices
        assert np.allclose(demand_custom, expected_custom, rtol=1e-10)

    def test_excess_demand(self):
        """Test excess demand calculation."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
        )

        prices = np.array([1.0, 0.8])
        excess_demand = agent.excess_demand(prices)

        # Expected: excess_demand = demand - total_endowment
        demand = agent.demand(prices)
        expected_excess = demand - agent.total_endowment

        assert np.allclose(excess_demand, expected_excess, rtol=1e-10)

    def test_inventory_transfer(self):
        """Test goods transfer between home and personal inventory."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
        )

        initial_total = agent.total_endowment.copy()

        # Test transfer from home to personal
        transfer_amount = np.array([1.0, 0.5])
        agent.transfer_goods(transfer_amount, to_personal=True)

        assert np.allclose(agent.home_endowment, [9.0, 4.5])
        assert np.allclose(agent.personal_endowment, [3.0, 1.5])
        assert np.allclose(agent.total_endowment, initial_total)  # Conservation

        # Test transfer from personal to home
        agent.transfer_goods(transfer_amount, to_personal=False)

        assert np.allclose(agent.home_endowment, [10.0, 5.0])
        assert np.allclose(agent.personal_endowment, [2.0, 1.0])
        assert np.allclose(agent.total_endowment, initial_total)  # Conservation

    def test_inventory_transfer_insufficient_goods(self):
        """Test that transfers with insufficient goods are rejected."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[1.0, 1.0],
            personal_endowment=[0.5, 0.5],
        )

        # Try to transfer more than available from home
        with pytest.raises(ValueError, match="Insufficient home inventory"):
            agent.transfer_goods(np.array([2.0, 0.5]), to_personal=True)

        # Try to transfer more than available from personal
        with pytest.raises(ValueError, match="Insufficient personal inventory"):
            agent.transfer_goods(np.array([1.0, 1.0]), to_personal=False)

    def test_position_tracking(self):
        """Test position tracking and movement."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
            position=(5, 3),
        )

        assert agent.position == (5, 3)

        # Test movement
        new_position = (2, 7)
        agent.move_to(new_position)
        assert agent.position == new_position

    def test_marketplace_detection(self):
        """Test marketplace boundary detection."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
            position=(1, 1),
        )

        # Test marketplace bounds: ((min_x, max_x), (min_y, max_y))
        marketplace_bounds = ((0, 2), (0, 2))  # 3x3 marketplace

        # Agent at (1,1) should be inside
        assert agent.is_at_marketplace(marketplace_bounds)

        # Move outside marketplace
        agent.move_to((3, 1))
        assert not agent.is_at_marketplace(marketplace_bounds)

        # Test edge cases
        agent.move_to((0, 0))  # Corner
        assert agent.is_at_marketplace(marketplace_bounds)

        agent.move_to((2, 2))  # Other corner
        assert agent.is_at_marketplace(marketplace_bounds)

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        agent = Agent(
            agent_id=1,
            alpha=[0.6, 0.4],
            home_endowment=[10.0, 5.0],
            personal_endowment=[2.0, 1.0],
            position=(4, 3),
        )

        marketplace_center = (1, 1)
        distance = agent.distance_to_marketplace(marketplace_center)

        # Manhattan distance: |4-1| + |3-1| = 3 + 2 = 5
        expected_distance = abs(4 - 1) + abs(3 - 1)
        assert distance == expected_distance

        # Test zero distance
        agent.move_to(marketplace_center)
        assert agent.distance_to_marketplace(marketplace_center) == 0


class TestTradeDataclass:
    """Test suite for Trade dataclass."""

    def test_trade_creation(self):
        """Test Trade dataclass creation and validation."""
        trade = Trade(agent_id=1, good_id=0, quantity=2.5, price=1.2)

        assert trade.agent_id == 1
        assert trade.good_id == 0
        assert trade.quantity == 2.5
        assert trade.price == 1.2

        # Test value calculation
        assert trade.value == 2.5 * 1.2

        # Test immutability (frozen dataclass) - should raise FrozenInstanceError
        with pytest.raises(Exception):  # Frozen dataclass raises FrozenInstanceError
            trade.quantity = 3.0


class TestSimulationState:
    """Test suite for SimulationState dataclass."""

    def test_simulation_state_creation(self):
        """Test SimulationState dataclass creation."""
        from src.core.agent import Agent

        # Create mock agents
        agents = [
            Agent(
                agent_id=1,
                alpha=np.array([0.6, 0.4]),
                home_endowment=np.array([1.0, 0.0]),
                personal_endowment=np.array([0.5, 0.5]),
                position=(0, 0),
            ),
            Agent(
                agent_id=2,
                alpha=np.array([0.3, 0.7]),
                home_endowment=np.array([0.0, 1.0]),
                personal_endowment=np.array([0.5, 0.5]),
                position=(1, 1),
            ),
        ]

        trades = [Trade(1, 0, 2.5, 1.2)]
        prices = np.array([1.0, 0.8, 1.1])

        state = SimulationState(
            round_number=1,
            agents=agents,
            prices=prices,
            executed_trades=trades,
            z_rest_norm=1e-9,
            walras_dot=1e-10,
            total_welfare=2.5,
            marketplace_participants=2,
        )

        assert state.round_number == 1
        assert len(state.agents) == 2
        assert np.array_equal(state.prices, prices)
        assert state.executed_trades == trades
        assert state.marketplace_participants == 2

        # Test with None prices (no trading round)
        state_no_prices = SimulationState(
            round_number=2,
            agents=agents,
            prices=None,
            executed_trades=[],
            z_rest_norm=0.0,
            walras_dot=0.0,
            total_welfare=1.0,
            marketplace_participants=0,
        )

        assert state_no_prices.prices is None
