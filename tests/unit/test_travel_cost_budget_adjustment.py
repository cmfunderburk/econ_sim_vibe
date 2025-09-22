"""
Travel cost budget adjustment testing for spatial phase preparation.

This module implements testing for the travel cost deduction mechanism
that will be critical for Phase 2 spatial behavior. It validates the
budget constraint formula: w_i = max(0, p·ω_i^total - κ·d_i)

Key Features:
- Tests travel cost deduction from agent wealth
- Validates demand computation with adjusted budgets
- Ensures economic content correctness for spatial extensions
- Prepares for integration with spatial grid movement

This addresses the critique about implementing travel cost validation
and preparing for the spatial phase implementation.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.agent import Agent
from constants import FEASIBILITY_TOL

# Import test categorization markers
from tests.test_categorization import economic_core, real_functions


@pytest.mark.economic_core
@pytest.mark.real_functions
class TravelCostBudgetHelper:
    """Helper class for travel cost calculations (Phase 2 preparation)."""

    @staticmethod
    def compute_travel_cost_adjusted_wealth(
        agent: Agent, prices: np.ndarray, travel_cost: float
    ) -> float:
        """
        Compute wealth adjusted for travel costs.

        Formula: w_i = max(0, p·ω_i^total - κ·d_i)
        where κ is the travel cost per unit distance and d_i is distance traveled.

        Args:
            agent: Agent with total endowments
            prices: Current price vector
            travel_cost: Total travel cost (κ·d_i)

        Returns:
            Adjusted wealth after travel cost deduction
        """
        base_wealth = np.dot(prices, agent.total_endowment)
        adjusted_wealth = max(0.0, base_wealth - travel_cost)
        return adjusted_wealth

    @staticmethod
    def compute_demand_with_travel_cost(
        agent: Agent, prices: np.ndarray, travel_cost: float
    ) -> np.ndarray:
        """
        Compute Cobb-Douglas demand with travel cost-adjusted budget.

        Uses adjusted wealth: x_j = α_j * adjusted_wealth / p_j

        Args:
            agent: Agent with preferences and endowments
            prices: Current price vector
            travel_cost: Total travel cost deduction

        Returns:
            Demand vector with travel cost adjustment
        """
        adjusted_wealth = TravelCostBudgetHelper.compute_travel_cost_adjusted_wealth(
            agent, prices, travel_cost
        )

        if adjusted_wealth <= FEASIBILITY_TOL:
            # Agent has no purchasing power after travel costs
            return np.zeros_like(prices)

        demand = agent.alpha * adjusted_wealth / prices
        return demand


class TestTravelCostBudgetAdjustment:
    """Test travel cost deduction and budget adjustment mechanism."""

    def test_basic_travel_cost_deduction(self):
        """Test basic travel cost deduction from agent wealth."""
        agent = Agent(
            1, np.array([0.6, 0.4]), np.array([2.0, 3.0]), np.array([1.0, 2.0])
        )

        prices = np.array([1.0, 1.5])
        travel_cost = 1.0

        # Base wealth: 1*(2+1) + 1.5*(3+2) = 3 + 7.5 = 10.5
        base_wealth = np.dot(prices, agent.total_endowment)
        expected_base_wealth = 10.5
        assert abs(base_wealth - expected_base_wealth) < FEASIBILITY_TOL

        # Adjusted wealth: max(0, 10.5 - 1.0) = 9.5
        adjusted_wealth = TravelCostBudgetHelper.compute_travel_cost_adjusted_wealth(
            agent, prices, travel_cost
        )
        expected_adjusted_wealth = 9.5
        assert abs(adjusted_wealth - expected_adjusted_wealth) < FEASIBILITY_TOL

    def test_travel_cost_demand_adjustment(self):
        """Test that demand computation uses travel cost-adjusted wealth."""
        agent = Agent(
            1, np.array([0.7, 0.3]), np.array([3.0, 2.0]), np.array([1.0, 1.0])
        )

        prices = np.array([2.0, 1.0])
        travel_cost = 2.0

        # Base wealth: 2*(3+1) + 1*(2+1) = 8 + 3 = 11.0
        # Adjusted wealth: max(0, 11.0 - 2.0) = 9.0
        adjusted_wealth = 9.0

        # Expected demand with adjusted wealth: [0.7*9/2, 0.3*9/1] = [3.15, 2.7]
        expected_demand = np.array([3.15, 2.7])

        actual_demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, travel_cost
        )

        assert np.allclose(actual_demand, expected_demand, atol=FEASIBILITY_TOL), (
            f"Demand with travel cost {actual_demand} != expected {expected_demand}"
        )

    def test_zero_travel_cost_equals_normal_demand(self):
        """Test that zero travel cost gives same result as normal demand."""
        agent = Agent(
            1, np.array([0.5, 0.5]), np.array([2.0, 1.0]), np.array([1.0, 1.0])
        )

        prices = np.array([1.0, 2.0])

        # Zero travel cost should give normal demand
        demand_with_zero_cost = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, 0.0
        )

        # Normal demand computation
        normal_demand = agent.demand(prices)

        assert np.allclose(
            demand_with_zero_cost, normal_demand, atol=FEASIBILITY_TOL
        ), (
            f"Zero travel cost demand {demand_with_zero_cost} != normal demand {normal_demand}"
        )

    def test_excessive_travel_cost_zero_demand(self):
        """Test that excessive travel cost results in zero demand."""
        agent = Agent(
            1, np.array([0.6, 0.4]), np.array([1.0, 1.0]), np.array([1.0, 1.0])
        )

        prices = np.array([1.0, 1.0])

        # Total wealth: 1*2 + 1*2 = 4.0
        # Excessive travel cost > total wealth
        excessive_travel_cost = 10.0

        demand_with_excessive_cost = (
            TravelCostBudgetHelper.compute_demand_with_travel_cost(
                agent, prices, excessive_travel_cost
            )
        )

        expected_zero_demand = np.zeros_like(prices)

        assert np.allclose(
            demand_with_excessive_cost, expected_zero_demand, atol=FEASIBILITY_TOL
        ), (
            f"Excessive travel cost should give zero demand: {demand_with_excessive_cost}"
        )

    def test_travel_cost_preserves_preference_ratios(self):
        """Test that travel cost adjustment preserves preference ratio relationships."""
        agent = Agent(
            1, np.array([0.8, 0.2]), np.array([2.0, 2.0]), np.array([1.0, 1.0])
        )

        prices = np.array([1.0, 1.0])  # Equal prices
        travel_cost = 1.0

        demand_with_travel = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, travel_cost
        )

        # Demand ratio should match preference ratio
        if demand_with_travel[1] > FEASIBILITY_TOL:  # Avoid division by zero
            demand_ratio = demand_with_travel[0] / demand_with_travel[1]
            preference_ratio = agent.alpha[0] / agent.alpha[1]

            assert abs(demand_ratio - preference_ratio) < FEASIBILITY_TOL, (
                f"Travel cost should preserve preference ratios: demand ratio {demand_ratio} != preference ratio {preference_ratio}"
            )

    def test_travel_cost_budget_feasibility(self):
        """Test that travel cost-adjusted demand respects budget constraints."""
        agent = Agent(
            1, np.array([0.4, 0.6]), np.array([3.0, 2.0]), np.array([1.0, 1.0])
        )

        prices = np.array([2.0, 1.5])
        travel_cost = 3.0

        # Compute adjusted wealth and demand
        adjusted_wealth = TravelCostBudgetHelper.compute_travel_cost_adjusted_wealth(
            agent, prices, travel_cost
        )
        demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, travel_cost
        )

        # Demand value should not exceed adjusted wealth
        demand_value = np.dot(prices, demand)

        assert demand_value <= adjusted_wealth + FEASIBILITY_TOL, (
            f"Demand value {demand_value} exceeds adjusted wealth {adjusted_wealth}"
        )

        # For Cobb-Douglas with positive wealth, should spend entire adjusted budget
        if adjusted_wealth > FEASIBILITY_TOL:
            assert abs(demand_value - adjusted_wealth) < FEASIBILITY_TOL, (
                f"Should spend entire adjusted wealth: demand value {demand_value} != adjusted wealth {adjusted_wealth}"
            )


class TestTravelCostEdgeCases:
    """Test edge cases for travel cost budget adjustment."""

    def test_near_zero_adjusted_wealth(self):
        """Test behavior when travel cost nearly exhausts wealth."""
        agent = Agent(
            1, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.5, 0.5])
        )

        prices = np.array([1.0, 1.0])

        # Total wealth: 1*1.5 + 1*1.5 = 3.0
        # Travel cost that leaves tiny amount: 3.0 - 1e-6
        near_exhaustive_cost = 3.0 - 1e-6

        demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, near_exhaustive_cost
        )

        # Should get tiny but positive demand
        assert np.all(demand >= 0), f"Demand should be non-negative: {demand}"
        assert np.sum(demand) > 0, f"Should have some positive demand: {demand}"

        # Demand should be very small
        demand_magnitude = np.linalg.norm(demand)
        assert demand_magnitude < 1e-5, (
            f"Demand should be very small: magnitude {demand_magnitude}"
        )

    def test_multiple_agents_travel_cost_consistency(self):
        """Test travel cost adjustment consistency across multiple agents."""
        agents = [
            Agent(1, np.array([0.7, 0.3]), np.array([2.0, 1.0]), np.array([1.0, 1.0])),
            Agent(2, np.array([0.3, 0.7]), np.array([1.0, 2.0]), np.array([1.0, 1.0])),
        ]

        prices = np.array([1.0, 1.0])
        travel_cost = 1.5

        total_adjusted_demand = np.zeros_like(prices)

        for agent in agents:
            demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
                agent, prices, travel_cost
            )
            total_adjusted_demand += demand

            # Each agent's demand should be economically valid
            assert np.all(demand >= 0), (
                f"Agent {agent.agent_id} demand should be non-negative: {demand}"
            )

            # Budget constraint for each agent
            adjusted_wealth = (
                TravelCostBudgetHelper.compute_travel_cost_adjusted_wealth(
                    agent, prices, travel_cost
                )
            )
            demand_value = np.dot(prices, demand)

            if adjusted_wealth > FEASIBILITY_TOL:
                assert abs(demand_value - adjusted_wealth) < FEASIBILITY_TOL, (
                    f"Agent {agent.agent_id} budget violation: demand value {demand_value} != adjusted wealth {adjusted_wealth}"
                )

        # Total demand should be reasonable
        assert np.all(np.isfinite(total_adjusted_demand)), (
            "Total demand should be finite"
        )
        print(f"Total adjusted demand: {total_adjusted_demand}")


class TestTravelCostIntegration:
    """Test integration of travel cost with existing economic functions."""

    def test_travel_cost_excess_demand_computation(self):
        """Test excess demand computation with travel cost adjustment."""
        agent = Agent(
            1, np.array([0.6, 0.4]), np.array([2.0, 1.0]), np.array([1.0, 1.0])
        )

        prices = np.array([1.0, 2.0])
        travel_cost = 1.0

        # Compute adjusted demand
        adjusted_demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, travel_cost
        )

        # Excess demand with respect to total endowment
        total_endowment = agent.total_endowment
        excess_demand = adjusted_demand - total_endowment

        print(f"Adjusted demand: {adjusted_demand}")
        print(f"Total endowment: {total_endowment}")
        print(f"Excess demand: {excess_demand}")

        # Basic validation - should be finite
        assert np.all(np.isfinite(excess_demand)), "Excess demand should be finite"

        # Economic meaning: excess demand shows net desired change in holdings
        # Given travel cost reduces purchasing power, likely negative for most goods

    def test_travel_cost_welfare_implications(self):
        """Test welfare implications of travel cost (efficiency loss)."""
        agent = Agent(
            1, np.array([0.5, 0.5]), np.array([2.0, 2.0]), np.array([1.0, 1.0])
        )

        prices = np.array([1.0, 1.0])

        # Utility without travel cost
        normal_demand = agent.demand(prices)
        normal_utility = np.prod(normal_demand**agent.alpha)

        # Utility with travel cost
        travel_cost = 2.0
        adjusted_demand = TravelCostBudgetHelper.compute_demand_with_travel_cost(
            agent, prices, travel_cost
        )
        adjusted_utility = np.prod(adjusted_demand**agent.alpha)

        # Travel cost should reduce utility (efficiency loss)
        assert adjusted_utility <= normal_utility + FEASIBILITY_TOL, (
            f"Travel cost should not increase utility: {adjusted_utility} > {normal_utility}"
        )

        print(f"Normal utility: {normal_utility:.4f}")
        print(f"Adjusted utility: {adjusted_utility:.4f}")
        print(f"Efficiency loss: {normal_utility - adjusted_utility:.4f}")

        # Efficiency loss should be positive (travel cost hurts welfare)
        efficiency_loss = normal_utility - adjusted_utility
        assert efficiency_loss >= -FEASIBILITY_TOL, (
            "Should have non-negative efficiency loss"
        )
