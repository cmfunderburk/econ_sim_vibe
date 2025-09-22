#!/usr/bin/env python3
"""DEPRECATED MODULE (2025-09-21)

This legacy runner has been superseded by `scripts/run_simulation.py` which now
delegates to the library runtime API in `src/core/simulation.py`.

Why deprecated:
- Duplicated SimulationState / initialization logic
- Divergent order generation semantics
- Lacked structured logging integration

Retained temporarily to avoid breaking any unpublished local workflows; direct
execution now raises a RuntimeError guiding users to the supported CLI.

Planned removal after packaging & console entry points are in place.
"""

import argparse
import sys
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml
import numpy as np

# Import from source package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.agent import Agent
from core.types import SimulationState
from spatial.grid import Grid
from econ.equilibrium import solve_walrasian_equilibrium
from econ.market import execute_constrained_clearing


@dataclass
class SimulationConfig:
    """Configuration for spatial simulation."""

    name: str
    n_agents: int
    n_goods: int
    grid_width: int
    grid_height: int
    marketplace_width: int
    marketplace_height: int
    movement_cost: float
    max_rounds: int
    random_seed: int


@dataclass
class SimulationState:
    """Current state of simulation."""

    round: int
    agents: List[Agent]
    grid: Grid
    prices: np.ndarray
    # Track cumulative travel costs for each agent
    agent_travel_costs: Dict[int, float]  # agent_id -> cumulative travel cost


def load_config(config_path: str) -> SimulationConfig:
    """Load simulation configuration from YAML file."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    sim_config = config_data.get("simulation", {})

    return SimulationConfig(
        name=config_data.get("name", "unnamed"),
        n_agents=sim_config.get("n_agents", 10),
        n_goods=sim_config.get("n_goods", 3),
        grid_width=sim_config.get("grid_width", 15),
        grid_height=sim_config.get("grid_height", 15),
        marketplace_width=sim_config.get("marketplace_width", 2),
        marketplace_height=sim_config.get("marketplace_height", 2),
        movement_cost=sim_config.get("movement_cost", 0.1),
        max_rounds=sim_config.get("max_rounds", 50),
        random_seed=sim_config.get("random_seed", 42),
    )


def initialize_simulation(config: SimulationConfig) -> SimulationState:
    """Initialize agents and grid from configuration."""
    # Set seeds for reproducibility
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Create grid
    grid = Grid(
        width=config.grid_width,
        height=config.grid_height,
        marketplace_width=config.marketplace_width,
        marketplace_height=config.marketplace_height,
    )

    # Create agents with random preferences and endowments
    agents = []
    for i in range(config.n_agents):
        # Random preference weights (will be normalized)
        alpha = np.random.dirichlet(np.ones(config.n_goods))

        # Random endowments - split between home and personal
        total_endowment = np.random.exponential(2.0, config.n_goods)

        # Split randomly between home and personal
        split_fraction = np.random.beta(2, 2, config.n_goods)
        home_endowment = total_endowment * split_fraction
        personal_endowment = total_endowment * (1 - split_fraction)

        # Random position
        position = (
            np.random.randint(0, config.grid_width),
            np.random.randint(0, config.grid_height),
        )

        agent = Agent(
            agent_id=i,
            alpha=alpha,
            home_endowment=home_endowment,
            personal_endowment=personal_endowment,
            position=position,
        )

        agents.append(agent)
        grid.place_agent(i, position)

    # Initialize prices (equal to 1 for all goods, normalized)
    initial_prices = np.ones(config.n_goods)
    initial_prices[0] = 1.0  # Numéraire constraint

    # Initialize travel cost tracking
    agent_travel_costs = {agent.agent_id: 0.0 for agent in agents}

    return SimulationState(
        round=0,
        agents=agents,
        grid=grid,
        prices=initial_prices,
        agent_travel_costs=agent_travel_costs,
    )


def run_simulation_round(
    state: SimulationState, config: SimulationConfig
) -> SimulationState:
    """Execute one round of spatial simulation."""

    # 1. Move agents toward marketplace (one step each)
    for agent in state.agents:
        distance_moved = state.grid.move_agent_toward_marketplace(agent.agent_id)
        # Update agent's position
        new_pos = state.grid.get_position(agent.agent_id)
        agent.position = (new_pos.x, new_pos.y)

        # Apply movement cost (budget reduction)
        if distance_moved > 0 and config.movement_cost > 0:
            # Implement travel cost integration: track cumulative travel costs per agent
            travel_cost_this_round = config.movement_cost * distance_moved
            state.agent_travel_costs[agent.agent_id] += travel_cost_this_round
            print(
                f"Agent {agent.agent_id} moved {distance_moved} steps, cost: {travel_cost_this_round:.3f}, cumulative: {state.agent_travel_costs[agent.agent_id]:.3f}"
            )

    # 2. Determine marketplace participants
    marketplace_agent_ids = state.grid.get_agents_in_marketplace()
    marketplace_agents = [
        agent for agent in state.agents if agent.agent_id in marketplace_agent_ids
    ]

    # 3. Solve equilibrium with marketplace participants only
    if len(marketplace_agents) >= 2:
        # Filter zero-wealth agents for numerical stability
        viable_agents = []
        for agent in marketplace_agents:
            # Use total endowment for LTE pricing (not travel-adjusted)
            wealth = np.dot(state.prices, agent.total_endowment)
            if wealth > 1e-10:
                viable_agents.append(agent)

        if len(viable_agents) >= 2 and len(state.prices) >= 2:
            try:
                # Solve for equilibrium prices using total endowments (LTE)
                new_prices, convergence_norm = solve_walrasian_equilibrium(
                    viable_agents
                )
                if new_prices is not None:
                    state.prices = new_prices
                    print(
                        f"Round {state.round}: New prices = {state.prices}, convergence = {convergence_norm:.2e}"
                    )
                else:
                    print(
                        f"Round {state.round}: Equilibrium solver failed, keeping previous prices"
                    )
            except Exception as e:
                print(f"Round {state.round}: Equilibrium error: {e}")

    # 4. Generate orders using travel-adjusted budgets
    orders = []
    for agent in marketplace_agents:
        # Calculate travel-adjusted wealth: w_i = max(0, p·ω_total - κ·d_i)
        base_wealth = np.dot(state.prices, agent.total_endowment)
        cumulative_travel_cost = state.agent_travel_costs[agent.agent_id]
        travel_adjusted_wealth = max(0.0, base_wealth - cumulative_travel_cost)

        # Generate optimal demand with travel-adjusted budget
        if travel_adjusted_wealth > 1e-10:
            optimal_demand = agent.demand(state.prices, wealth=travel_adjusted_wealth)

            # Convert to buy/sell orders (demand - personal inventory)
            net_order = optimal_demand - agent.personal_endowment

            # Create buy/sell orders
            for good_idx, quantity in enumerate(net_order):
                if abs(quantity) > 1e-10:  # Non-trivial order
                    if quantity > 0:
                        # Buy order
                        orders.append(
                            {
                                "agent_id": agent.agent_id,
                                "good": good_idx,
                                "quantity": quantity,
                                "is_buy": True,
                            }
                        )
                    else:
                        # Sell order
                        orders.append(
                            {
                                "agent_id": agent.agent_id,
                                "good": good_idx,
                                "quantity": -quantity,
                                "is_buy": False,
                            }
                        )
        else:
            print(
                f"Agent {agent.agent_id} has insufficient wealth after travel costs: base={base_wealth:.3f}, travel_cost={cumulative_travel_cost:.3f}"
            )

    # 5. Execute constrained clearing
    if orders and len(marketplace_agents) >= 2:
        try:
            trades = execute_constrained_clearing(
                orders, marketplace_agents, state.prices
            )
            print(f"Round {state.round}: Executed {len(trades)} trades")
        except Exception as e:
            print(f"Round {state.round}: Market clearing error: {e}")

    # Update round counter
    state.round += 1

    return state


def main():
    parser = argparse.ArgumentParser(description="Run spatial economic simulation")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/", help="Output directory")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.seed:
        config.random_seed = args.seed

    print(f"Starting simulation: {config.name}")
    print(f"Grid size: {config.grid_width}×{config.grid_height}")
    print(f"Marketplace: {config.marketplace_width}×{config.marketplace_height}")
    print(f"Agents: {config.n_agents}, Goods: {config.n_goods}")
    print(f"Movement cost: {config.movement_cost}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Random seed: {config.random_seed}")

    # Initialize simulation
    state = initialize_simulation(config)

    # Run simulation rounds
    results = []
    for round_num in range(config.max_rounds):
        state = run_simulation_round(state, config)

        # Record round results
        round_result = {
            "round": state.round,
            "prices": state.prices.tolist(),
            "agents_in_marketplace": len(state.grid.get_agents_in_marketplace()),
            "total_agents": len(state.agents),
            "cumulative_travel_costs": dict(state.agent_travel_costs),
        }
        results.append(round_result)

        # Check termination conditions
        agents_in_marketplace = len(state.grid.get_agents_in_marketplace())
        if agents_in_marketplace == len(state.agents):
            print(f"All agents reached marketplace at round {state.round}")
            break

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    final_results = {
        "config": {
            "name": config.name,
            "n_agents": config.n_agents,
            "n_goods": config.n_goods,
            "grid_size": [config.grid_width, config.grid_height],
            "movement_cost": config.movement_cost,
            "random_seed": config.random_seed,
        },
        "final_round": state.round,
        "rounds": results,
        "final_prices": state.prices.tolist(),
        "agents_in_marketplace": len(state.grid.get_agents_in_marketplace()),
        "total_agents": len(state.agents),
        "grid_summary": state.grid.get_grid_summary(),
        "final_travel_costs": dict(state.agent_travel_costs),
    }

    output_file = output_dir / f"simulation_{config.name}_{config.random_seed}.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Final travel costs: {state.agent_travel_costs}")


if __name__ == "__main__":  # pragma: no cover - explicit runtime guard
    raise RuntimeError(
        "scripts/run_simulation_fixed.py is deprecated. Use scripts/run_simulation.py instead."
    )
