#!/usr/bin/env python3
"""
Main simulation runner script.

Usage:
    python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42
    python scripts/run_simulation.py --config config/zero_movement_cost.yaml --seed 123 --output results/
"""

import argparse
import sys
import random
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml
import numpy as np

# Import our implemented modules
sys.path.append(str(Path(__file__).parent.parent))
from src.core.agent import Agent
from src.econ.equilibrium import solve_walrasian_equilibrium
from src.econ.market import execute_constrained_clearing
from src.spatial.grid import Grid, Position, create_random_positions


class AgentPhase(str, Enum):
    """Lifecycle phases for agent movement between home and marketplace."""

    HOME_PREP = "home_prep"
    TO_MARKET = "to_market"
    AT_MARKET = "at_market"
    TO_HOME = "to_home"


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""

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
    trades: List[Any]
    # Track cumulative travel costs for each agent
    agent_travel_costs: Dict[int, float]  # agent_id -> cumulative travel cost
    agent_phases: Dict[int, AgentPhase]


def load_config(config_path: Path) -> SimulationConfig:
    """Load simulation configuration from YAML."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    sim_config = config_data.get("simulation", {})
    agents_config = config_data.get("agents", {})
    economy_config = config_data.get("economy", {})

    return SimulationConfig(
        name=sim_config.get("name", "Unknown Simulation"),
        n_agents=agents_config.get("count", 10),
        n_goods=economy_config.get("goods", 3),
        grid_width=economy_config.get("grid_size", [15, 15])[0],
        grid_height=economy_config.get("grid_size", [15, 15])[1],
        marketplace_width=economy_config.get("marketplace_size", [2, 2])[0],
        marketplace_height=economy_config.get("marketplace_size", [2, 2])[1],
        movement_cost=economy_config.get("movement_cost", 0.1),
        max_rounds=sim_config.get("max_rounds", 50),
        random_seed=sim_config.get("seed", 42),
    )


def initialize_simulation(config: SimulationConfig) -> SimulationState:
    """Initialize agents and grid from configuration."""
    # Set seeds for reproducibility
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Create grid
    grid = Grid(
        config.grid_width,
        config.grid_height,
        config.marketplace_width,
        config.marketplace_height,
    )

    # Generate random starting positions
    positions = create_random_positions(
        config.n_agents, config.grid_width, config.grid_height, config.random_seed
    )

    # Create agents with random endowments and preferences
    agents = []
    for i in range(config.n_agents):
        # Random Cobb-Douglas preferences (Dirichlet)
        alpha = np.random.dirichlet(np.ones(config.n_goods))
        alpha = np.maximum(alpha, 0.05)  # Ensure interiority
        alpha = alpha / np.sum(alpha)  # Renormalize

        # Random endowments (exponential distribution)
        home_endowment = np.random.exponential(2.0, config.n_goods)
        personal_endowment = np.random.exponential(1.0, config.n_goods)

        # Create agent
        agent_home = (positions[i].x, positions[i].y)
        agent = Agent(
            agent_id=i,
            alpha=alpha,
            home_endowment=home_endowment,
            personal_endowment=personal_endowment,
            position=agent_home,
            home_position=agent_home,
        )
        agents.append(agent)

        # Add to grid
        grid.add_agent(i, positions[i])

    return SimulationState(
        round=0,
        agents=agents,
        grid=grid,
        prices=np.ones(config.n_goods),  # Initial uniform prices
        trades=[],
        agent_travel_costs={agent.agent_id: 0.0 for agent in agents},
        agent_phases={agent.agent_id: AgentPhase.HOME_PREP for agent in agents},
    )


def run_simulation_round(
    state: SimulationState, config: SimulationConfig
) -> SimulationState:
    """Execute one round of spatial simulation."""

    # 1. Home preparation: load inventory when agents are at home
    for agent in state.agents:
        phase = state.agent_phases[agent.agent_id]

        if phase == AgentPhase.HOME_PREP:
            if agent.is_at_home(agent.home_position):
                agent.load_inventory_for_travel()
                state.agent_phases[agent.agent_id] = AgentPhase.TO_MARKET
            else:
                # Agent is misplaced; send them home before loading
                state.agent_phases[agent.agent_id] = AgentPhase.TO_HOME

    # 2. Move agents according to their lifecycle phase
    for agent in state.agents:
        phase = state.agent_phases[agent.agent_id]
        distance_moved = 0

        if phase == AgentPhase.TO_MARKET:
            distance_moved = state.grid.move_agent_toward_marketplace(agent.agent_id)

            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)

            if state.grid.is_in_marketplace(current_pos):
                state.agent_phases[agent.agent_id] = AgentPhase.AT_MARKET

        elif phase == AgentPhase.TO_HOME:
            home_position = Position(*agent.home_position)
            distance_moved = state.grid.move_agent_toward_position(
                agent.agent_id, home_position
            )

            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)

            if agent.is_at_home(agent.home_position):
                agent.deposit_inventory_at_home()
                state.agent_travel_costs[agent.agent_id] = 0.0
                state.agent_phases[agent.agent_id] = AgentPhase.HOME_PREP

        else:
            # Agent stays in place (either preparing at home or trading in marketplace)
            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)

        if distance_moved > 0 and config.movement_cost > 0:
            travel_cost_this_round = config.movement_cost * distance_moved
            state.agent_travel_costs[agent.agent_id] += travel_cost_this_round

    # 3. Determine marketplace participants (phase-aware)
    marketplace_agent_ids = {
        agent.agent_id
        for agent in state.agents
        if state.agent_phases[agent.agent_id] == AgentPhase.AT_MARKET
        and state.grid.is_in_marketplace(state.grid.get_position(agent.agent_id))
    }
    marketplace_agents = [
        agent for agent in state.agents if agent.agent_id in marketplace_agent_ids
    ]

    # 4. Solve equilibrium with marketplace participants only
    if len(marketplace_agents) >= 2:
        # Filter zero-wealth agents for numerical stability
        viable_agents = []
        for agent in marketplace_agents:
            wealth = np.dot(state.prices, agent.total_endowment)
            if wealth > 1e-10:
                viable_agents.append(agent)

        if len(viable_agents) >= 2 and len(state.prices) >= 2:
            try:
                prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(
                    viable_agents
                )
                if z_rest_norm < 1e-8:  # Check convergence
                    state.prices = prices

                    # 4. Execute market clearing with travel-adjusted budgets
                    # Use core market API and pass cumulative travel costs
                    from src.econ.market import apply_trades_to_agents

                    market_result = execute_constrained_clearing(
                        viable_agents,
                        prices,
                        capacity=None,
                        travel_costs=state.agent_travel_costs,
                    )

                    # Apply executed trades to agent inventories
                    apply_trades_to_agents(viable_agents, market_result.executed_trades)
                    state.trades = market_result.executed_trades

                    # After trading, agents head back home to complete the cycle
                    for agent in viable_agents:
                        state.agent_phases[agent.agent_id] = AgentPhase.TO_HOME
                else:
                    print(
                        f"Warning: Equilibrium solver failed to converge in round {state.round}, z_rest_norm={z_rest_norm}"
                    )
            except Exception as e:
                print(
                    f"Warning: Equilibrium computation failed in round {state.round}: {e}"
                )

    state.round += 1
    return state


def run_simulation(
    config_path: Path,
    output_path: Optional[Path] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run complete simulation from configuration."""

    # Load configuration
    config = load_config(config_path)

    # Apply seed override if provided
    if seed_override is not None:
        config.random_seed = seed_override

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
            "n_marketplace_agents": len(state.grid.get_agents_in_marketplace()),
            "total_agents": len(state.agents),
            "prices": state.prices.tolist(),
            "n_trades": len(state.trades),
            "cumulative_travel_costs": dict(state.agent_travel_costs),
        }
        results.append(round_result)

        # Print progress every 10 rounds
        if state.round % 10 == 0:
            n_market = len(state.grid.get_agents_in_marketplace())
            print(
                f"Round {state.round}: {n_market}/{len(state.agents)} agents in marketplace"
            )

        # Early termination if all agents reach marketplace
        if len(state.grid.get_agents_in_marketplace()) == len(state.agents):
            print(f"All agents reached marketplace at round {state.round}")
            break

    # Compile final results
    simulation_results = {
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

    # Save results if output path specified
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{config.name}_seed{config.random_seed}.json"
        with open(result_file, "w") as f:
            json.dump(simulation_results, f, indent=2)
        print(f"Results saved to: {result_file}")

    return simulation_results


def main():
    parser = argparse.ArgumentParser(description="Run economic simulation")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument(
        "--no-gui", action="store_true", help="Disable pygame visualization"
    )

    args = parser.parse_args()

    # Validate configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Set output directory
    output_path = Path(args.output) if args.output else None

    try:
        # Run simulation
        results = run_simulation(config_path, output_path, args.seed)

        # Print summary
        print(f"\nSimulation completed in {results['final_round']} rounds")
        print(
            f"Final marketplace participation: {results['agents_in_marketplace']}/{results['total_agents']}"
        )
        print(f"Final prices: {results['final_prices']}")

        if results["agents_in_marketplace"] == results["total_agents"]:
            print("✅ All agents reached marketplace!")
        else:
            remaining = results["total_agents"] - results["agents_in_marketplace"]
            print(f"⚠️  {remaining} agents still outside marketplace")

        return 0

    except Exception as e:
        print(f"Error: Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
