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
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
import numpy as np

# Temporary path hack retained until packaging task (see architectural plan)
sys.path.append(str(Path(__file__).parent.parent))  # TODO: remove after console_scripts

from src.core.simulation import (
    SimulationConfig,
    RuntimeSimulationState,
    initialize_runtime_state,
    run_round,
    AgentPhase,
)
from src.visualization import build_frame, ASCIIRenderer  # type: ignore
try:  # Optional pygame renderer import
    from src.visualization import PygameRenderer  # type: ignore
except Exception:  # pragma: no cover
    PygameRenderer = None  # type: ignore
from src.core.config_validation import validate_simulation_config
from src.core.agent import Agent  # re-used for type hints

# Logging layer (optional)
try:
    from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION
except ImportError:  # pragma: no cover - logging module optional
    RunLogger = None  # type: ignore
    RoundLogRecord = None  # type: ignore
    SCHEMA_VERSION = "0.0.0"  # type: ignore


#############################
# Legacy Dataclasses Removed #
#############################
# Local SimulationState & AgentPhase definitions replaced by runtime module.


def load_config(config_path: Path) -> SimulationConfig:
    """Load simulation configuration from YAML."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    sim_config = config_data.get("simulation", {})
    agents_config = config_data.get("agents", {})
    economy_config = config_data.get("economy", {})

    config = SimulationConfig(
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
    # Validate and raise early if invalid
    validate_simulation_config(config)
    return config


def initialize_simulation(config: SimulationConfig) -> RuntimeSimulationState:  # Backwards compatibility wrapper
    """Wrapper preserving previous function name for CLI; delegates to runtime initializer."""
    random.seed(config.random_seed)
    return initialize_runtime_state(config)


def run_simulation_round(state: RuntimeSimulationState, config: SimulationConfig) -> RuntimeSimulationState:
    """Backward compatible wrapper delegating to core.simulation.run_round."""
    return run_round(state, config)


def run_simulation(
    config_path: Path,
    output_path: Optional[Path] = None,
    seed_override: Optional[int] = None,
    ascii_interval: Optional[int] = None,
    gui: bool = False,
    tick_ms: int = 120,
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

    # Initialize structured logger if output path provided
    run_logger = None  # defer typing; optional logger instance
    if output_path and RunLogger is not None:  # type: ignore
        run_name = f"{config.name}_seed{config.random_seed}"
        run_logger = RunLogger(output_path, run_name)  # type: ignore

    # Run simulation rounds
    results = []
    ascii_renderer = ASCIIRenderer() if ascii_interval else None
    gui_renderer = None
    if gui and PygameRenderer is not None:  # type: ignore
        try:
            gui_renderer = PygameRenderer(tick_ms=tick_ms, title=f"{config.name} (seed {config.random_seed})")
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to initialize GUI renderer: {e}")
            gui_renderer = None

    for _ in range(config.max_rounds):
        state = run_simulation_round(state, config)

        # Record round results (summary)
        round_result = {
            "round": state.round,
            "n_marketplace_agents": len(state.grid.get_agents_in_marketplace()),
            "total_agents": len(state.agents),
            "prices": state.prices.tolist(),
            "n_trades": len(state.trades),
            "cumulative_travel_costs": dict(state.agent_travel_costs),
        }
        results.append(round_result)

        # Visualization (Phase 0–2 minimal)
        if ascii_renderer and ascii_interval and (state.round % ascii_interval == 0):
            frame = build_frame(state, config)
            ascii_renderer.render(frame)
        if gui_renderer:
            try:
                frame = build_frame(state, config)
                gui_renderer.render(frame)  # type: ignore
            except SystemExit:
                print("GUI window closed by user — terminating simulation loop early.")
                break

        # Structured per-agent logging
        if run_logger is not None and RoundLogRecord is not None:  # type: ignore
            marketplace_ids = {
                a.agent_id
                for a in state.agents
                if state.grid.is_in_marketplace(state.grid.get_position(a.agent_id))
            }

            # Aggregate executed net trades per agent and good for this round
            # Build matrix of zeros if no trades
            n_goods = len(state.prices)
            executed_net = {a.agent_id: [0.0] * n_goods for a in state.agents}
            for tr in state.trades:
                executed_net[tr.agent_id][tr.good_id] += tr.quantity

            unmet_demand = None
            unmet_supply = None
            if state.last_market_result is not None:
                unmet_demand = state.last_market_result.unmet_demand.tolist()
                unmet_supply = state.last_market_result.unmet_supply.tolist()

            records = []
            for agent in state.agents:
                pos = state.grid.get_position(agent.agent_id)
                # Compute simple utility snapshot (total endowment bundle)
                try:
                    util = agent.utility(agent.total_endowment)
                except Exception:
                    util = None
                records.append(
                    RoundLogRecord(
                        core_schema_version=SCHEMA_VERSION,
                        core_round=state.round,
                        core_agent_id=agent.agent_id,
                        spatial_pos_x=pos.x,
                        spatial_pos_y=pos.y,
                        spatial_in_marketplace=agent.agent_id in marketplace_ids,
                        econ_prices=state.prices.tolist(),
                        econ_executed_net=executed_net[agent.agent_id],
                        ration_unmet_demand=unmet_demand,
                        ration_unmet_supply=unmet_supply,
                        wealth_travel_cost=state.agent_travel_costs[agent.agent_id],
                        wealth_effective_budget=None,  # Placeholder (future: adjusted wealth at pricing)
                        financing_mode="PERSONAL",  # Default financing mode (schema-guarded)
                        utility=util,
                    )
                )
            run_logger.log_round(records)

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

    # Finalize structured logging
    if run_logger is not None:
        try:
            log_path = run_logger.finalize()
            simulation_results["structured_log_path"] = str(log_path)
            simulation_results["schema_version"] = SCHEMA_VERSION
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to finalize structured log: {e}")

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
    parser.add_argument("--no-gui", action="store_true", help="(Deprecated placeholder) Use --gui instead")
    parser.add_argument("--gui", action="store_true", help="Enable pygame visualization (if installed)")
    parser.add_argument("--tick-ms", type=int, default=120, help="GUI: delay between frames in milliseconds (default 120)")
    parser.add_argument("--ascii-viz-interval", type=int, help="Render ASCII grid every N rounds (headless visualization)")

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
        results = run_simulation(
            config_path,
            output_path,
            args.seed,
            ascii_interval=args.ascii_viz_interval,
            gui=args.gui,
            tick_ms=args.tick_ms,
        )

        # Print summary
        print(f"\nSimulation completed in {results['final_round']} rounds")
        print(
            f"Final marketplace participation: {results['agents_in_marketplace']}/{results['total_agents']}"
        )
        print(f"Final prices: {results['final_prices']}")
        if 'structured_log_path' in results:
            print(f"Structured log: {results['structured_log_path']} (schema {results.get('schema_version')})")

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
