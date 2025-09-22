#!/usr/bin/env python3
"""Simulation runner (clean restored version).

Orchestrates execution of spatial Walrasian rounds, optional GUI / ASCII
rendering, and structured per-agent logging (schema >=1.4.x) including
solver diagnostic fields.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Temporary path injection (until packaged console script is set up)
sys.path.append(str(Path(__file__).parent.parent))

from src.core.simulation import (
    SimulationConfig,
    RuntimeSimulationState,
    initialize_runtime_state,
    run_round,
    AgentPhase,  # re-export for legacy tests expecting AgentPhase here
)
from src.visualization import build_frame, ASCIIRenderer  # type: ignore
from src.visualization.playback import (
    LiveSimulationStream,
    PlaybackController,
    LogReplayStream,
)  # type: ignore

try:  # optional GUI
    from src.visualization import PygameRenderer  # type: ignore
except Exception:  # pragma: no cover
    PygameRenderer = None  # type: ignore

from src.core.config_validation import validate_simulation_config

try:  # structured logging (optional)
    from src.logging.run_logger import RunLogger, RoundLogRecord, SCHEMA_VERSION
except Exception:  # pragma: no cover
    RunLogger = None  # type: ignore
    RoundLogRecord = None  # type: ignore
    SCHEMA_VERSION = "0.0.0"  # type: ignore


def load_config(config_path: Path) -> SimulationConfig:
    import yaml

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    sim_cfg = data.get("simulation", {})
    agents_cfg = data.get("agents", {})
    econ_cfg = data.get("economy", {})
    cfg = SimulationConfig(
        name=sim_cfg.get("name", config_path.stem),
        n_agents=agents_cfg.get("count", 10),
        n_goods=econ_cfg.get("goods", 3),
        grid_width=econ_cfg.get("grid_size", [15, 15])[0],
        grid_height=econ_cfg.get("grid_size", [15, 15])[1],
        marketplace_width=econ_cfg.get("marketplace_size", [2, 2])[0],
        marketplace_height=econ_cfg.get("marketplace_size", [2, 2])[1],
        movement_cost=econ_cfg.get("movement_cost", 0.0),
        max_rounds=sim_cfg.get("max_rounds", 50),
        random_seed=sim_cfg.get("seed", 42),
        movement_policy=econ_cfg.get("movement_policy", "greedy"),
    )
    validate_simulation_config(cfg)
    return cfg


def initialize_simulation(config: SimulationConfig) -> RuntimeSimulationState:
    random.seed(config.random_seed)
    return initialize_runtime_state(config)


def run_simulation_round(
    state: RuntimeSimulationState, config: SimulationConfig
) -> RuntimeSimulationState:
    return run_round(state, config)

# Backward compatibility: some tests import AgentPhase from scripts.run_simulation
# even though it lives in src.core.simulation. We re-export by binding it here.
__all__ = [
    "SimulationConfig",
    "RuntimeSimulationState",
    "initialize_simulation",
    "run_simulation_round",
    "run_simulation",
    "AgentPhase",
]


def run_simulation(
    config_path: Path,
    output_path: Optional[Path] = None,
    seed_override: Optional[int] = None,
    ascii_interval: Optional[int] = None,
    gui: bool = False,
    tick_ms: int = 120,
    rps: float = 2.0,
    snapshot_dir: Optional[Path] = None,
    snapshot_every: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute full simulation returning summary dictionary."""

    config = load_config(config_path)
    if seed_override is not None:
        config.random_seed = seed_override

    print(f"Starting simulation: {config.name}")
    print(
        f"Grid {config.grid_width}x{config.grid_height}  Marketplace {config.marketplace_width}x{config.marketplace_height}  Agents={config.n_agents} Goods={config.n_goods}"
    )
    print(
        f"Movement cost={config.movement_cost}  Max rounds={config.max_rounds}  Seed={config.random_seed}"
    )

    state = initialize_simulation(config)

    run_logger = None
    geometry_sidecar_path = None
    if output_path and RunLogger is not None:  # type: ignore
        run_name = f"{config.name}_seed{config.random_seed}"
        run_logger = RunLogger(output_path, run_name)  # type: ignore
        try:  # geometry sidecar
            from src.logging.geometry import GeometrySpec, write_geometry_sidecar  # type: ignore

            gx_min = (config.grid_width - config.marketplace_width) // 2
            gy_min = (config.grid_height - config.marketplace_height) // 2
            geom = GeometrySpec(
                run_name=run_name,
                grid_width=config.grid_width,
                grid_height=config.grid_height,
                market_x_min=gx_min,
                market_x_max=gx_min + config.marketplace_width - 1,
                market_y_min=gy_min,
                market_y_max=gy_min + config.marketplace_height - 1,
                movement_policy=config.movement_policy,
                random_seed=config.random_seed,
            )
            geometry_sidecar_path = write_geometry_sidecar(output_path, geom)
        except Exception:  # pragma: no cover
            geometry_sidecar_path = None

    results: List[Dict[str, Any]] = []
    ascii_renderer = ASCIIRenderer() if ascii_interval else None
    gui_renderer = None
    if gui and PygameRenderer is not None:  # type: ignore
        try:
            gui_renderer = PygameRenderer(
                tick_ms=tick_ms, title=f"{config.name} (seed {config.random_seed})"
            )
        except Exception as e:  # pragma: no cover
            print(f"Warning: GUI init failed: {e}")
            gui_renderer = None

    def _build_round_log_records(
        state: RuntimeSimulationState,
    ) -> Optional[List["RoundLogRecord"]]:  # type: ignore[name-defined, return-type]  # noqa: C901
        if RunLogger is None or RoundLogRecord is None:  # type: ignore
            return None
        try:
            n_goods = len(state.prices)
            marketplace_ids = {
                a.agent_id
                for a in state.agents
                if state.grid.is_in_marketplace(state.grid.get_position(a.agent_id))
            }

            # --- Executed trade aggregation ---
            executed_net: Dict[int, List[float]] = {a.agent_id: [0.0] * n_goods for a in state.agents}
            executed_buys: Dict[int, List[float]] = {a.agent_id: [0.0] * n_goods for a in state.agents}
            executed_sells: Dict[int, List[float]] = {a.agent_id: [0.0] * n_goods for a in state.agents}
            for tr in state.trades:
                if tr.quantity > 0:
                    executed_buys[tr.agent_id][tr.good_id] += tr.quantity
                elif tr.quantity < 0:
                    executed_sells[tr.agent_id][tr.good_id] += -tr.quantity
                executed_net[tr.agent_id][tr.good_id] += tr.quantity

            # --- Per-agent rationing diagnostics placeholders ---
            per_agent_requested_buys: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}
            per_agent_requested_sells: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}
            per_agent_unmet_buys: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}
            per_agent_unmet_sells: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}
            # fill rate arrays
            per_agent_fill_buy: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}
            per_agent_fill_sell: Dict[int, Optional[List[float]]] = {a.agent_id: None for a in state.agents}

            unmet_demand = None
            unmet_supply = None
            if state.last_market_result is not None:
                mr = state.last_market_result
                unmet_demand = mr.unmet_demand.tolist()
                unmet_supply = mr.unmet_supply.tolist()
                diag = mr.rationing_diagnostics
                if diag is not None:
                    for aid, arr in diag.agent_unmet_buys.items():
                        per_agent_unmet_buys[aid] = arr.tolist()
                    for aid, arr in diag.agent_unmet_sells.items():
                        per_agent_unmet_sells[aid] = arr.tolist()
                    for aid, arr in diag.agent_fill_rates_buy.items():
                        per_agent_fill_buy[aid] = arr.tolist()
                    for aid, arr in diag.agent_fill_rates_sell.items():
                        per_agent_fill_sell[aid] = arr.tolist()
                # Reconstruct requested quantities (executed + unmet)
                for a in state.agents:
                    ub = per_agent_unmet_buys[a.agent_id]
                    us = per_agent_unmet_sells[a.agent_id]
                    if ub is not None and us is not None:
                        per_agent_requested_buys[a.agent_id] = [
                            executed_buys[a.agent_id][g] + ub[g] for g in range(n_goods)
                        ]
                        per_agent_requested_sells[a.agent_id] = [
                            executed_sells[a.agent_id][g] + us[g] for g in range(n_goods)
                        ]

            # --- Spatial distances ---
            try:
                from src.logging.geometry import manhattan_distance_to_market  # type: ignore

                gx_min = state.grid.marketplace_x
                gx_max = gx_min + state.grid.marketplace_width - 1
                gy_min = state.grid.marketplace_y
                gy_max = gy_min + state.grid.marketplace_height - 1
                per_agent_distance: Dict[int, int] = {}
                max_dist = 0
                total_dist = 0
                for a in state.agents:
                    pos_a = state.grid.get_position(a.agent_id)
                    d_val = manhattan_distance_to_market(pos_a.x, pos_a.y, gx_min, gx_max, gy_min, gy_max)
                    per_agent_distance[a.agent_id] = d_val
                    max_dist = max(max_dist, d_val)
                    total_dist += d_val
                avg_dist = total_dist / len(state.agents) if state.agents else 0.0
            except Exception:  # pragma: no cover
                per_agent_distance = {a.agent_id: None for a in state.agents}  # type: ignore
                max_dist = None  # type: ignore
                avg_dist = None  # type: ignore

            # --- Solver metrics (single snapshot, duplicated per row) ---
            solver_metrics: Dict[str, Any] = {}
            try:
                from src.econ.equilibrium import get_last_solver_metrics  # type: ignore

                solver_metrics = get_last_solver_metrics() or {}
            except Exception:
                solver_metrics = {}

            records: List["RoundLogRecord"] = []  # type: ignore[name-defined]
            for a in state.agents:
                pos = state.grid.get_position(a.agent_id)
                try:
                    util_val = a.utility(a.total_endowment)
                except Exception:
                    util_val = None
                try:
                    total_val = float(np.dot(state.prices, a.total_endowment)) if len(state.prices) else None
                except Exception:
                    total_val = None
                travel_cost = state.agent_travel_costs.get(a.agent_id, 0.0)
                travel_adjusted = None if total_val is None else max(0.0, total_val - travel_cost)
                financing_mode = getattr(state, "financing_mode", "PERSONAL")
                effective_budget = travel_adjusted if financing_mode == "TOTAL_WEALTH" else None
                rec = RoundLogRecord(
                    core_schema_version=SCHEMA_VERSION,
                    core_round=state.round,
                    core_agent_id=a.agent_id,
                    spatial_pos_x=pos.x,
                    spatial_pos_y=pos.y,
                    spatial_in_marketplace=a.agent_id in marketplace_ids,
                    spatial_distance_to_market=per_agent_distance.get(a.agent_id),
                    spatial_max_distance_round=max_dist,
                    spatial_avg_distance_round=avg_dist,
                    spatial_initial_max_distance=state.initial_max_distance,
                    econ_prices=state.prices.tolist(),
                    econ_executed_net=executed_net[a.agent_id],
                    econ_requested_buys=per_agent_requested_buys[a.agent_id],
                    econ_requested_sells=per_agent_requested_sells[a.agent_id],
                    econ_executed_buys=executed_buys[a.agent_id],
                    econ_executed_sells=executed_sells[a.agent_id],
                    econ_unmet_buys=per_agent_unmet_buys[a.agent_id],
                    econ_unmet_sells=per_agent_unmet_sells[a.agent_id],
                    econ_fill_rate_buys=per_agent_fill_buy[a.agent_id],
                    econ_fill_rate_sells=per_agent_fill_sell[a.agent_id],
                    ration_unmet_demand=unmet_demand,
                    ration_unmet_supply=unmet_supply,
                    wealth_travel_cost=travel_cost,
                    wealth_effective_budget=effective_budget,
                    wealth_total_endowment_value=total_val,
                    wealth_travel_adjusted_total=travel_adjusted,
                    financing_mode=financing_mode,
                    utility=util_val,
                    solver_status=state.last_solver_status,
                    solver_rest_norm=state.last_solver_rest_norm,
                    solver_walras_dot=solver_metrics.get("walras_dot"),
                    solver_total_time=solver_metrics.get("total_time"),
                    solver_fsolve_time=solver_metrics.get("fsolve_time"),
                    solver_tatonnement_time=solver_metrics.get("tatonnement_time"),
                    solver_tatonnement_iterations=solver_metrics.get("tatonnement_iterations"),
                    solver_fallback_used=solver_metrics.get("fallback_used"),
                    solver_method=solver_metrics.get("method"),
                )
                records.append(rec)  # type: ignore[arg-type]
            return records
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to build log records (round {state.round}): {e}")
            return None

    # Initialize snapshotter if requested
    snapshotter = None
    if snapshot_dir is not None:
        from src.visualization.snapshot import (
            Snapshotter,
        )  # local import to avoid overhead when unused

        snapshotter = Snapshotter(snapshot_dir, prefix=config.name.replace(" ", "_"))

    if gui_renderer:
        # GUI with playback controller
        stream = LiveSimulationStream(state=state, config=config)
        # Use user-specified rounds-per-second (rps) value; fallback to legacy 5.0 only if rps <= 0
        live_rps = rps if rps > 0 else 5.0
        controller = PlaybackController(stream=stream, rounds_per_second=live_rps)
        import pygame  # type: ignore

        running = True
        while running:
            manual_frame = None
            for event in pygame.event.get():  # type: ignore
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_SPACE:
                        controller.toggle_play()
                    elif event.key == pygame.K_RIGHT:
                        if mods & pygame.KMOD_SHIFT:
                            manual_frame = controller.jump(10)
                        else:
                            manual_frame = controller.step_once()
                    elif event.key == pygame.K_LEFT:
                        if mods & pygame.KMOD_SHIFT:
                            manual_frame = controller.jump(-10)
                        else:
                            manual_frame = controller.step_back()
                    elif event.key == pygame.K_HOME:
                        manual_frame = controller.goto(1)
                    elif event.key == pygame.K_END:
                        manual_frame = controller.goto(10**9)
                    elif event.key == pygame.K_UP:
                        controller.speed_up()
                    elif event.key == pygame.K_DOWN:
                        controller.slow_down()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            frame = manual_frame if manual_frame is not None else controller.update()
            if frame is not None:
                state = stream.state  # sync external reference
                try:
                    gui_renderer.render(
                        frame,
                        is_playing=controller.is_playing,
                        speed_rps=controller.rounds_per_second,
                    )  # type: ignore
                except SystemExit:
                    running = False
                if (
                    snapshotter
                    and snapshot_every
                    and (frame.round % snapshot_every == 0)
                ):
                    try:
                        snapshotter.save(
                            frame,
                            surface=gui_renderer.surface
                            if hasattr(gui_renderer, "surface")
                            else None,
                        )  # type: ignore[arg-type]
                    except Exception:
                        pass
                if run_logger is not None:
                    records = _build_round_log_records(state)
                    if records:
                        run_logger.log_round(records)  # type: ignore[arg-type]

            if stream.state.round >= config.max_rounds:
                controller.is_playing = False
        state = stream.state
    else:
        for _ in range(config.max_rounds):
            # --- Advance one round ---
            state = run_simulation_round(state, config)

            # --- Snapshots (optional) ---
            if snapshotter and snapshot_every and (state.round % snapshot_every == 0):
                frame = build_frame(state, config)
                try:
                    snapshotter.save(frame)
                except Exception:
                    pass

            # --- Per-round summary aggregation ---
            round_result = {
                "round": state.round,
                "n_marketplace_agents": len(state.grid.get_agents_in_marketplace()),
                "total_agents": len(state.agents),
                "prices": state.prices.tolist(),
                "n_trades": len(state.trades),
                "cumulative_travel_costs": dict(state.agent_travel_costs),
            }
            results.append(round_result)

            # --- ASCII visualization (headless HUD) ---
            if ascii_renderer and ascii_interval and (state.round % ascii_interval == 0):
                try:
                    frame = build_frame(state, config)
                    ascii_renderer.render(frame)
                except Exception:
                    pass

            # --- Structured per-agent logging ---
            if run_logger is not None:
                try:
                    records = _build_round_log_records(state)
                    if records:
                        run_logger.log_round(records)  # type: ignore[arg-type]
                except Exception as e:  # pragma: no cover - defensive
                    print(f"Warning: failed to log round {state.round}: {e}")

            # --- Progress output ---
            if state.round % 10 == 0:
                n_market = len(state.grid.get_agents_in_marketplace())
                print(
                    f"Round {state.round}: {n_market}/{len(state.agents)} agents in marketplace"
                )

            # --- Early termination: all agents in marketplace ---
            if len(state.grid.get_agents_in_marketplace()) == len(state.agents):
                print(f"All agents reached marketplace at round {state.round}")
                break

    # Compile final results
    simulation_results: Dict[str, Any] = {
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
            # Attach integrity digest if exists
            integrity_candidate = (
                log_path.parent
                / f"{log_path.stem.rsplit('_round_log', 1)[0]}_integrity.json"
            )
            if integrity_candidate.exists():
                simulation_results["integrity_digest_path"] = str(integrity_candidate)
            # Augment metadata JSON with geometry file reference if present
            if geometry_sidecar_path is not None and geometry_sidecar_path.exists():
                simulation_results["geometry_file"] = str(geometry_sidecar_path)
                try:
                    meta_path = (
                        log_path.parent
                        / f"{log_path.stem.rsplit('_round_log', 1)[0]}_metadata.json"
                    )
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        meta["geometry_file"] = geometry_sidecar_path.name
                        meta_path.write_text(json.dumps(meta, indent=2))
                except Exception:
                    pass
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
    parser = argparse.ArgumentParser(
        description="Run economic simulation or replay a prior run log"
    )
    parser.add_argument(
        "--config", help="Configuration YAML file (omit when using --replay)"
    )
    parser.add_argument(
        "--replay",
        help="Path to structured log (.jsonl or .parquet) to replay instead of running new simulation",
    )
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="(Deprecated placeholder) Use --gui instead",
    )
    parser.add_argument(
        "--gui", action="store_true", help="Enable pygame visualization (if installed)"
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=120,
        help="GUI: delay between frames in milliseconds (default 120)",
    )
    parser.add_argument(
        "--ascii-viz-interval",
        type=int,
        help="Render ASCII grid every N rounds (headless visualization)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=2.0,
        help="Target rounds per second for GUI playback (2.0 => 0.5s per frame)",
    )
    parser.add_argument(
        "--snapshot-dir",
        help="Directory to write frame snapshots (JSON + optional PNG)",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        help="Capture snapshot every N rounds (requires --snapshot-dir)",
    )

    args = parser.parse_args()

    # Mutual exclusivity check
    if (args.config is None) == (args.replay is None):
        print("Error: provide exactly one of --config or --replay")
        return 1

    # Replay mode short-circuit
    if args.replay is not None:
        log_path = Path(args.replay)
        if not log_path.exists():
            print(f"Error: replay log not found: {log_path}")
            return 1
        if args.gui and PygameRenderer is None:  # type: ignore
            print(
                "Warning: GUI requested but pygame not available; falling back to headless replay."
            )
            args.gui = False
        stream = LogReplayStream(log_path=log_path)
        controller = PlaybackController(stream=stream, rounds_per_second=args.rps)
        if args.gui and PygameRenderer is not None:  # GUI replay
            try:
                gui_renderer = PygameRenderer(
                    tick_ms=args.tick_ms, title=f"Replay: {log_path.name}"
                )  # type: ignore
            except Exception as e:  # pragma: no cover
                print(f"Warning: failed to initialize GUI renderer: {e}")
                gui_renderer = None  # type: ignore
            import pygame  # type: ignore

            running = True
            while running:
                manual_frame = None
                for event in pygame.event.get():  # type: ignore
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        mods = pygame.key.get_mods()
                        if event.key == pygame.K_SPACE:
                            controller.toggle_play()
                        elif event.key == pygame.K_RIGHT:
                            manual_frame = (
                                controller.jump(10)
                                if mods & pygame.KMOD_SHIFT
                                else controller.step_once()
                            )
                        elif event.key == pygame.K_LEFT:
                            manual_frame = (
                                controller.jump(-10)
                                if mods & pygame.KMOD_SHIFT
                                else controller.step_back()
                            )
                        elif event.key == pygame.K_HOME:
                            manual_frame = controller.goto(1)
                        elif event.key == pygame.K_END:
                            manual_frame = controller.goto(10**9)
                        elif event.key == pygame.K_UP:
                            controller.speed_up()
                        elif event.key == pygame.K_DOWN:
                            controller.slow_down()
                        elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False

                frame = manual_frame if manual_frame is not None else controller.update()
                if frame is not None and gui_renderer is not None:
                    try:
                        gui_renderer.render(
                            frame,
                            is_playing=controller.is_playing,
                            speed_rps=controller.rounds_per_second,
                        )  # type: ignore
                    except SystemExit:
                        running = False
                if frame is None and not controller.is_playing:
                    running = False
            print("Replay complete.")
            return 0
        else:  # Headless replay
            # NOTE: PlaybackController.update() is time-gated and returns None
            # until frame_interval elapses, which caused the original headless
            # replay loop to exit immediately (rendered 0 frames). For headless
            # deterministic replays we ignore timing and pull frames directly
            # from the underlying stream.
            frames_rendered = 0
            while True:
                frame = stream.next_frame()
                if frame is None:
                    break
                frames_rendered += 1
                if frames_rendered % 10 == 0:
                    print(
                        f"Replay frame {frame.round}  prices={frame.prices}  participants={frame.participation_count}/{frame.total_agents}"
                    )
            print(f"Replay complete. Frames: {frames_rendered}")
            return 0

    # Validate configuration file (simulation mode only)
    config_path = Path(args.config)  # type: ignore[arg-type]
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
            rps=args.rps,
            snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
            snapshot_every=args.snapshot_every,
        )

        # Print summary
        print(f"\nSimulation completed in {results['final_round']} rounds")
        print(
            f"Final marketplace participation: {results['agents_in_marketplace']}/{results['total_agents']}"
        )
        print(f"Final prices: {results['final_prices']}")
        if "structured_log_path" in results:
            print(
                f"Structured log: {results['structured_log_path']} (schema {results.get('schema_version')})"
            )

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
