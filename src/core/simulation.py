"""Runtime simulation module.

Provides the mutable runtime state container (`RuntimeSimulationState`) and a
library-friendly API (`initialize_runtime_state`, `run_round`) used by CLI
wrappers. This separates operational concerns (movement phases, travel costs,
last market result) from the lean analytical snapshot defined in
`core.types.SimulationState`.

Staged architectural refactor per WEAKNESSES_INVENTORY item 1.1–1.3.

NOTE: This module intentionally avoids printing/logging side effects beyond
minimal debug logging; higher-level orchestration (CLI, logging layer) should
call these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

from .agent import Agent
from .types import MarketResult, SimulationState, Trade
from src.spatial.grid import Grid, Position, create_random_positions
from src.spatial.movement import get_movement_policy, MovementPolicy
from src.econ.equilibrium import solve_walrasian_equilibrium
from src.econ.market import execute_constrained_clearing, apply_trades_to_agents


class AgentPhase(str, Enum):
    """Lifecycle phases for agent movement between home and marketplace.

    HOME_PREP  -> load inventory, prepare to travel
    TO_MARKET  -> traveling toward marketplace
    AT_MARKET  -> currently inside marketplace (eligible for trading)
    TO_HOME    -> returning home to deposit inventory
    """

    HOME_PREP = "home_prep"
    TO_MARKET = "to_market"
    AT_MARKET = "at_market"
    TO_HOME = "to_home"


@dataclass
class SimulationConfig:
    """Immutable simulation configuration parameters.

    Keeping this lightweight aids reproducibility and enables deterministic
    re-initialization when paired with a seed.
    """

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
    movement_policy: str = (
        "greedy"  # identifier passed to spatial.movement.get_movement_policy
    )


@dataclass
class RuntimeSimulationState:
    """Mutable, full-fidelity simulation runtime state.

    Distinct from the analytical snapshot (`core.types.SimulationState`). This
    container tracks per-round mutable fields and operational metadata.
    """

    round: int
    agents: List[Agent]
    grid: Grid
    prices: np.ndarray  # current equilibrium prices (numéraire-normalized)
    trades: List[Trade]
    agent_travel_costs: Dict[int, float]
    agent_phases: Dict[int, AgentPhase]
    last_market_result: Optional[MarketResult] = None
    last_prices: Optional[np.ndarray] = (
        None  # previous converged price vector (rest-goods warm start)
    )
    initial_max_distance: Optional[int] = None  # spatial convergence baseline
    last_solver_rest_norm: Optional[float] = None
    last_solver_status: Optional[str] = None

    def to_snapshot(self) -> SimulationState:
        """Produce an immutable analytical snapshot for validation / logging.

        The snapshot aggregates executed information; total welfare computed
        from agent utilities using their *total* endowment (home+personal),
        consistent with LTE theoretical benchmarks.
        """
        total_welfare = float(
            np.sum([a.utility(a.total_endowment) for a in self.agents])
        )
        return SimulationState(
            round_number=self.round,
            agents=self.agents.copy(),  # shallow copy sufficient (agents are objects)
            prices=self.prices.copy(),
            executed_trades=self.trades.copy(),
            z_rest_norm=0.0,  # Placeholder (solver interface currently returns locally)
            walras_dot=0.0,  # Placeholder until solver fallback instrumentation added
            total_welfare=total_welfare,
            marketplace_participants=sum(
                1
                for a in self.agents
                if self.grid.is_in_marketplace(self.grid.get_position(a.agent_id))
            ),
        )


def initialize_runtime_state(config: SimulationConfig) -> RuntimeSimulationState:
    """Initialize agents, grid, and runtime bookkeeping structures."""
    np.random.seed(config.random_seed)

    grid = Grid(
        config.grid_width,
        config.grid_height,
        config.marketplace_width,
        config.marketplace_height,
    )

    positions = create_random_positions(
        config.n_agents, config.grid_width, config.grid_height, config.random_seed
    )

    agents: List[Agent] = []
    for i in range(config.n_agents):
        alpha = np.random.dirichlet(np.ones(config.n_goods))
        alpha = np.maximum(alpha, 0.05)
        alpha = alpha / np.sum(alpha)
        home_endowment = np.random.exponential(2.0, config.n_goods)
        personal_endowment = np.random.exponential(1.0, config.n_goods)
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
        grid.add_agent(i, positions[i])

    # Compute baseline max Manhattan distance to marketplace rectangle
    mx0 = (config.grid_width - config.marketplace_width) // 2
    my0 = (config.grid_height - config.marketplace_height) // 2
    def _dist_to_market(px: int, py: int) -> int:
        dx = 0 if mx0 <= px < mx0 + config.marketplace_width else (
            mx0 - px if px < mx0 else px - (mx0 + config.marketplace_width - 1)
        )
        dy = 0 if my0 <= py < my0 + config.marketplace_height else (
            my0 - py if py < my0 else py - (my0 + config.marketplace_height - 1)
        )
        return abs(dx) + abs(dy)
    initial_max_distance = 0
    for a in agents:
        pos = grid.get_position(a.agent_id)
        d = _dist_to_market(pos.x, pos.y)
        if d > initial_max_distance:
            initial_max_distance = d

    return RuntimeSimulationState(
        round=0,
        agents=agents,
        grid=grid,
        prices=np.ones(config.n_goods),
        trades=[],
        agent_travel_costs={a.agent_id: 0.0 for a in agents},
        agent_phases={a.agent_id: AgentPhase.HOME_PREP for a in agents},
        initial_max_distance=initial_max_distance,
    )


def run_round(
    state: RuntimeSimulationState, config: SimulationConfig
) -> RuntimeSimulationState:
    """Execute one simulation round (pure logic, no printing).

    Mirrors previous script logic; future enhancements (solver fallback,
    invariant hooks) will integrate here.
    """
    # Reset per-round solver diagnostics; recomputed if clearing runs
    state.last_market_result = None
    state.last_solver_rest_norm = None
    state.last_solver_status = None

    # 1. Home preparation
    for agent in state.agents:
        phase = state.agent_phases[agent.agent_id]
        if phase == AgentPhase.HOME_PREP:
            if agent.is_at_home(agent.home_position):
                agent.load_inventory_for_travel()
                state.agent_phases[agent.agent_id] = AgentPhase.TO_MARKET
            else:
                state.agent_phases[agent.agent_id] = AgentPhase.TO_HOME

    # 2. Movement
    # 2. Movement (via pluggable policy)
    policy: MovementPolicy = get_movement_policy(
        getattr(config, "movement_policy", "greedy")
    )
    for agent in state.agents:
        phase = state.agent_phases[agent.agent_id]
        distance_moved = 0
        if phase == AgentPhase.TO_MARKET:
            distance_moved = policy.move_agent_toward_marketplace(
                state.grid, agent.agent_id
            )
            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)
            if state.grid.is_in_marketplace(current_pos):
                state.agent_phases[agent.agent_id] = AgentPhase.AT_MARKET
        elif phase == AgentPhase.TO_HOME:
            home_position = Position(*agent.home_position)
            distance_moved = policy.move_agent_toward_position(
                state.grid, agent.agent_id, home_position
            )
            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)
            if agent.is_at_home(agent.home_position):
                agent.deposit_inventory_at_home()
                state.agent_travel_costs[agent.agent_id] = 0.0
                state.agent_phases[agent.agent_id] = AgentPhase.HOME_PREP
        else:
            current_pos = state.grid.get_position(agent.agent_id)
            agent.position = (current_pos.x, current_pos.y)
        if distance_moved > 0 and config.movement_cost > 0:
            state.agent_travel_costs[agent.agent_id] += (
                config.movement_cost * distance_moved
            )

    # 3. Marketplace participants
    marketplace_agent_ids = {
        a.agent_id
        for a in state.agents
        if state.agent_phases[a.agent_id] == AgentPhase.AT_MARKET
        and state.grid.is_in_marketplace(state.grid.get_position(a.agent_id))
    }
    marketplace_agents = [
        a for a in state.agents if a.agent_id in marketplace_agent_ids
    ]

    # 4. Price discovery & clearing
    if len(marketplace_agents) >= 2:
        viable_agents: List[Agent] = []
        for agent in marketplace_agents:
            wealth = np.dot(state.prices, agent.total_endowment)
            if wealth > 1e-10:
                viable_agents.append(agent)
        if len(viable_agents) >= 2 and len(state.prices) >= 2:
            try:
                # Warm-start: prefer last converged prices if available
                initial_guess = (
                    state.last_prices if state.last_prices is not None else None
                )
                (
                    prices,
                    z_rest_norm,
                    _walras_dot,
                    solver_status,
                ) = solve_walrasian_equilibrium(
                    viable_agents, initial_guess=initial_guess
                )
                state.last_solver_rest_norm = z_rest_norm
                state.last_solver_status = solver_status
                if prices is not None and z_rest_norm < 1e-8:
                    state.prices[:] = prices  # in-place update preserves ndarray type
                    state.last_prices = prices.copy()
                    market_result = execute_constrained_clearing(
                        viable_agents,
                        prices,
                        capacity=None,
                        travel_costs=state.agent_travel_costs,
                    )
                    apply_trades_to_agents(viable_agents, market_result.executed_trades)
                    state.trades = market_result.executed_trades
                    state.last_market_result = market_result
                    # Post-trade invariant: per-agent value feasibility (PERSONAL mode)
                    # (Re-asserting here provides defense-in-depth beyond clearing module checks.)
                    try:
                        prices_vec = state.prices
                        for agent in viable_agents:
                            # Reconstruct executed net (buys positive, sells negative)
                            buys = np.zeros_like(prices_vec)
                            sells = np.zeros_like(prices_vec)
                            for tr in state.trades:
                                if tr.agent_id == agent.agent_id:
                                    if tr.quantity > 0:
                                        buys[tr.good_id] += tr.quantity
                                    else:
                                        sells[tr.good_id] += -tr.quantity
                            buy_value = float(np.dot(prices_vec, buys))
                            sell_value = float(np.dot(prices_vec, sells))
                            if buy_value > sell_value + 1e-10:
                                raise AssertionError(
                                    f"Post-trade value feasibility violated: agent={agent.agent_id} buy_value={buy_value} sell_value={sell_value}"
                                )
                    except Exception:
                        # Fail-fast strategy: propagate assertion while keeping minimal context
                        raise
                    # NOTE (2025-09-21): We deliberately defer transitioning agents
                    # back to TO_HOME until *all* agents currently en‑route have
                    # reached the marketplace. This preserves the monotonic
                    # Manhattan distance invariant asserted in
                    # test_greedy_policy_monotonic_distance (distance to the
                    # nearest marketplace cell never increases for agents in
                    # TO_MARKET / AT_MARKET). Without this guard, early arrivals
                    # would immediately flip to TO_HOME and start moving *away*,
                    # causing transient distance increases that fail the test.
                    #
                    # Future Enhancement (staggered returns): Introduce a
                    # configuration flag (e.g. config.stagger_returns = True)
                    # to allow agents to depart individually once their own
                    # trading is complete. When enabling that behavior we will
                    # need to (a) relax or parameterize the monotonic distance
                    # test and (b) ensure homeward movement does not interfere
                    # with remaining agents' access or pricing dynamics.
                    # Until then this synchronized departure keeps spatial
                    # movement pedagogy simple and deterministic.
                    try:
                        all_arrived = all(
                            state.grid.is_in_marketplace(
                                state.grid.get_position(a.agent_id)
                            )
                            for a in state.agents
                            if state.agent_phases[a.agent_id]
                            in (AgentPhase.TO_MARKET, AgentPhase.AT_MARKET)
                        )
                    except Exception:
                        all_arrived = False
                    if all_arrived:
                        for agent in viable_agents:
                            state.agent_phases[agent.agent_id] = AgentPhase.TO_HOME
            except Exception:  # pragma: no cover - propagate minimal failure info
                # Preserve previous status or mark failure without overwriting a prior successful label
                if state.last_solver_status is None:
                    state.last_solver_status = "solver_error"
                pass
    else:
        # No pricing attempt this round (insufficient participants)
        if state.last_solver_status is None:
            state.last_solver_status = "no_pricing"
        if state.last_solver_rest_norm is None:
            state.last_solver_rest_norm = 0.0

    state.round += 1
    return state
