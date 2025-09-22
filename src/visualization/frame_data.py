from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Any, Optional, Dict


if TYPE_CHECKING:  # Avoid runtime import cycles
    from src.core.simulation import (
        RuntimeSimulationState,
        SimulationConfig,
    )  # pragma: no cover


@dataclass(frozen=True)
class AgentFrame:
    agent_id: int
    x: int
    y: int
    in_marketplace: bool
    distance_to_market: int


@dataclass(frozen=True)
class FrameData:
    round: int
    grid_width: int
    grid_height: int
    market_x0: int
    market_y0: int
    market_width: int
    market_height: int
    agents: List[AgentFrame]
    prices: List[float]
    participation_count: int
    total_agents: int

    # Enriched economic overlays (optional depending on market activity):
    avg_buy_fill_rate: Optional[float] = None
    avg_sell_fill_rate: Optional[float] = None
    unmet_buy_share: Optional[float] = (
        None  # total unmet buys / (unmet + executed buys)
    )
    solver_rest_goods_norm: Optional[float] = None
    solver_status: Optional[str] = None
    total_travel_cost: Optional[float] = None
    max_distance_to_market: Optional[int] = None
    # Newly added aggregate spatial metric (replay + live):
    avg_distance_to_market: Optional[float] = None
    # Per-agent quick lookup dictionaries (agent_id -> metric lists)
    agent_requested_buys: Optional[Dict[int, List[float]]] = None
    agent_executed_buys: Optional[Dict[int, List[float]]] = None
    agent_unmet_buys: Optional[Dict[int, List[float]]] = None
    agent_fill_rates_buy: Optional[Dict[int, List[float]]] = None
    agent_fill_rates_sell: Optional[Dict[int, List[float]]] = None
    clearing_efficiency: Optional[float] = None
    unmet_sell_share: Optional[float] = None
    # HUD overlay fields (computed live or reconstructed in replay):
    convergence_index: Optional[float] = None  # avg_distance / initial_max_distance
    hud_round_digest: Optional[str] = None  # short hash/digest snippet for frame integrity


def _manhattan_distance_to_rect(
    x: int, y: int, rx: int, ry: int, rw: int, rh: int
) -> int:
    """Compute Manhattan distance from point (x,y) to axis-aligned rectangle.

    Distance is 0 if point lies inside the rectangle.
    """
    dx = 0 if rx <= x < rx + rw else (rx - x if x < rx else x - (rx + rw - 1))
    dy = 0 if ry <= y < ry + rh else (ry - y if y < ry else y - (ry + rh - 1))
    return abs(dx) + abs(dy)


def build_frame(
    state: "RuntimeSimulationState", config: "SimulationConfig"
) -> FrameData:
    """Extract an immutable visualization snapshot from runtime state & config.

    The function is deliberately light on dependencies to avoid tight coupling.
    """
    # Marketplace rectangle (centered) replicates logic from simulation initialization
    mx0 = (config.grid_width - config.marketplace_width) // 2
    my0 = (config.grid_height - config.marketplace_height) // 2

    agents: List[AgentFrame] = []
    grid = state.grid
    total_agents = len(state.agents)
    participation = 0
    max_distance = 0
    distance_sum = 0
    for a in state.agents:  # each is an Agent-like object (duck-typed)
        a_obj: Any = a  # explicit Any cast for type checker
        pos = grid.get_position(a_obj.agent_id)
        in_market = grid.is_in_marketplace(pos)
        if in_market:
            participation += 1
        dist = _manhattan_distance_to_rect(
            pos.x, pos.y, mx0, my0, config.marketplace_width, config.marketplace_height
        )
        if dist > max_distance:
            max_distance = dist
        distance_sum += dist
        agents.append(
            AgentFrame(
                agent_id=a_obj.agent_id,
                x=pos.x,
                y=pos.y,
                in_marketplace=in_market,
                distance_to_market=dist,
            )
        )

    # Enriched diagnostics from last_market_result (if available)
    avg_buy_fill = None
    avg_sell_fill = None
    unmet_buy_share = None
    solver_norm = state.last_solver_rest_norm
    solver_status = state.last_solver_status
    requested_buys: Optional[Dict[int, List[float]]] = None
    executed_buys: Optional[Dict[int, List[float]]] = None
    unmet_buys: Optional[Dict[int, List[float]]] = None
    fill_rates_buy: Optional[Dict[int, List[float]]] = None
    fill_rates_sell: Optional[Dict[int, List[float]]] = None
    clearing_efficiency: Optional[float] = None
    unmet_sell_share: Optional[float] = None

    mr = state.last_market_result
    if mr is not None and mr.rationing_diagnostics is not None:
        diag = mr.rationing_diagnostics
        from typing import List as _ListFloat

        fill_rates: _ListFloat[float] = []
        sell_fill_rates: _ListFloat[float] = []
        requested_buys = {}
        executed_buys = {}
        unmet_buys = {}
        fill_rates_buy = {}
        fill_rates_sell = {}
        # Per-agent unmet buys
        for agent_id, unmet_arr in diag.agent_unmet_buys.items():
            unmet_list = [float(x) for x in unmet_arr]
            unmet_buys[agent_id] = unmet_list
        # Fill rates
        for agent_id, fr in diag.agent_fill_rates_buy.items():
            fr_list = [float(x) for x in fr]
            fill_rates_buy[agent_id] = fr_list
            if fr_list:
                fill_rates.append(sum(fr_list) / len(fr_list))
        for agent_id, fr in diag.agent_fill_rates_sell.items():
            fr_list = [float(x) for x in fr]
            fill_rates_sell[agent_id] = fr_list
            if fr_list:
                sell_fill_rates.append(sum(fr_list) / len(fr_list))
        if fill_rates:
            avg_buy_fill = float(sum(fill_rates) / len(fill_rates))
        if sell_fill_rates:
            avg_sell_fill = float(sum(sell_fill_rates) / len(sell_fill_rates))
        # Unmet buy share
        total_unmet = float(sum(mr.unmet_demand))
        total_executed_buy_volume = float(sum(mr.total_volume))
        denom = total_unmet + total_executed_buy_volume
        if denom > 0:
            unmet_buy_share = total_unmet / denom
        total_unmet_supply = float(sum(mr.unmet_supply))
        denom_supply = total_unmet_supply + total_executed_buy_volume
        if denom_supply > 0:
            unmet_sell_share = total_unmet_supply / denom_supply
        try:
            clearing_efficiency = float(mr.clearing_efficiency)
        except Exception:
            clearing_efficiency = None
    # Travel cost aggregate
    total_travel_cost = (
        float(sum(state.agent_travel_costs.values()))
        if state.agent_travel_costs
        else 0.0
    )

    # Potential solver norm placeholder (if later stored on state; currently None)
    # solver_norm remains None until integrated into RuntimeSimulationState.

    # Lazy import to avoid circular dependency (metrics imports FrameData)
    from . import metrics as _metrics  # type: ignore

    # HUD enrichment values (computed prior to frozen dataclass construction)
    convergence_index: Optional[float] = None
    if state.initial_max_distance is not None and total_agents > 0:
        try:
            convergence_index = _metrics.spatial_convergence_index(
                (distance_sum / total_agents) if total_agents > 0 else 0.0,
                state.initial_max_distance,
            )
        except Exception:
            convergence_index = None
    # Temporary skeleton for digest (populate using minimal dict first then finalize after instantiation if needed)
    # We'll compute digest on a provisional dict replicating hashing fields.
    hud_round_digest: Optional[str] = None
    try:
        # Minimal transient object mimic for hashing fields
        _temp_fd = FrameData(  # type: ignore[misc]
            round=state.round,
            grid_width=config.grid_width,
            grid_height=config.grid_height,
            market_x0=mx0,
            market_y0=my0,
            market_width=config.marketplace_width,
            market_height=config.marketplace_height,
            agents=agents,
            prices=[float(p) for p in state.prices],
            participation_count=participation,
            total_agents=total_agents,
            avg_buy_fill_rate=avg_buy_fill,
            avg_sell_fill_rate=avg_sell_fill,
            unmet_buy_share=unmet_buy_share,
            solver_rest_goods_norm=solver_norm,
            solver_status=solver_status,
            total_travel_cost=total_travel_cost,
            max_distance_to_market=max_distance,
            avg_distance_to_market=(
                (distance_sum / total_agents) if total_agents > 0 else 0.0
            ),
            agent_requested_buys=requested_buys,
            agent_executed_buys=executed_buys,
            agent_unmet_buys=unmet_buys,
            agent_fill_rates_buy=fill_rates_buy,
            agent_fill_rates_sell=fill_rates_sell,
            clearing_efficiency=clearing_efficiency,
            unmet_sell_share=unmet_sell_share,
            convergence_index=convergence_index,
            hud_round_digest=None,
        )
        hud_round_digest = _metrics.frame_hash(_temp_fd)[0:12]
    except Exception:
        hud_round_digest = None

    return FrameData(
        round=state.round,
        grid_width=config.grid_width,
        grid_height=config.grid_height,
        market_x0=mx0,
        market_y0=my0,
        market_width=config.marketplace_width,
        market_height=config.marketplace_height,
        agents=agents,
        prices=[float(p) for p in state.prices],
        participation_count=participation,
        total_agents=total_agents,
        avg_buy_fill_rate=avg_buy_fill,
        avg_sell_fill_rate=avg_sell_fill,
        unmet_buy_share=unmet_buy_share,
        solver_rest_goods_norm=solver_norm,
        solver_status=solver_status,
        total_travel_cost=total_travel_cost,
        max_distance_to_market=max_distance,
        avg_distance_to_market=(
            (distance_sum / total_agents) if total_agents > 0 else 0.0
        ),
        agent_requested_buys=requested_buys,
        agent_executed_buys=executed_buys,
        agent_unmet_buys=unmet_buys,
        agent_fill_rates_buy=fill_rates_buy,
        agent_fill_rates_sell=fill_rates_sell,
        clearing_efficiency=clearing_efficiency,
        unmet_sell_share=unmet_sell_share,
        convergence_index=convergence_index,
        hud_round_digest=hud_round_digest,
    )
