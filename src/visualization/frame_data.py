from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Any

if TYPE_CHECKING:  # Avoid runtime import cycles
    from src.core.simulation import RuntimeSimulationState, SimulationConfig  # pragma: no cover


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


def _manhattan_distance_to_rect(x: int, y: int, rx: int, ry: int, rw: int, rh: int) -> int:
    """Compute Manhattan distance from point (x,y) to axis-aligned rectangle.

    Distance is 0 if point lies inside the rectangle.
    """
    dx = 0 if rx <= x < rx + rw else (rx - x if x < rx else x - (rx + rw - 1))
    dy = 0 if ry <= y < ry + rh else (ry - y if y < ry else y - (ry + rh - 1))
    return abs(dx) + abs(dy)


def build_frame(state: 'RuntimeSimulationState', config: 'SimulationConfig') -> FrameData:
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
    for a in state.agents:  # each is an Agent-like object (duck-typed)
        a_obj: Any = a  # explicit Any cast for type checker
        pos = grid.get_position(a_obj.agent_id)
        in_market = grid.is_in_marketplace(pos)
        if in_market:
            participation += 1
        dist = _manhattan_distance_to_rect(pos.x, pos.y, mx0, my0, config.marketplace_width, config.marketplace_height)
        agents.append(AgentFrame(
            agent_id=a_obj.agent_id,
            x=pos.x,
            y=pos.y,
            in_marketplace=in_market,
            distance_to_market=dist,
        ))

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
    )
