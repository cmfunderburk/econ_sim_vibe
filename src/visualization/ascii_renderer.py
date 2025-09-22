from __future__ import annotations

from typing import List, TextIO, Optional
from .frame_data import FrameData
from .base_renderer import BaseRenderer


class ASCIIRenderer(BaseRenderer):
    """Minimal ASCII grid renderer.

    Agents: digit (agent_id % 10). Marketplace cells: '#'. Empty: '.'.
    """

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        import sys

        self.stream: TextIO = stream if stream is not None else sys.stdout

    def render(self, frame: FrameData) -> None:  # pragma: no cover - simple IO
        # Initialize grid
        grid: List[List[str]] = [
            ["." for _ in range(frame.grid_width)] for _ in range(frame.grid_height)
        ]

        # Marketplace rectangle
        for y in range(frame.market_y0, frame.market_y0 + frame.market_height):
            for x in range(frame.market_x0, frame.market_x0 + frame.market_width):
                grid[y][x] = "#"

        # Agents (overwrite marketplace marker with agent id char for visibility)
        for ag in frame.agents:
            ch = str(ag.agent_id % 10)
            grid[ag.y][ag.x] = ch

        # Compose lines
        ci_part = (
            f" CI={frame.convergence_index:.2f}" if frame.convergence_index is not None else ""
        )
        dist_bits: list[str] = []
        if frame.max_distance_to_market is not None:
            dist_bits.append(f"maxD={frame.max_distance_to_market}")
        if frame.avg_distance_to_market is not None:
            dist_bits.append(f"avgD={frame.avg_distance_to_market:.2f}")
        if frame.solver_rest_goods_norm is not None:
            dist_bits.append(f"resid={frame.solver_rest_goods_norm:.2e}")
        if frame.clearing_efficiency is not None:
            dist_bits.append(f"eff={frame.clearing_efficiency:.2f}")
        if frame.unmet_buy_share is not None:
            dist_bits.append(f"unmetB={frame.unmet_buy_share:.2f}")
        if frame.unmet_sell_share is not None:
            dist_bits.append(f"unmetS={frame.unmet_sell_share:.2f}")
        if frame.solver_status:
            dist_bits.append(f"status={frame.solver_status}")
        if frame.hud_round_digest is not None:
            dist_bits.append(f"dgst={frame.hud_round_digest}")
        dist_part = (" ".join(dist_bits)) if dist_bits else ""
        lines = [
            "Round {}{} participation {}/{} prices: {} {}".format(
                frame.round,
                ci_part,
                frame.participation_count,
                frame.total_agents,
                frame.prices,
                dist_part,
            )
        ]
        lines.extend("".join(r) for r in grid)
        print("\n".join(lines), file=self.stream)
