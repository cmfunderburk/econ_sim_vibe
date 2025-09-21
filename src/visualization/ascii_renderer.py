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
        grid: List[List[str]] = [ ["." for _ in range(frame.grid_width)] for _ in range(frame.grid_height) ]

        # Marketplace rectangle
        for y in range(frame.market_y0, frame.market_y0 + frame.market_height):
            for x in range(frame.market_x0, frame.market_x0 + frame.market_width):
                grid[y][x] = "#"

        # Agents (overwrite marketplace marker with agent id char for visibility)
        for ag in frame.agents:
            ch = str(ag.agent_id % 10)
            grid[ag.y][ag.x] = ch

        # Compose lines
        lines = ["Round {}  participation {}/{}  prices: {}".format(
            frame.round, frame.participation_count, frame.total_agents, frame.prices
        )]
        lines.extend(''.join(r) for r in grid)
        print('\n'.join(lines), file=self.stream)
