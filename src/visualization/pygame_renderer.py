from __future__ import annotations

try:  # pragma: no cover - runtime optional
    import pygame
except Exception as e:  # pragma: no cover
    raise RuntimeError("pygame not available: {}".format(e))

from .frame_data import FrameData, AgentFrame
from .base_renderer import BaseRenderer


class PygameRenderer(BaseRenderer):  # pragma: no cover - UI difficult to test automatically
    def __init__(self, cell_size: int = 26, tick_ms: int = 120, title: str = "Economic Simulation") -> None:
        pygame.init()
        self.cell = cell_size
        self.tick_ms = tick_ms
        self.title = title
        self.surface = None
        self.font = None

    def _ensure_surface(self, frame: FrameData) -> None:
        if self.surface is None:
            w = frame.grid_width * self.cell
            h = frame.grid_height * self.cell + 40  # HUD area
            self.surface = pygame.display.set_mode((w, h))
            pygame.display.set_caption(self.title)
            self.font = pygame.font.SysFont(None, 16)

    def _draw_agent(self, ag: AgentFrame, surface) -> None:  # type: ignore[no-untyped-def]
        import pygame
        cx = ag.x * self.cell + self.cell // 2
        cy = ag.y * self.cell + self.cell // 2
        # Color gradient: inside marketplace = gold, else green scaled by distance
        if ag.in_marketplace:
            color = (230, 200, 70)
        else:
            # Simple distance shading (cap to avoid fade to black)
            d = min(ag.distance_to_market, 10)
            shade = 160 - d * 10
            color = (80, shade, 120)
        pygame.draw.circle(surface, color, (cx, cy), self.cell // 3)

    def render(self, frame: FrameData) -> None:
        import pygame
        self._ensure_surface(frame)
        assert self.surface is not None and self.font is not None
        surf = self.surface
        surf.fill((25, 25, 32))

        # Marketplace rectangle
        mx = frame.market_x0 * self.cell
        my = frame.market_y0 * self.cell
        mw = frame.market_width * self.cell
        mh = frame.market_height * self.cell
        pygame.draw.rect(surf, (60, 80, 150), (mx, my, mw, mh))

        # Agents
        for ag in frame.agents:
            self._draw_agent(ag, surf)
            id_img = self.font.render(str(ag.agent_id), True, (15, 15, 15))
            surf.blit(id_img, (ag.x * self.cell + 2, ag.y * self.cell + 2))

        # HUD
        price_text = self.font.render(
            f"Round {frame.round}  p={frame.prices}  InMkt {frame.participation_count}/{frame.total_agents}", True, (230, 230, 235)
        )
        surf.blit(price_text, (5, frame.grid_height * self.cell + 8))

        pygame.display.flip()
        pygame.time.wait(self.tick_ms)

        # Basic event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
