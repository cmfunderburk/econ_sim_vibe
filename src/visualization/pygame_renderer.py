from __future__ import annotations

try:  # pragma: no cover - runtime optional
    import pygame
except Exception as e:  # pragma: no cover
    raise RuntimeError("pygame not available: {}".format(e))

from .frame_data import FrameData, AgentFrame
from .base_renderer import BaseRenderer


class PygameRenderer(
    BaseRenderer
):  # pragma: no cover - UI difficult to test automatically
    def __init__(
        self,
        cell_size: int = 26,
        tick_ms: int = 120,
        title: str = "Economic Simulation",
    ) -> None:
        pygame.init()
        self.cell = cell_size
        self.tick_ms = tick_ms
        self.title = title
        self.surface = None
        self.font = None

    def _ensure_surface(self, frame: FrameData) -> None:
        if self.surface is None:
            if frame.grid_width <= 0 or frame.grid_height <= 0:
                raise ValueError(
                    "Frame has non-positive grid dimensions; ensure replay logs include geometry sidecar or fallback inference."
                )
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

    def render(
        self,
        frame: FrameData,
        *,
        is_playing: bool | None = None,
        speed_rps: float | None = None,
    ) -> None:
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

        # HUD primary line
        status = (
            "PLAY"
            if (is_playing or False)
            else "PAUSE"
            if is_playing is not None
            else "LIVE"
        )
        speed_part = f" {speed_rps:.1f}r/s" if speed_rps is not None else ""
        ci_part = (
            f" CI={frame.convergence_index:.2f}" if frame.convergence_index is not None else ""
        )
        price_text = self.font.render(
            f"{status}{speed_part}  Round {frame.round}  p={frame.prices}  InMkt {frame.participation_count}/{frame.total_agents}{ci_part}",
            True,
            (230, 230, 235),
        )
        surf.blit(price_text, (5, frame.grid_height * self.cell + 4))

        # Secondary metrics line (only if available)
        metrics: list[str] = []
        alert_level = 0
        if frame.avg_buy_fill_rate is not None:
            metrics.append(f"buyFill={frame.avg_buy_fill_rate:.2f}")
        if frame.avg_sell_fill_rate is not None:
            metrics.append(f"sellFill={frame.avg_sell_fill_rate:.2f}")
        if frame.unmet_buy_share is not None:
            metrics.append(f"unmetShare={frame.unmet_buy_share:.2f}")
            if frame.unmet_buy_share > 0.25:
                alert_level = max(alert_level, 1)
        if frame.unmet_sell_share is not None:
            metrics.append(f"unmetSell={frame.unmet_sell_share:.2f}")
            if frame.unmet_sell_share > 0.25:
                alert_level = max(alert_level, 1)
        if frame.total_travel_cost is not None:
            metrics.append(f"travelCost={frame.total_travel_cost:.2f}")
        if frame.max_distance_to_market is not None:
            metrics.append(f"maxDist={frame.max_distance_to_market}")
        if frame.avg_distance_to_market is not None:
            metrics.append(f"avgDist={frame.avg_distance_to_market:.2f}")
        if frame.solver_rest_goods_norm is not None:
            metrics.append(f"resid={frame.solver_rest_goods_norm:.2e}")
            if frame.solver_rest_goods_norm > 1e-5:
                alert_level = max(alert_level, 2)
            elif frame.solver_rest_goods_norm > 1e-6:
                alert_level = max(alert_level, 1)
        if frame.clearing_efficiency is not None:
            metrics.append(f"eff={frame.clearing_efficiency:.2f}")
            if frame.clearing_efficiency < 0.75:
                alert_level = max(alert_level, 1)
        if frame.solver_status:
            metrics.append(f"status={frame.solver_status}")
            if frame.solver_status not in {"converged", "no_participants"}:
                alert_level = max(alert_level, 2)
        if frame.hud_round_digest is not None:
            metrics.append(f"dgst={frame.hud_round_digest}")
        if metrics:
            if alert_level >= 2:
                color = (235, 120, 120)
            elif alert_level == 1:
                color = (230, 190, 120)
            else:
                color = (205, 205, 210)
            metrics_text = self.font.render("  ".join(metrics), True, color)
            surf.blit(metrics_text, (5, frame.grid_height * self.cell + 20))

        pygame.display.flip()
        if is_playing is None or is_playing:
            pygame.time.wait(self.tick_ms)

        # Basic event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
