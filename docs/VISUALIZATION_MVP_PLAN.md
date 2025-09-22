# Visualization MVP Plan
Date: 2025-09-22
Status: Draft

## 1. Objectives
Deliver a minimal but pedagogically effective visualization layer to:
- Illustrate spatial convergence (agent movement toward marketplace)
- Display equilibrium prices & convergence diagnostics
- Show participation, fill, unmet demand/supply, liquidity gaps
- Replay deterministically from logs or run live

## 2. Scope (MVP)
Included:
- pygame window (configurable size; cell size scaling with grid)
- Grid rendering (agents as colored circles/squares; marketplace cells highlighted)
- HUD overlay (text panel) with:
  - Round number / total rounds (if horizon known)
  - Participants: k / N inside marketplace
  - Prices p (p[0] ≡ 1) & rest-goods norm ||Z_rest||_∞
  - avg_distance_to_market, max_distance_to_market
  - Average buy fill rate (executed_buys / requested_buys where requested>0)
  - Aggregate unmet buy/sell share + clearing efficiency
  - Solver status + residual with alert colouring when tolerance breached
  - Spatial convergence index (avg / initial_max)
- Pause / Play / Step / Speed / Jump controls (keyboard)
- Replay mode: load log + geometry sidecar; frame stepping identical controls

Excluded (Future):
- Agent inspector panel
- Live charts (time-series) at bottom or side
- Video export
- A* path traces
- Capacity / congestion visualization

## 3. Architecture
Classes / Modules:
- VisualizationApp: main loop orchestration
- Renderer: draws grid, agents, marketplace
- HUDRenderer: lays out metrics text block(s)
- InputController: maps pygame events → playback/controls
- FrameProvider Protocol:
  - LiveFrameProvider (pulls from simulation state each round)
  - ReplayFrameProvider (streams from parsed log frames)

Dependency Inversion:
VisualizationApp depends on FrameProvider interface returning FrameData:
```
@dataclass
class FrameData:
    round: int
    grid_width: int
    grid_height: int
    market_x0: int
    market_y0: int
    market_width: int
    market_height: int
    agents: list[AgentFrame]
    prices: list[float]
    participation_count: int
    total_agents: int
    avg_buy_fill_rate: float | None
    avg_sell_fill_rate: float | None
    unmet_buy_share: float | None
    unmet_sell_share: float | None
    solver_rest_goods_norm: float | None
    solver_status: str | None
    clearing_efficiency: float | None
    max_distance_to_market: int | None
    avg_distance_to_market: float | None
    convergence_index: float | None
    hud_round_digest: str | None
```

## 4. Data Derivations
Metric | Source | Derivation
------ | ------ | ----------
rest_goods_norm | log field or solver output | Provided by simulation
avg_buy_fill_rate | existing diagnostics arrays | sum(executed_buys)/sum(requested_buys)
unmet_buy_share | sum(unmet_buys)/sum(requested_buys)
unmet_sell_share | sum(unmet_sells)/sum(requested_sells)
clearing_efficiency | MarketResult.clearing_efficiency | Provided property
spatial_index | derived | avg_distance / initial_max_distance (store initial)

## 5. Controls (Keyboard)
Key | Action
--- | ------
Space | Play/Pause toggle
Right Arrow | Step forward (pause mode)
Shift + Right | Jump +10 frames (replay only)
Left Arrow | Step backward (replay only)
Shift + Left | Jump -10 frames (replay only)
Up Arrow | Increase speed (×2 capped)
Down Arrow | Decrease speed (÷2 floored)
Home | Jump to first frame (replay)
End | Jump to final frame (replay)
R | Reset to first frame (legacy binding)
Q / ESC | Quit

## 6. Rendering Guidelines
- Use integer scaling so each cell >= 12px for clarity.
- Distinct agent colors hashed from agent_id (stable across runs).
- Marketplace cells shaded lightly; in-market agents outlined.
- HUD: monospace font; right-align price vector; highlight norm if > SOLVER_TOL.
- When paused, bypass fixed tick delay to keep scrubbing responsive.

## 7. Performance Targets
- 60 FPS when unthrottled visualization for ≤ 100 agents, 40×40 grid.
- Frame render < 5 ms average; simulation step decoupled (fixed update vs variable render).

## 8. Update Loop Strategy
Live Mode:
```
while running:
  process_events()
  if playing and sim_has_next_round(): advance_simulation_one_round()
  frame = build_frame_from_state()
  render(frame)
  regulate_frame_rate()
```
Replay Mode similar but frame advancement pulls from log list.

## 9. Testing Strategy
Test | Purpose
---- | -------
test_frame_provider_live_shapes | FrameData fields populated, lengths consistent
test_spatial_index_monotonic | Index non-increasing except minor float noise
test_replay_seek_consistency | frame_at(n) == sequential_step(n)
test_hud_flag_rest_goods_norm | HUD highlights norm when > tolerance
test_marketplace_bounds_render | Bounds match config geometry sidecar
test_playback_jump_controls | ±1/±10/home/end bindings return expected rounds
test_hud_solver_metrics | Solver residual/status, unmet shares populate HUD with alert colours

## 10. Incremental Delivery Plan
Milestone | Deliverable
--------- | -----------
MVP-1 | FrameProvider interface + replay provider
MVP-2 | Basic renderer (grid + agents + marketplace)
MVP-3 | HUD metrics block
MVP-4 | Controls (play/pause/step/speed)
MVP-5 | Spatial index metric & highlighting
MVP-6 | Refinement (color polish, performance pass)

## 11. Risks
- Drift between live simulation state & log representation (Mitigation: unify through same FrameBuilder helper).
- Font rendering performance (Mitigation: cache rendered text surfaces per value where stable per frame).
- Input handling complexity (Mitigation: dedicated InputController with small state machine).

## 12. Open Questions
- Should we show per-good volume bars? (Defer to post-MVP.)
- Include utility changes per agent? (Inspector phase.)

---
End of Visualization MVP Plan
