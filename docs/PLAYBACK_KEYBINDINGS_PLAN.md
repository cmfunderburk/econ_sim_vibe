# Playback Keybindings Plan
Date: 2025-09-22
Status: Partial (core keyboard delivered)

## Goals
Provide intuitive, low-latency navigation of replay frames for classroom demonstrations and analysis:
- Rapid scrubbing through early price discovery dynamics.
- Fine-grained inspection of rationing or spatial arrival waves.
- Deterministic shortcuts (no platform ambiguity).

## Proposed Default Keybindings
| Key | Action | Notes |
|-----|--------|-------|
| Space | Play / Pause | 🚢 Implemented 2025-09-22 |
| Right Arrow | Step +1 frame | 🚢 Implemented 2025-09-22; pauses when invoked |
| Left Arrow | Step -1 frame | 🚢 Implemented 2025-09-22; clamps at round 1 |
| Shift + Right | Jump +10 frames | 🚢 Implemented 2025-09-22 (configurable later) |
| Shift + Left | Jump -10 frames | 🚢 Implemented 2025-09-22 |
| Home | Jump to first frame | 🚢 Implemented 2025-09-22 |
| End | Jump to last frame | 🚢 Implemented 2025-09-22 (clamps to final round) |
| G | Open "Go To Round" prompt | Numeric entry, validates range |
| [ | Decrease playback speed | E.g. speed /= 1.5 (min clamp) |
| ] | Increase playback speed | E.g. speed *= 1.5 (max clamp) |
| = (or +) | Reset speed to 1.0 | Convenience |
| P | Toggle price overlay | Show/hide price vector panel |
| D | Toggle distance overlay | Requires geometry sidecar |
| F | Toggle fill-rate overlay | Requires enriched diagnostics |
| H | Show help overlay | Lists keybindings |

## Controller API Additions
```
class PlaybackController:
    def prev_frame(self): ...  # existing or to add
    def jump(self, delta: int): ...  # +/- arbitrary
    def goto(self, round_number: int): ...  # clamp + update state
    def set_speed(self, speed: float): ...
    def increase_speed(self, factor: float = 1.5): ...
    def decrease_speed(self, factor: float = 1.5): ...
```

Implemented: `prev_frame`, `jump`, and `goto` landed in 2025-09-22 scrub controls update.

## State Needed for Overlays
- Price overlay: last frame prices
- Distance overlay: per-agent distances + aggregate stats (min/max/mean)
- Fill-rate overlay: buy/sell fill rates per good (aggregate and possibly histogram)

## Configuration Hooks
Optional `replay_config.yaml` (future):
```
keybindings:
  jump_large: 10
  speed_factor: 1.5
  speed_min: 0.1
  speed_max: 8.0
```

## Testing Strategy
- Unit test controller jump/seek clamping (see `test_step_back_and_jump`).
- Snapshot test: keybinding help overlay includes required keys.
- Integration test: simulate sequence (play -> step -> jump -> goto) ensures round indices match expectations.

## Future Considerations
- Mouse / scroll-wheel scrub bar.
- Timeline markers (first clearing round, first all-arrived round).
- Bookmark frames (user-defined, exported in separate JSON).
