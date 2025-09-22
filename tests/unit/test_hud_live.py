from pathlib import Path

from scripts.run_simulation import load_config
from src.visualization import build_frame
from src.core.simulation import initialize_runtime_state, run_round


def test_hud_fields_present_live(tmp_path: Path):
    # Use a small config to advance one round and inspect frame
    cfg = load_config(Path("config/small_market.yaml"))
    state = initialize_runtime_state(cfg)
    # Build initial frame (round 0) and then advance
    frame0 = build_frame(state, cfg)
    assert frame0.hud_round_digest is not None  # even at initial snapshot
    assert frame0.convergence_index is not None
    # Run one round
    run_round(state, cfg)
    frame1 = build_frame(state, cfg)
    # Basic invariants
    assert frame1.hud_round_digest is not None
    assert 0.0 <= (frame1.convergence_index or 0.0) <= 1.0
    # Digest should be hex substring
    assert all(c in "0123456789abcdef" for c in frame1.hud_round_digest.lower())
