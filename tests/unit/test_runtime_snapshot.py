"""Tests for RuntimeSimulationState and snapshot conversion.

Ensures the staged architectural refactor (runtime vs analytical snapshot) preserves
core economic invariants and expected mapping semantics.
"""

import numpy as np

from src.core.simulation import (
    SimulationConfig,
    initialize_runtime_state,
    run_round,
)


def _total_goods(agents):
    return np.sum([a.home_endowment + a.personal_endowment for a in agents], axis=0)


def test_runtime_snapshot_conservation_and_round_mapping():
    """Snapshot must preserve total goods and reflect runtime round number.

    Steps:
    1. Initialize runtime state
    2. Capture baseline total goods
    3. Run a few rounds (movement + possible trading)
    4. Convert to snapshot and re-check conservation and round number mapping
    """
    cfg = SimulationConfig(
        name="snapshot-test",
        n_agents=6,
        n_goods=3,
        grid_width=10,
        grid_height=10,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=5,
        random_seed=1234,
    )

    state = initialize_runtime_state(cfg)
    baseline_total = _total_goods(state.agents)

    # Advance a few rounds (no assertion about trading volume needed)
    for _ in range(3):
        state = run_round(state, cfg)

    snapshot = state.to_snapshot()

    # Round mapping
    assert snapshot.round_number == state.round

    # Conservation (total goods across home + personal must be identical)
    after_total = _total_goods(snapshot.agents)
    assert np.allclose(baseline_total, after_total), (
        f"Total goods changed: baseline={baseline_total}, after={after_total}"
    )

    # Prices shape integrity (either None or vector of length n_goods)
    if snapshot.prices is not None:
        assert len(snapshot.prices) == cfg.n_goods

    # Trades list in snapshot should reference executed trades (may be empty early)
    assert hasattr(snapshot, "executed_trades")

    # Utility monotonicity not enforced here; just ensure utilities computable
    for agent in snapshot.agents:
        u = agent.utility(agent.total_endowment)
        assert u > 0.0


def test_runtime_snapshot_idempotent_multiple_calls():
    """Calling to_snapshot multiple times should not mutate runtime state."""
    cfg = SimulationConfig(
        name="snapshot-idempotent",
        n_agents=4,
        n_goods=2,
        grid_width=8,
        grid_height=8,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=3,
        random_seed=99,
    )
    state = initialize_runtime_state(cfg)
    _ = run_round(state, cfg)

    snap1 = state.to_snapshot()
    snap2 = state.to_snapshot()

    # Ensure identical structural properties
    assert snap1.round_number == snap2.round_number == state.round
    assert len(snap1.executed_trades) == len(snap2.executed_trades)
    assert (snap1.prices is None) == (snap2.prices is None)

    # Total goods equal across snapshots and runtime
    total_runtime = _total_goods(state.agents)
    total_snap1 = _total_goods(snap1.agents)
    total_snap2 = _total_goods(snap2.agents)
    assert np.allclose(total_runtime, total_snap1)
    assert np.allclose(total_runtime, total_snap2)
