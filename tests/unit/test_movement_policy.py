"""Tests for pluggable movement policy abstraction.

Currently only the greedy Manhattan policy is implemented; these tests
assert core invariants so future policies can be validated against the
baseline spatial economics assumptions (deterministic, monotonic distance
descent toward marketplace absent obstacles).
"""

import numpy as np

from src.core.simulation import (
    SimulationConfig,
    initialize_runtime_state,
    run_round,
    AgentPhase,
)


def test_greedy_policy_monotonic_distance():
    """Agents under greedy policy reduce Manhattan distance by at most 1 and never increase it until arrival."""
    cfg = SimulationConfig(
        name="greedy-distance-test",
        n_agents=5,
        n_goods=2,
        grid_width=12,
        grid_height=12,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=30,
        random_seed=11,
        movement_policy="greedy",
    )
    state = initialize_runtime_state(cfg)

    # Force agents into TO_MARKET phase immediately (simulate post-home prep) for a clearer movement trace
    for a in state.agents:
        state.agent_phases[a.agent_id] = AgentPhase.TO_MARKET

    # Track previous distances
    prev_dist = {
        a.agent_id: state.grid.distance_to_marketplace(
            state.grid.get_position(a.agent_id)
        )
        for a in state.agents
    }

    for _ in range(25):  # sufficient rounds to reach center on 12x12 grid
        state = run_round(state, cfg)
        for a in state.agents:
            pos = state.grid.get_position(a.agent_id)
            dist = state.grid.distance_to_marketplace(pos)
            # Distance never increases
            assert dist <= prev_dist[a.agent_id]
            # Step size at most 1 while not yet at marketplace
            if prev_dist[a.agent_id] > 0:
                assert prev_dist[a.agent_id] - dist in {0, 1}
            prev_dist[a.agent_id] = dist
        if all(d == 0 for d in prev_dist.values()):
            break

    assert all(d == 0 for d in prev_dist.values()), (
        "Not all agents reached marketplace under greedy policy"
    )
