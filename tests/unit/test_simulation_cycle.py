"""Integration-style tests for the explicit home/market cycle in the simulation."""

import numpy as np

from scripts.run_simulation import (
    SimulationConfig,
    initialize_simulation,
    run_simulation_round,
    AgentPhase,
)


def test_agents_follow_home_market_cycle():
    """Agents load, trade, and return home under the new lifecycle."""
    config = SimulationConfig(
        name="cycle-test",
        n_agents=2,
        n_goods=2,
        grid_width=6,
        grid_height=6,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=20,
        random_seed=7,
    )

    state = initialize_simulation(config)

    # Initial phase: everyone prepping at home
    assert set(state.agent_phases.values()) == {AgentPhase.HOME_PREP}

    # First round should load inventory for travel
    state = run_simulation_round(state, config)
    assert all(
        state.agent_phases[agent.agent_id]
        in {AgentPhase.TO_MARKET, AgentPhase.AT_MARKET}
        for agent in state.agents
    )
    assert all(np.allclose(agent.home_endowment, 0.0) for agent in state.agents)

    # Advance until agents complete a full market trip and return home
    returned_home = False
    for _ in range(30):
        state = run_simulation_round(state, config)
        if all(
            state.agent_phases[agent.agent_id] == AgentPhase.HOME_PREP
            for agent in state.agents
        ) and all(np.allclose(agent.personal_endowment, 0.0) for agent in state.agents):
            returned_home = True
            break

    assert returned_home, "Agents never returned home to reset the daily cycle"
