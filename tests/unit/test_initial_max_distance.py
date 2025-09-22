from src.core.simulation import SimulationConfig, initialize_runtime_state, run_round


def test_initial_max_distance_stability():
    config = SimulationConfig(
        name="baseline_test",
        n_agents=6,
        n_goods=3,
        grid_width=15,
        grid_height=15,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.1,
        max_rounds=5,
        random_seed=42,
    )
    state = initialize_runtime_state(config)
    baseline = state.initial_max_distance
    assert baseline is not None and baseline >= 0
    for _ in range(4):
        run_round(state, config)
        assert state.initial_max_distance == baseline
