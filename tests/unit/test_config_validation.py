import pytest
from typing import Any
from src.core.simulation import SimulationConfig
from src.core.config_validation import validate_simulation_config


def test_valid_config_passes():
    cfg = SimulationConfig(
        name="Test",
        n_agents=10,
        n_goods=3,
        grid_width=10,
        grid_height=10,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.1,
        max_rounds=20,
        random_seed=1,
    )
    validate_simulation_config(cfg)  # should not raise


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("n_agents", 0, "n_agents must be positive"),
        ("n_goods", 1, "n_goods must be at least 2"),
        ("grid_width", 0, "grid dimensions must be positive"),
        ("marketplace_width", 0, "marketplace dimensions must be positive"),
        ("movement_cost", -0.5, "movement_cost cannot be negative"),
    ],
)
def test_invalid_single_field(field: str, value: Any, expected_fragment: str):
    kwargs = dict(
        name="Bad",
        n_agents=5,
        n_goods=3,
        grid_width=8,
        grid_height=8,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=5,
        random_seed=0,
    )
    kwargs[field] = value
    cfg = SimulationConfig(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as e:
        validate_simulation_config(cfg)
    assert expected_fragment in str(e.value)


def test_market_exceeds_grid():
    cfg = SimulationConfig(
        name="Bad",
        n_agents=5,
        n_goods=3,
        grid_width=5,
        grid_height=5,
        marketplace_width=6,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=5,
        random_seed=0,
    )
    with pytest.raises(ValueError) as e:
        validate_simulation_config(cfg)
    assert "marketplace dimensions cannot exceed" in str(e.value)


def test_full_grid_market_rejection():
    cfg = SimulationConfig(
        name="Bad",
        n_agents=10,
        n_goods=3,
        grid_width=4,
        grid_height=4,
        marketplace_width=4,
        marketplace_height=4,
        movement_cost=0.0,
        max_rounds=5,
        random_seed=0,
    )
    with pytest.raises(ValueError) as e:
        validate_simulation_config(cfg)
    assert "marketplace spans entire grid" in str(e.value)


def test_density_warning_escalated():
    # 8x8 grid -> 64 cells, 50 agents ~78% density > 75%
    cfg = SimulationConfig(
        name="Bad",
        n_agents=50,
        n_goods=3,
        grid_width=8,
        grid_height=8,
        marketplace_width=2,
        marketplace_height=2,
        movement_cost=0.0,
        max_rounds=5,
        random_seed=0,
    )
    with pytest.raises(ValueError) as e:
        validate_simulation_config(cfg)
    assert "agent density" in str(e.value)
