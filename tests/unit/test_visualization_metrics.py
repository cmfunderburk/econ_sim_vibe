import dataclasses
from types import SimpleNamespace

import numpy as np

from src.visualization.metrics import spatial_convergence_index, frame_hash
from src.visualization.frame_data import FrameData, AgentFrame, build_frame
from src.visualization.frame_provider import ReplayFrameProvider
from src.visualization.frame_provider import LiveFrameProvider
from src.core.types import MarketResult, RationingDiagnostics, Trade


def _frame(seed: int = 0) -> FrameData:
    agents = [
        AgentFrame(agent_id=1, x=0, y=0, in_marketplace=True, distance_to_market=0),
        AgentFrame(agent_id=2, x=1, y=1, in_marketplace=False, distance_to_market=5),
    ]
    return FrameData(
        round=seed,
        grid_width=10,
        grid_height=10,
        market_x0=4,
        market_y0=4,
        market_width=2,
        market_height=2,
        agents=agents,
        prices=[1.0, 2.0, 3.0],
        participation_count=1,
        total_agents=2,
        max_distance_to_market=10,
        avg_distance_to_market=5.0,
    )


def test_spatial_convergence_index_basic():
    assert spatial_convergence_index(5.0, 10.0) == 0.5


def test_spatial_convergence_index_clamps():
    # Negative inputs clamp to 0, division by zero yields 0
    assert spatial_convergence_index(-1.0, 10.0) == 0.0
    assert spatial_convergence_index(5.0, -2.0) == 0.0
    assert spatial_convergence_index(5.0, 0.0) == 0.0
    # Overshoot clamps at 1
    assert spatial_convergence_index(12.0, 10.0) == 1.0


def test_frame_hash_stability_and_delta():
    f1 = _frame(1)
    f2 = dataclasses.replace(f1)  # identical copy
    assert frame_hash(f1) == frame_hash(f2)
    # Modify a priced field -> hash changes
    f3 = dataclasses.replace(f1, prices=[1.0, 2.1, 3.0])
    assert frame_hash(f1) != frame_hash(f3)


def test_replay_frame_provider_sequence():
    frames = [_frame(i) for i in range(3)]
    provider = ReplayFrameProvider(frames)
    assert provider.current().round == 0
    assert provider.advance() is True
    assert provider.current().round == 1
    assert provider.advance() is True
    assert provider.current().round == 2
    # End reached
    assert provider.advance() is False
    assert provider.has_next() is False
    # Seek and replay
    provider.seek(1)
    assert provider.current().round == 1


def test_replay_provider_seek_out_of_bounds():
    provider = ReplayFrameProvider([_frame(0)])
    try:
        provider.seek(5)
        assert False, "Expected IndexError"
    except IndexError:
        pass


def test_frame_hash_unknown_field():
    f = _frame(0)
    try:
        frame_hash(f, extra_fields=["nonexistent_field"])  # type: ignore[arg-type]
        assert False, "Expected KeyError for unknown field"
    except KeyError:
        pass


def test_spatial_convergence_index_nan():
    nan_val = float("nan")
    assert spatial_convergence_index(nan_val, 10) == 0.0


def test_provider_convergence_and_digest():
    frames = [_frame(0), _frame(1), _frame(2)]
    rp = ReplayFrameProvider(frames)
    # Baseline captured on first access
    cur_index = rp.convergence_index()
    assert cur_index is not None and 0 <= cur_index <= 1
    d1 = rp.frame_digest()
    rp.advance()
    d2 = rp.frame_digest()
    assert d1 != d2  # different round -> different hash (prices same but round hashed)

    # Live provider builder returns evolving frame (simulate by changing round)
    counter = {"r": 0}

    def build():
        r = counter["r"]
        counter["r"] += 1
        return _frame(r)

    lp = LiveFrameProvider(build)
    h1 = lp.frame_digest()
    lp.advance()
    h2 = lp.frame_digest()
    assert h1 != h2


def test_build_frame_enriched_metrics():
    agents = [SimpleNamespace(agent_id=0), SimpleNamespace(agent_id=1)]
    positions = {
        0: SimpleNamespace(x=0, y=0),
        1: SimpleNamespace(x=1, y=1),
    }

    class StubGrid:
        def __init__(self) -> None:
            self._market_bounds = (0, 0, 2, 2)

        def get_position(self, agent_id: int):
            return positions[agent_id]

        def is_in_marketplace(self, pos):
            x0, y0, w, h = self._market_bounds
            return x0 <= pos.x < x0 + w and y0 <= pos.y < y0 + h

    diagnostics = RationingDiagnostics(
        agent_unmet_buys={0: np.array([0.0, 0.5]), 1: np.array([0.0, 0.0])},
        agent_unmet_sells={0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.25])},
        agent_fill_rates_buy={0: np.array([1.0, 0.5]), 1: np.array([1.0, 1.0])},
        agent_fill_rates_sell={0: np.array([1.0, 1.0]), 1: np.array([1.0, 0.75])},
        good_demand_excess=np.array([0.0, 0.5]),
        good_liquidity_gaps=np.array([0.0, 0.25]),
    )
    market_result = MarketResult(
        executed_trades=[Trade(agent_id=0, good_id=1, quantity=2.0, price=2.0)],
        unmet_demand=np.array([0.0, 0.5]),
        unmet_supply=np.array([0.0, 0.25]),
        total_volume=np.array([0.0, 2.0]),
        prices=np.array([1.0, 2.0]),
        participant_count=2,
        rationing_diagnostics=diagnostics,
    )

    state = SimpleNamespace(
        round=3,
        agents=agents,
        grid=StubGrid(),
        prices=np.array([1.0, 2.0]),
        agent_travel_costs={0: 1.0, 1: 2.0},
        last_market_result=market_result,
        last_solver_rest_norm=5e-7,
        last_solver_status="converged",
        initial_max_distance=2,
    )
    config = SimpleNamespace(
        grid_width=4,
        grid_height=4,
        marketplace_width=2,
        marketplace_height=2,
    )

    frame = build_frame(state, config)
    assert frame.solver_rest_goods_norm == 5e-7
    assert frame.solver_status == "converged"
    assert frame.clearing_efficiency is not None
    assert frame.unmet_sell_share is not None
    # Clearing efficiency should be between 0 and 1
    assert 0.0 <= frame.clearing_efficiency <= 1.0
    # Unmet share should be non-negative
    assert frame.unmet_sell_share >= 0.0
