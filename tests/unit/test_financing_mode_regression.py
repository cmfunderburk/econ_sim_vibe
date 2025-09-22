import os
import sys
import numpy as np
import pytest

# Standalone execution safeguard: ensure project src/ is on sys.path when running this
# test in isolation (some earlier tests may inject it implicitly when running full suite).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from econ.market import execute_constrained_clearing, FinancingMode  # type: ignore  # noqa: E402
from econ.equilibrium import solve_walrasian_equilibrium  # type: ignore  # noqa: E402
from core.agent import Agent  # type: ignore  # noqa: E402


@pytest.mark.economic_core
@pytest.mark.real_functions
def test_default_equals_explicit_personal_mode():
    """Regression guard:
    Calling execute_constrained_clearing without financing_mode argument must
    be EXACTLY equivalent to passing financing_mode=FinancingMode.PERSONAL.

    We construct heterogeneous agents so scaling logic actually engages, then
    check structural equality of all salient MarketResult fields.
    """
    rng = np.random.default_rng(123)
    num_agents = 5
    num_goods = 3

    agents_default = []
    agents_explicit = []
    for i in range(num_agents):
        # Random Cobb-Douglas weights normalized to sum to 1
        alpha_raw = rng.uniform(0.05, 1.0, size=num_goods)
        alpha = alpha_raw / alpha_raw.sum()
        # Split endowment into home (all) and personal (none) so trading capacity arises
        home_endowment = rng.uniform(0.2, 2.5, size=num_goods)
        personal_endowment = np.zeros(num_goods)
        agent_a = Agent(
            agent_id=i,
            alpha=alpha.copy(),
            home_endowment=home_endowment.copy(),
            personal_endowment=personal_endowment.copy(),
        )
        agent_b = Agent(
            agent_id=100 + i,
            alpha=alpha.copy(),
            home_endowment=home_endowment.copy(),
            personal_endowment=personal_endowment.copy(),
        )
        # Simulate loading inventory to personal so trading can occur (mirrors daily pattern)
        agent_a.load_inventory_for_travel()
        agent_b.load_inventory_for_travel()
        agents_default.append(agent_a)
        agents_explicit.append(agent_b)

    prices, _, _, status = solve_walrasian_equilibrium(agents_default)
    assert status == "converged", (
        f"Equilibrium did not converge in regression setup: status={status}"
    )

    result_default = execute_constrained_clearing(agents_default, prices)
    result_explicit = execute_constrained_clearing(
        agents_explicit, prices, financing_mode=FinancingMode.PERSONAL
    )

    # Compare prices echoed back (should be identical reference)
    np.testing.assert_allclose(
        result_default.prices, result_explicit.prices, atol=0, rtol=0
    )

    # Compare unmet demand/supply and total volume (core aggregate outputs)
    np.testing.assert_allclose(
        result_default.unmet_demand, result_explicit.unmet_demand, atol=0, rtol=0
    )
    np.testing.assert_allclose(
        result_default.unmet_supply, result_explicit.unmet_supply, atol=0, rtol=0
    )
    np.testing.assert_allclose(
        result_default.total_volume, result_explicit.total_volume, atol=0, rtol=0
    )

    # Executed trades: agent ids differ because we constructed different id ranges.
    # We assert equivalence of multiset of (good_id, quantity, price) tuples.
    trades_shape_a = sorted(
        [
            (t.good_id, float(t.quantity), float(t.price))
            for t in result_default.executed_trades
        ]
    )
    trades_shape_b = sorted(
        [
            (t.good_id, float(t.quantity), float(t.price))
            for t in result_explicit.executed_trades
        ]
    )
    assert trades_shape_a == trades_shape_b

    # Participant count equality
    assert result_default.participant_count == result_explicit.participant_count
