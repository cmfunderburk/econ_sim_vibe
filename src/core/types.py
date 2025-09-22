"""Core datatypes for economic simulation tests.

These are trimmed to only what current tests access. Additional
fields can be appended in future commits without breaking tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass(frozen=True)
class Trade:
    """Executed trade record.

    Sign convention:
      quantity > 0  => buy (flows into agent)
      quantity < 0  => sell (flows out of agent)
    """

    agent_id: int
    good_id: int
    quantity: float
    price: float

    @property
    def value(self) -> float:
        return self.quantity * self.price

    @property
    def is_purchase(self) -> bool:
        """Return True if this trade is a purchase (positive quantity)."""
        return self.quantity > 0

    @property
    def is_sale(self) -> bool:
        """Return True if this trade is a sale (negative quantity)."""
        return self.quantity < 0


@dataclass
class SimulationState:
    """Minimal simulation state snapshot used by unit tests."""

    round_number: int
    agents: List["Agent"]
    prices: Optional[np.ndarray]
    executed_trades: List[Trade]
    z_rest_norm: float
    walras_dot: float
    total_welfare: float
    marketplace_participants: int


@dataclass
class RationingDiagnostics:
    """Per-agent and per-good rationing diagnostics used for enriched metrics.

    Attributes:
        agent_unmet_buys: agent_id -> np.ndarray of unmet buy quantities
        agent_unmet_sells: agent_id -> np.ndarray of unmet sell quantities
        agent_fill_rates_buy: agent_id -> np.ndarray of (0-1) buy fill rates
        agent_fill_rates_sell: agent_id -> np.ndarray of (0-1) sell fill rates
        good_demand_excess: np.ndarray demand - supply per good (can be negative)
        good_liquidity_gaps: np.ndarray max(0, demand - supply) per good
    """

    agent_unmet_buys: Dict[int, np.ndarray]
    agent_unmet_sells: Dict[int, np.ndarray]
    agent_fill_rates_buy: Dict[int, np.ndarray]
    agent_fill_rates_sell: Dict[int, np.ndarray]
    good_demand_excess: np.ndarray
    good_liquidity_gaps: np.ndarray


@dataclass
class MarketResult:
    """Result bundle from market clearing routine with diagnostics.

    The arrays are sized by number of goods. For goods not traded, volume may be 0.
    """

    executed_trades: List[Trade]
    unmet_demand: np.ndarray
    unmet_supply: np.ndarray
    total_volume: np.ndarray
    prices: np.ndarray
    participant_count: int
    rationing_diagnostics: Optional[RationingDiagnostics] = None

    # Backward compatibility note: earlier versions stored clearing_efficiency as a field.
    # We now expose it as a computed property so that manual instantiation in tests
    # (without providing the field) continues to work.
    @property
    def clearing_efficiency(self) -> float:
        try:
            from constants import FEASIBILITY_TOL  # type: ignore
        except Exception:
            # Fallback small tolerance
            FEASIBILITY_TOL = 1e-10  # noqa: N806

        total_exec = float(np.sum(self.total_volume))
        total_unmet = float(np.sum(self.unmet_demand))
        denom = total_exec + total_unmet
        if denom <= FEASIBILITY_TOL:
            return 1.0  # Convention: perfect efficiency when no demand
        eff = total_exec / denom
        # Numerical safety clamp
        if eff < 0.0:
            return 0.0
        if eff > 1.0:
            return 1.0
        return eff

__all__ = [
    "Trade",
    "SimulationState",
    "MarketResult",
    "RationingDiagnostics",
]
