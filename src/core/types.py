"""Core type definitions for the economic simulation.

This module defines the essential data structures used throughout the simulation,
including Trade objects for market clearing and simulation state tracking.

Author: AI Assistant
Date: 2024-12-19
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass(frozen=True)
class Trade:
    """Represents a single executed trade in the marketplace.

    Each trade records the transfer of a specific good between the marketplace
    clearing mechanism and an individual agent. Positive quantities represent
    purchases (agent receives goods), negative quantities represent sales
    (agent provides goods to marketplace).

    This follows the specification contract: (agent_id, good_id, quantity, price)
    tuples that preserve economic invariants and enable proper accounting.

    Attributes:
        agent_id: Unique identifier for the trading agent
        good_id: Index of the good being traded (0-based)
        quantity: Amount traded (positive = buy, negative = sell)
        price: Per-unit price at which trade executed (equilibrium price)
    """

    agent_id: int
    good_id: int
    quantity: float
    price: float

    def __post_init__(self):
        """Validate trade data on creation."""
        if self.good_id < 0:
            raise ValueError(f"good_id must be non-negative, got {self.good_id}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if np.isnan(self.quantity) or np.isinf(self.quantity):
            raise ValueError(f"quantity must be finite, got {self.quantity}")
        if np.isnan(self.price) or np.isinf(self.price):
            raise ValueError(f"price must be finite, got {self.price}")

    @property
    def value(self) -> float:
        """Compute the monetary value of this trade."""
        return abs(self.quantity) * self.price

    @property
    def is_purchase(self) -> bool:
        """True if this represents a purchase (positive quantity)."""
        return self.quantity > 0

    @property
    def is_sale(self) -> bool:
        """True if this represents a sale (negative quantity)."""
        return self.quantity < 0


@dataclass
class SimulationState:
    """Captures the complete state of the simulation at a given round.

    This provides a snapshot for analysis, checkpointing, and validation
    of economic invariants across simulation rounds.

    Attributes:
        round_number: Current simulation round (0-based)
        agents: List of all agents in their current state
        prices: Current equilibrium prices (None if no marketplace participants)
        executed_trades: List of trades executed this round
        z_rest_norm: Convergence metric from equilibrium solver
        walras_dot: Walras' Law validation metric
        total_welfare: Sum of all agent utilities
        marketplace_participants: Number of agents currently in marketplace
    """

    round_number: int
    agents: List[Any]  # List[Agent] - avoiding circular import
    prices: Optional[np.ndarray]
    executed_trades: List[Trade]
    z_rest_norm: float
    walras_dot: float
    total_welfare: float
    marketplace_participants: int

    def __post_init__(self):
        """Validate simulation state data."""
        if self.round_number < 0:
            raise ValueError(
                f"round_number must be non-negative, got {self.round_number}"
            )
        if self.marketplace_participants < 0:
            raise ValueError(
                f"marketplace_participants must be non-negative, got {self.marketplace_participants}"
            )
        if self.prices is not None and len(self.prices) == 0:
            raise ValueError("prices array cannot be empty if not None")


@dataclass(frozen=True)
class RationingDiagnostics:
    """Detailed per-agent rationing diagnostics for liquidity gap analysis.
    
    This captures the granular information needed to analyze market liquidity
    constraints and carry-over effects in multi-period simulations.
    
    Attributes:
        agent_unmet_buys: Dict mapping agent_id -> unmet buy quantities per good
        agent_unmet_sells: Dict mapping agent_id -> unmet sell quantities per good
        agent_fill_rates_buy: Dict mapping agent_id -> buy fill rate per good (0-1)  
        agent_fill_rates_sell: Dict mapping agent_id -> sell fill rate per good (0-1)
        good_demand_excess: Per-good excess demand (demand - supply)
        good_liquidity_gaps: Per-good liquidity shortfall indicators
    """
    agent_unmet_buys: Dict[int, np.ndarray]
    agent_unmet_sells: Dict[int, np.ndarray] 
    agent_fill_rates_buy: Dict[int, np.ndarray]
    agent_fill_rates_sell: Dict[int, np.ndarray]
    good_demand_excess: np.ndarray
    good_liquidity_gaps: np.ndarray


@dataclass
class MarketResult:
    """Result of a market clearing operation.

    Contains all the information about what happened during market clearing,
    including executed trades, unmet demand/supply, and economic metrics.

    Attributes:
        executed_trades: List of successfully executed trades
        unmet_demand: Per-good unmet buy orders after rationing
        unmet_supply: Per-good unmet sell orders after rationing
        total_volume: Per-good total quantity traded
        prices: Equilibrium prices used for clearing
        participant_count: Number of agents participating in this clearing
    """

    executed_trades: List[Trade]
    unmet_demand: np.ndarray
    unmet_supply: np.ndarray
    total_volume: np.ndarray
    prices: np.ndarray
    participant_count: int
    rationing_diagnostics: Optional[RationingDiagnostics] = None

    def __post_init__(self):
        """Validate market result consistency."""
        n_goods = len(self.prices)
        if len(self.unmet_demand) != n_goods:
            raise ValueError(
                f"unmet_demand length {len(self.unmet_demand)} != n_goods {n_goods}"
            )
        if len(self.unmet_supply) != n_goods:
            raise ValueError(
                f"unmet_supply length {len(self.unmet_supply)} != n_goods {n_goods}"
            )
        if len(self.total_volume) != n_goods:
            raise ValueError(
                f"total_volume length {len(self.total_volume)} != n_goods {n_goods}"
            )
        if self.participant_count < 0:
            raise ValueError(
                f"participant_count must be non-negative, got {self.participant_count}"
            )

    @property
    def total_trades(self) -> int:
        """Total number of executed trades."""
        return len(self.executed_trades)

    @property
    def clearing_efficiency(self) -> float:
        """Fraction of total demand that was satisfied (0-1)."""
        total_demand = np.sum(self.unmet_demand + self.total_volume)
        if total_demand == 0:
            return 1.0  # Perfect efficiency when no demand
        return np.sum(self.total_volume) / total_demand
