"""
Core module: Agent definitions, simulation state, and utilities.

This module contains the fundamental building blocks for the economic simulation:
- Agent class with preferences, endowments, and behavior
- SimulationState for tracking round-by-round progress  
- Trade dataclass for logging transactions
- Utility functions and common data structures
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set
import numpy as np

# TODO: Implement these core classes as specified in SPECIFICATION.md

@dataclass
class Trade:
    """
    Record of a single trade transaction.
    
    Attributes:
        agent_id: ID of the trading agent
        good_id: Which good was traded (0-indexed)
        quantity: Amount traded (+ = buy, - = sell)
        price: Price per unit for this good
        round: Simulation round when trade occurred
    """
    agent_id: int
    good_id: int
    quantity: float  # + = buy, - = sell
    price: float
    round: int

@dataclass  
class SimulationState:
    """
    Complete state of simulation at end of a round.
    
    Attributes:
        round: Current round number
        prices: Equilibrium prices (None if no pricing this round)
        trades: All trades executed this round
        agent_positions: Map from agent ID to (x, y) grid position
        marketplace_participants: Set of agent IDs currently in marketplace
    """
    round: int
    prices: Optional[np.ndarray]  # None if no pricing this round
    trades: List[Trade]
    agent_positions: Dict[int, Tuple[int, int]]
    marketplace_participants: Set[int]

# TODO: Implement Agent class with Cobb-Douglas preferences
# TODO: Implement utility calculation functions
# TODO: Implement state management utilities