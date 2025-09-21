"""
Agent class implementation for economic simulation.

This module implements the core Agent class with Cobb-Douglas utility functions,
inventory management, and spatial positioning for the economic simulation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# Constants from SPECIFICATION.md
FEASIBILITY_TOL = 1e-10  # Conservation and feasibility checks
MIN_ALPHA = 0.05  # Minimum preference weight (ensures interior solutions)


class Agent:
    """
    Economic agent with Cobb-Douglas preferences and spatial positioning.
    
    Each agent has:
    - Cobb-Douglas utility function: U(x) = ∏_j x_j^α_j where ∑_j α_j = 1
    - Home inventory (strategic storage, cannot be traded remotely)
    - Personal inventory (carried goods, can be traded at marketplace)
    - Grid position for spatial movement
    """
    
    def __init__(
        self, 
        agent_id: int,
        alpha: np.ndarray,
        home_endowment: np.ndarray,
        personal_endowment: np.ndarray,
        position: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize agent with preferences and endowments.
        
        Args:
            agent_id: Unique identifier for this agent
            alpha: Cobb-Douglas preference weights (will be normalized to sum to 1)
            home_endowment: Initial inventory at agent's home
            personal_endowment: Initial inventory carried by agent
            position: Initial (x, y) grid position
        """
        self.agent_id = agent_id
        
        # Normalize preferences and ensure interiority conditions
        alpha = np.array(alpha, dtype=float)
        n_goods = len(alpha)
        
        # Ensure minimum alpha for all goods (interiority condition)
        # If any alpha would be below MIN_ALPHA after normalization, 
        # set all to at least MIN_ALPHA and distribute remaining weight proportionally
        if np.sum(alpha) <= 0:
            # Degenerate case: set equal weights
            alpha = np.ones(n_goods) / n_goods
        else:
            # Ensure all alphas are at least MIN_ALPHA
            min_total = n_goods * MIN_ALPHA
            
            # If the sum is less than required minimum, scale up proportionally
            if np.sum(alpha) < min_total:
                alpha = alpha / np.sum(alpha) * min_total
            
            # Clip to minimum and renormalize
            alpha = np.maximum(alpha, MIN_ALPHA)
            alpha = alpha / np.sum(alpha)
            
            # Double check and fix if still below minimum (shouldn't happen now)
            if np.any(alpha < MIN_ALPHA):
                # Fallback: equal weights
                alpha = np.ones(n_goods) / n_goods
        
        self.alpha = alpha
        
        # Initialize inventories
        self.home_endowment = np.array(home_endowment, dtype=float)
        self.personal_endowment = np.array(personal_endowment, dtype=float)
        
        # Validate non-negativity
        if np.any(self.home_endowment < -FEASIBILITY_TOL):
            raise ValueError(f"Negative home endowment: {self.home_endowment}")
        if np.any(self.personal_endowment < -FEASIBILITY_TOL):
            raise ValueError(f"Negative personal endowment: {self.personal_endowment}")
        
        # Spatial positioning
        self.position = position
        
        # Ensure consistent dimensions
        n_goods = len(self.alpha)
        if len(self.home_endowment) != n_goods or len(self.personal_endowment) != n_goods:
            raise ValueError(f"Dimension mismatch: alpha has {n_goods} goods, "
                           f"home has {len(self.home_endowment)}, "
                           f"personal has {len(self.personal_endowment)}")
    
    @property
    def total_endowment(self) -> np.ndarray:
        """Total endowment across home and personal inventory."""
        return self.home_endowment + self.personal_endowment
    
    @property
    def n_goods(self) -> int:
        """Number of goods in the economy."""
        return len(self.alpha)
    
    def utility(self, consumption: np.ndarray) -> float:
        """
        Compute Cobb-Douglas utility of consumption bundle.
        
        Args:
            consumption: Consumption bundle x
            
        Returns:
            Utility U(x) = ∏_j x_j^α_j
        """
        consumption = np.array(consumption, dtype=float)
        if len(consumption) != self.n_goods:
            raise ValueError(f"Consumption bundle has wrong dimension: "
                           f"expected {self.n_goods}, got {len(consumption)}")
        
        # Ensure non-negativity for log computation
        consumption = np.maximum(consumption, 1e-10)  # Avoid log(0)
        
        # Cobb-Douglas: U(x) = ∏_j x_j^α_j = exp(∑_j α_j * log(x_j))
        log_utility = np.sum(self.alpha * np.log(consumption))
        return np.exp(log_utility)
    
    def demand(self, prices: np.ndarray, wealth: Optional[float] = None) -> np.ndarray:
        """
        Compute optimal demand using Cobb-Douglas closed form.
        
        For Cobb-Douglas utility, optimal demand is:
        x_j = α_j * wealth / p_j
        
        Args:
            prices: Price vector p
            wealth: Available wealth (defaults to p·ω_total)
            
        Returns:
            Optimal consumption bundle x*
        """
        prices = np.array(prices, dtype=float)
        if len(prices) != self.n_goods:
            raise ValueError(f"Price vector has wrong dimension: "
                           f"expected {self.n_goods}, got {len(prices)}")
        
        # Guard against zero/negative prices
        prices = np.maximum(prices, 1e-10)
        
        # Compute wealth if not provided
        if wealth is None:
            wealth = float(np.dot(prices, self.total_endowment))
        
        # Cobb-Douglas demand: x_j = α_j * wealth / p_j
        demand = self.alpha * wealth / prices
        
        return demand
    
    def excess_demand(self, prices: np.ndarray, wealth: Optional[float] = None) -> np.ndarray:
        """
        Compute excess demand: z = demand - total_endowment.
        
        Args:
            prices: Price vector p
            wealth: Available wealth (defaults to p·ω_total)
            
        Returns:
            Excess demand vector z
        """
        optimal_demand = self.demand(prices, wealth)
        return optimal_demand - self.total_endowment
    
    def transfer_goods(self, goods: np.ndarray, to_personal: bool = True) -> None:
        """
        Transfer goods between home and personal inventory.
        
        This can only be done when the agent is at their home position.
        
        Args:
            goods: Quantity of each good to transfer
            to_personal: If True, transfer from home to personal; otherwise reverse
        """
        goods = np.array(goods, dtype=float)
        if len(goods) != self.n_goods:
            raise ValueError(f"Transfer vector has wrong dimension: "
                           f"expected {self.n_goods}, got {len(goods)}")
        
        # Store initial state for conservation check
        initial_total = self.home_endowment + self.personal_endowment
        
        if to_personal:
            # Transfer from home to personal
            if np.any(self.home_endowment - goods < -FEASIBILITY_TOL):
                raise ValueError(f"Insufficient home inventory for transfer: "
                               f"home={self.home_endowment}, transfer={goods}")
            
            self.home_endowment -= goods
            self.personal_endowment += goods
        else:
            # Transfer from personal to home
            if np.any(self.personal_endowment - goods < -FEASIBILITY_TOL):
                raise ValueError(f"Insufficient personal inventory for transfer: "
                               f"personal={self.personal_endowment}, transfer={goods}")
            
            self.personal_endowment -= goods
            self.home_endowment += goods
        
        # Validate conservation invariant
        final_total = self.home_endowment + self.personal_endowment
        if not np.allclose(initial_total, final_total, atol=FEASIBILITY_TOL):
            raise RuntimeError(f"Conservation violated in transfer: "
                             f"initial={initial_total}, final={final_total}")
        
        # Validate non-negativity
        if np.any(self.home_endowment < -FEASIBILITY_TOL):
            raise RuntimeError(f"Negative home endowment after transfer: {self.home_endowment}")
        if np.any(self.personal_endowment < -FEASIBILITY_TOL):
            raise RuntimeError(f"Negative personal endowment after transfer: {self.personal_endowment}")
    
    def move_to(self, new_position: Tuple[int, int]) -> None:
        """
        Update agent's grid position.
        
        Args:
            new_position: New (x, y) grid coordinates
        """
        self.position = new_position
    
    def is_at_marketplace(self, marketplace_bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        Check if agent is currently inside the marketplace.
        
        Args:
            marketplace_bounds: ((min_x, max_x), (min_y, max_y)) marketplace boundaries
            
        Returns:
            True if agent is inside marketplace boundaries
        """
        (min_x, max_x), (min_y, max_y) = marketplace_bounds
        x, y = self.position
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def distance_to_marketplace(self, marketplace_center: Tuple[int, int]) -> int:
        """
        Compute Manhattan distance to marketplace center.
        
        Args:
            marketplace_center: (x, y) center of marketplace
            
        Returns:
            Manhattan distance (L1 norm)
        """
        x_agent, y_agent = self.position
        x_market, y_market = marketplace_center
        return abs(x_agent - x_market) + abs(y_agent - y_market)
    
    def copy(self) -> 'Agent':
        """
        Create a deep copy of this agent.
        
        Returns:
            New Agent instance with identical state
        """
        return Agent(
            agent_id=self.agent_id,
            alpha=self.alpha.copy(),
            home_endowment=self.home_endowment.copy(),
            personal_endowment=self.personal_endowment.copy(),
            position=self.position
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Agent(id={self.agent_id}, α={self.alpha}, "
                f"home={self.home_endowment}, personal={self.personal_endowment}, "
                f"pos={self.position})")