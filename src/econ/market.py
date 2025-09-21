"""Market clearing mechanisms for the economic simulation.

This module implements the constrained clearing algorithm with proportional
rationing as specified in the economic specification. It handles trade execution
subject to personal inventory constraints and value feasibility checks.

The clearing mechanism preserves all economic invariants:
- Conservation: total buys = total sells for each good
- Personal inventory constraints: sells ≤ personal stock at market entry
- Value feasibility: buy value ≤ sell value for each agent
- Price consistency: all trades at equilibrium prices

Mathematical Foundation:
For each good g:
  B_g = Σ_i b_ig                           # requested buys
  S_g = Σ_i s_ig                           # available sells (capped by personal inventory)
  Q_g = min(B_g, S_g)                      # executed volume
  
Proportional fills:
  b_ig_exec = Q_g * b_ig / max(B_g, ε)     # buy-side proportional allocation
  s_ig_exec = Q_g * s_ig / max(S_g, ε)     # sell-side proportional allocation

Author: AI Assistant
Date: 2024-12-19
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Import from src.core to fix relative import issues
try:
    from src.core.types import Trade, MarketResult
except ImportError:
    try:
        # For testing context, try relative import
        from ..core.types import Trade, MarketResult
    except ImportError:
        # For standalone execution, try direct import
        from core.types import Trade, MarketResult

# Import agent after types to avoid circular imports
# from ..core.agent import Agent

# Constants from specification
RATIONING_EPS = 1e-10  # Prevent division by zero in rationing
FEASIBILITY_TOL = 1e-10  # Conservation and feasibility checks

logger = logging.getLogger(__name__)


@dataclass
class AgentOrder:
    """Represents an agent's buy/sell orders for all goods.
    
    This internal structure organizes agent orders for efficient processing
    during the market clearing algorithm.
    
    Attributes:
        agent_id: Unique identifier for the agent
        buy_orders: Per-good buy quantities (positive values)
        sell_orders: Per-good sell quantities (positive values)
        max_sell_capacity: Personal inventory at market entry (constrains sells)
    """
    agent_id: int
    buy_orders: np.ndarray
    sell_orders: np.ndarray
    max_sell_capacity: np.ndarray
    
    def __post_init__(self):
        """Validate order consistency."""
        n_goods = len(self.buy_orders)
        if len(self.sell_orders) != n_goods:
            raise ValueError(f"buy_orders and sell_orders must have same length")
        if len(self.max_sell_capacity) != n_goods:
            raise ValueError(f"max_sell_capacity must match goods count")
        
        # Ensure non-negative values
        if np.any(self.buy_orders < 0):
            raise ValueError(f"buy_orders must be non-negative: {self.buy_orders}")
        if np.any(self.sell_orders < 0):
            raise ValueError(f"sell_orders must be non-negative: {self.sell_orders}")
        if np.any(self.max_sell_capacity < 0):
            raise ValueError(f"max_sell_capacity must be non-negative: {self.max_sell_capacity}")
        
        # Check inventory constraints
        if np.any(self.sell_orders > self.max_sell_capacity + FEASIBILITY_TOL):
            excess = self.sell_orders - self.max_sell_capacity
            raise ValueError(f"sell_orders exceed inventory capacity: excess={excess}")


def _generate_agent_orders(agents: List, prices: np.ndarray) -> List[AgentOrder]:
    """Generate buy/sell orders for all marketplace agents.
    
    This computes each agent's desired trades based on Cobb-Douglas optimization
    subject to their current personal inventory constraints.
    
    The key insight: agents in the marketplace can only trade from their personal
    inventory, so their effective wealth for optimization is the value of their
    personal endowment, not their total endowment.
    
    Args:
        agents: List of Agent objects currently in marketplace
        prices: Equilibrium price vector for computing optimal demands
        
    Returns:
        List of AgentOrder objects with buy/sell quantities
        
    Notes:
        - Agents optimize based on personal endowment wealth (what they brought to market)
        - Buy orders: desired quantity minus current personal inventory
        - Sell orders: current personal inventory minus desired quantity  
        - Both are clipped to non-negative values
        - Personal inventory serves as hard constraint on sell capacity
    """
    orders = []
    n_goods = len(prices)
    
    for agent in agents:
        # Current personal inventory (tradeable at marketplace)
        current_personal = agent.personal_endowment.copy()
        
        # Wealth from personal endowment only (what agent brought to market)
        personal_wealth = np.dot(prices, current_personal)
        
        # Compute optimal Cobb-Douglas demand using personal wealth
        # x_j = α_j * wealth / p_j
        if personal_wealth > FEASIBILITY_TOL:
            desired_quantities = agent.alpha * personal_wealth / prices
        else:
            # Zero wealth agent cannot trade
            desired_quantities = np.zeros(n_goods)
        
        # Calculate net orders
        net_orders = desired_quantities - current_personal
        
        # Separate into buy and sell orders
        buy_orders = np.maximum(net_orders, 0.0)  # Positive net = want to buy
        sell_orders = np.maximum(-net_orders, 0.0)  # Negative net = want to sell
        
        # Personal inventory constrains maximum sells
        max_sell_capacity = current_personal.copy()
        
        # Ensure sell orders don't exceed inventory
        sell_orders = np.minimum(sell_orders, max_sell_capacity)
        
        orders.append(AgentOrder(
            agent_id=agent.agent_id,
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            max_sell_capacity=max_sell_capacity
        ))
        
        logger.debug(f"Agent {agent.agent_id}: personal_wealth={personal_wealth:.4f}, "
                    f"desired={desired_quantities}, current={current_personal}, "
                    f"buy={buy_orders}, sell={sell_orders}")
    
    return orders


def _compute_market_totals(orders: List[AgentOrder]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total buy and sell orders across all agents.
    
    Args:
        orders: List of individual agent orders
        
    Returns:
        Tuple of (total_buy_orders, total_sell_orders) arrays
    """
    if not orders:
        # No participants - return zeros
        return np.array([]), np.array([])
    
    n_goods = len(orders[0].buy_orders)
    total_buys = np.zeros(n_goods)
    total_sells = np.zeros(n_goods)
    
    for order in orders:
        total_buys += order.buy_orders
        total_sells += order.sell_orders
    
    return total_buys, total_sells


def _execute_proportional_rationing(orders: List[AgentOrder], 
                                  total_buys: np.ndarray, 
                                  total_sells: np.ndarray,
                                  capacity: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """Execute proportional rationing for each good.
    
    This implements the core clearing algorithm with proportional allocation
    when demand exceeds supply or vice versa.
    
    Args:
        orders: Individual agent orders
        total_buys: Total buy demand per good
        total_sells: Total sell supply per good  
        capacity: Optional per-good throughput limits
        
    Returns:
        Tuple of (executed_buys_dict, executed_sells_dict, executed_volumes):
        - executed_buys_dict: agent_id -> executed buy quantities
        - executed_sells_dict: agent_id -> executed sell quantities  
        - executed_volumes: total quantity traded per good
    """
    n_goods = len(total_buys)
    executed_buys_dict = {}
    executed_sells_dict = {}
    executed_volumes = np.zeros(n_goods)
    
    # Initialize agent dictionaries
    for order in orders:
        executed_buys_dict[order.agent_id] = np.zeros(n_goods)
        executed_sells_dict[order.agent_id] = np.zeros(n_goods)
    
    for good_id in range(n_goods):
        total_buy_g = total_buys[good_id]
        total_sell_g = total_sells[good_id]
        
        # Apply capacity constraints if specified
        max_throughput = capacity[good_id] if capacity is not None else np.inf
        
        # Market clearing volume is minimum of demand, supply, and capacity
        cleared_volume = min(total_buy_g, total_sell_g, max_throughput)
        executed_volumes[good_id] = cleared_volume
        
        logger.debug(f"Good {good_id}: demand={total_buy_g:.4f}, supply={total_sell_g:.4f}, "
                    f"capacity={max_throughput}, cleared={cleared_volume:.4f}")
        
        if cleared_volume <= RATIONING_EPS:
            # No meaningful trade for this good
            continue
        
        # Proportional rationing on buy side
        if total_buy_g > RATIONING_EPS:
            buy_fill_rate = cleared_volume / total_buy_g
            for order in orders:
                executed_buys_dict[order.agent_id][good_id] = order.buy_orders[good_id] * buy_fill_rate
        
        # Proportional rationing on sell side  
        if total_sell_g > RATIONING_EPS:
            sell_fill_rate = cleared_volume / total_sell_g
            for order in orders:
                executed_sells_dict[order.agent_id][good_id] = order.sell_orders[good_id] * sell_fill_rate
    
    return executed_buys_dict, executed_sells_dict, executed_volumes


def _validate_clearing_invariants(orders: List[AgentOrder],
                                executed_buys_dict: Dict[int, np.ndarray],
                                executed_sells_dict: Dict[int, np.ndarray],
                                executed_volumes: np.ndarray,
                                prices: np.ndarray) -> None:
    """Validate economic invariants after clearing.
    
    This performs comprehensive validation of the clearing results to ensure
    all economic constraints are satisfied.
    
    Args:
        orders: Original agent orders
        executed_buys_dict: Executed buy quantities by agent
        executed_sells_dict: Executed sell quantities by agent
        executed_volumes: Total volumes traded per good
        prices: Equilibrium prices
        
    Raises:
        AssertionError: If any economic invariant is violated
    """
    n_goods = len(prices)
    
    # Check market balance: total buys = total sells for each good
    total_executed_buys = np.zeros(n_goods)
    total_executed_sells = np.zeros(n_goods)
    
    for agent_id in executed_buys_dict:
        total_executed_buys += executed_buys_dict[agent_id]
        total_executed_sells += executed_sells_dict[agent_id]
    
    # Market balance invariant
    market_imbalance = np.abs(total_executed_buys - total_executed_sells)
    assert np.all(market_imbalance < FEASIBILITY_TOL), \
        f"Market imbalance detected: {market_imbalance}"
    
    # Volume consistency
    volume_error = np.abs(total_executed_buys - executed_volumes)
    assert np.all(volume_error < FEASIBILITY_TOL), \
        f"Volume inconsistency: {volume_error}"
    
    # Per-agent value feasibility and inventory constraints
    order_dict = {order.agent_id: order for order in orders}
    
    for agent_id in executed_buys_dict:
        order = order_dict[agent_id]
        executed_buys = executed_buys_dict[agent_id]
        executed_sells = executed_sells_dict[agent_id]
        
        # Value feasibility: buy value ≤ sell value
        buy_value = np.dot(prices, executed_buys)
        sell_value = np.dot(prices, executed_sells)
        
        assert buy_value <= sell_value + FEASIBILITY_TOL, \
            f"Agent {agent_id} value infeasible: buy_value={buy_value:.6f} > sell_value={sell_value:.6f}"
        
        # Inventory constraints: sells ≤ personal inventory at entry
        inventory_violation = executed_sells - order.max_sell_capacity
        assert np.all(inventory_violation <= FEASIBILITY_TOL), \
            f"Agent {agent_id} inventory constraint violated: excess={inventory_violation}"
        
        logger.debug(f"Agent {agent_id} validation: buy_value={buy_value:.4f}, "
                    f"sell_value={sell_value:.4f}, inventory_ok={np.all(inventory_violation <= FEASIBILITY_TOL)}")


def _convert_to_trades(executed_buys_dict: Dict[int, np.ndarray],
                     executed_sells_dict: Dict[int, np.ndarray],
                     prices: np.ndarray) -> List[Trade]:
    """Convert executed quantities to Trade objects.
    
    This creates the final trade list with proper sign conventions:
    - Positive quantity = purchase (agent receives goods)
    - Negative quantity = sale (agent provides goods)
    
    Args:
        executed_buys_dict: Executed buy quantities by agent
        executed_sells_dict: Executed sell quantities by agent
        prices: Equilibrium prices for trade pricing
        
    Returns:
        List of Trade objects representing all executed trades
    """
    trades = []
    n_goods = len(prices)
    
    for agent_id in executed_buys_dict:
        executed_buys = executed_buys_dict[agent_id]
        executed_sells = executed_sells_dict[agent_id]
        
        for good_id in range(n_goods):
            # Create buy trades (positive quantity)
            if executed_buys[good_id] > RATIONING_EPS:
                trades.append(Trade(
                    agent_id=agent_id,
                    good_id=good_id,
                    quantity=executed_buys[good_id],
                    price=prices[good_id]
                ))
            
            # Create sell trades (negative quantity)
            if executed_sells[good_id] > RATIONING_EPS:
                trades.append(Trade(
                    agent_id=agent_id,
                    good_id=good_id,
                    quantity=-executed_sells[good_id],
                    price=prices[good_id]
                ))
    
    logger.info(f"Generated {len(trades)} trades from clearing")
    return trades


def execute_constrained_clearing(agents: List, 
                               prices: np.ndarray,
                               capacity: Optional[np.ndarray] = None) -> MarketResult:
    """Execute constrained market clearing with proportional rationing.
    
    This is the main interface for the market clearing mechanism. It implements
    the full algorithm specified in SPECIFICATION.md with proper economic
    invariant validation.
    
    ALGORITHM CONTRACT: Constrained Clearing with Proportional Rationing
    1. Generate orders: For each marketplace agent, compute desired quantity per good
       (Cobb-Douglas: x_ij = alpha_ij * wealth / p_j minus current personal inventory)
    2. Separate buy/sell orders by sign
    3. For each good j:
       a. Execute sells up to min(total_sell_orders_j, personal_inventory_constraint_j)
       b. Execute buys up to min(total_buy_orders_j, executed_sells_j, capacity_j)
       c. If demand exceeds supply: ration buy orders proportionally by requested quantity
       d. If supply exceeds demand: ration sell orders proportionally by offered quantity
    4. Log unexecuted order quantities as carry-over for next round (repriced at future equilibrium)
    5. Return list of executed trades with (agent_id, good_id, quantity, price) tuples
    
    Args:
        agents: List of Agent objects currently in marketplace
        prices: Equilibrium price vector from solve_equilibrium
        capacity: Optional per-good throughput limits
        
    Returns:
        MarketResult with executed trades and clearing statistics
        
    Raises:
        ValueError: If input validation fails
        AssertionError: If economic invariants are violated
    """
    if not agents:
        # No participants - return empty result
        n_goods = len(prices)
        return MarketResult(
            executed_trades=[],
            unmet_demand=np.zeros(n_goods),
            unmet_supply=np.zeros(n_goods),
            total_volume=np.zeros(n_goods),
            prices=prices.copy(),
            participant_count=0
        )
    
    logger.info(f"Executing market clearing for {len(agents)} agents with {len(prices)} goods")
    
    # Input validation
    if len(prices) == 0:
        raise ValueError("prices array cannot be empty")
    if np.any(prices <= 0):
        raise ValueError(f"All prices must be positive: {prices}")
    if capacity is not None and len(capacity) != len(prices):
        raise ValueError(f"capacity length {len(capacity)} != prices length {len(prices)}")
    
    n_goods = len(prices)
    
    # Step 1: Generate agent orders
    orders = _generate_agent_orders(agents, prices)
    
    # Step 2: Compute market totals
    total_buys, total_sells = _compute_market_totals(orders)
    
    logger.info(f"Market totals - Buys: {total_buys}, Sells: {total_sells}")
    
    # Step 3: Execute proportional rationing
    executed_buys_dict, executed_sells_dict, executed_volumes = _execute_proportional_rationing(
        orders, total_buys, total_sells, capacity
    )
    
    # Step 4: Validate economic invariants
    _validate_clearing_invariants(orders, executed_buys_dict, executed_sells_dict, 
                                executed_volumes, prices)
    
    # Step 5: Convert to trade objects
    executed_trades = _convert_to_trades(executed_buys_dict, executed_sells_dict, prices)
    
    # Compute unmet demand/supply
    unmet_demand = total_buys - executed_volumes
    unmet_supply = total_sells - executed_volumes
    
    result = MarketResult(
        executed_trades=executed_trades,
        unmet_demand=unmet_demand,
        unmet_supply=unmet_supply,
        total_volume=executed_volumes,
        prices=prices.copy(),
        participant_count=len(agents)
    )
    
    logger.info(f"Clearing complete: {len(executed_trades)} trades, "
               f"efficiency={result.clearing_efficiency:.3f}")
    
    return result


def apply_trades_to_agents(agents: List, trades: List[Trade]) -> None:
    """Apply executed trades to agent inventories.
    
    This updates agent personal inventories based on the executed trades,
    maintaining conservation of goods across all agents.
    
    Args:
        agents: List of Agent objects to update
        trades: List of Trade objects to apply
        
    Notes:
        - Positive trade quantity adds to personal inventory (purchase)
        - Negative trade quantity removes from personal inventory (sale)
        - Conservation is maintained: total goods unchanged across all agents
    """
    if not trades:
        logger.debug("No trades to apply")
        return
    
    # Create agent lookup for efficiency
    agent_dict = {agent.agent_id: agent for agent in agents}
    
    # Track total changes for conservation validation
    n_goods = len(agents[0].personal_endowment) if agents else 0
    total_change = np.zeros(n_goods)
    
    for trade in trades:
        if trade.agent_id not in agent_dict:
            logger.warning(f"Trade for unknown agent {trade.agent_id}, skipping")
            continue
        
        agent = agent_dict[trade.agent_id]
        
        # Update personal inventory
        agent.personal_endowment[trade.good_id] += trade.quantity
        total_change[trade.good_id] += trade.quantity
        
        logger.debug(f"Applied trade to agent {trade.agent_id}: "
                    f"good {trade.good_id}, quantity {trade.quantity:.4f}")
    
    # Validate conservation
    if n_goods > 0:
        conservation_error = np.abs(total_change)
        if np.any(conservation_error > FEASIBILITY_TOL):
            logger.warning(f"Conservation violation in trade application: {conservation_error}")
    
    logger.info(f"Applied {len(trades)} trades to agent inventories")