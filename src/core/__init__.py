"""
Core module: Agent definitions, simulation state, and utilities.

This module contains the fundamental building blocks for the economic simulation:
- Agent class with preferences, endowments, and behavior
- Type definitions for trades, positions, and simulation state
- Core data structures and utilities
"""

# Import core classes
from .agent import Agent
from .types import Trade, Position, SimulationState, MarketResult

# Export classes for easy import
__all__ = ['Agent', 'Trade', 'Position', 'SimulationState', 'MarketResult']