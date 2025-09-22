"""Spatial grid primitives for movement tests.

Implements a simple rectangular grid with:
 - Position dataclass (hashable)
 - Grid storing agent positions
 - Helper methods used by unit tests: add_agent, get_position,
   move_agent_toward_position (single greedy step), marketplace center
   computation, and move_agent_toward_marketplace (no-op if already inside).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Position:
    x: int
    y: int


class Grid:
    def __init__(self, width: int, height: int, marketplace_width: int, marketplace_height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")
        self.width = width
        self.height = height
        self.marketplace_width = marketplace_width
        self.marketplace_height = marketplace_height
        self._positions: Dict[int, Position] = {}

    # ------------------------------------------------------------------
    # Marketplace geometry
    # ------------------------------------------------------------------
    def get_marketplace_center(self) -> Position:
        return Position(self.width // 2, self.height // 2)

    def _marketplace_bounds(self) -> Tuple[int, int, int, int]:
        cx, cy = self.get_marketplace_center()
        half_w = self.marketplace_width // 2
        half_h = self.marketplace_height // 2
        return cx - half_w, cx + half_w, cy - half_h, cy + half_h

    def is_inside_marketplace(self, pos: Position) -> bool:
        min_x, max_x, min_y, max_y = self._marketplace_bounds()
        return (min_x <= pos.x <= max_x) and (min_y <= pos.y <= max_y)

    # ------------------------------------------------------------------
    # Agent position management
    # ------------------------------------------------------------------
    def add_agent(self, agent_id: int, position: Position) -> None:
        if not (0 <= position.x < self.width and 0 <= position.y < self.height):
            raise ValueError("Agent position out of bounds")
        self._positions[agent_id] = position

    def get_position(self, agent_id: int) -> Position:
        return self._positions[agent_id]

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def move_agent_toward_position(self, agent_id: int, target: Position) -> int:
        if not (0 <= target.x < self.width and 0 <= target.y < self.height):
            raise ValueError("Target position out of bounds")
        current = self._positions[agent_id]
        if current == target:
            return 0
        # Greedy Manhattan step: prioritize x movement then y (lexicographic)
        dx = 0
        if target.x > current.x:
            dx = 1
        elif target.x < current.x:
            dx = -1
        else:
            dy = 0
            if target.y > current.y:
                dy = 1
            elif target.y < current.y:
                dy = -1
            new_pos = Position(current.x, current.y + dy)
            self._positions[agent_id] = new_pos
            return 1 if dy != 0 else 0
        # If we moved in x we do not move in y this step
        new_pos = Position(current.x + dx, current.y)
        self._positions[agent_id] = new_pos
        return 1

    def move_agent_toward_marketplace(self, agent_id: int) -> int:
        pos = self._positions[agent_id]
        if self.is_inside_marketplace(pos):
            return 0
        # Choose nearest point inside marketplace: move toward center
        center = self.get_marketplace_center()
        return self.move_agent_toward_position(agent_id, center)

__all__ = ["Grid", "Position"]
"""Spatial grid implementation for agent movement and marketplace detection.

This module provides the spatial infrastructure for Phase 2 of the economic simulation,
including agent positioning, movement mechanics, and marketplace access detection.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class Position:
    """Represents a position on the grid with basic distance calculations."""

    x: int
    y: int

    def manhattan_distance(self, other: "Position") -> int:
        """Calculate Manhattan (L1) distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, Position) and (self.x, self.y) == (other.x, other.y)


class Grid:
    """Spatial grid for agent movement and marketplace access.

    Provides core spatial functionality for Phase 2 implementation:
    - Agent positioning and movement
    - Marketplace boundary detection
    - Distance calculations for travel costs
    - Basic movement toward marketplace (myopic, one step per round)

    The marketplace is a configurable rectangular region in the center of the grid.
    Only agents within marketplace boundaries can participate in trading.
    """

    def __init__(
        self,
        width: int,
        height: int,
        marketplace_width: int = 2,
        marketplace_height: int = 2,
    ):
        """Initialize grid with specified dimensions and marketplace size.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            marketplace_width: Width of marketplace region (default 2)
            marketplace_height: Height of marketplace region (default 2)
        """
        if width < marketplace_width or height < marketplace_height:
            raise ValueError("Grid must be large enough to contain marketplace")

        self.width = width
        self.height = height
        self.marketplace_width = marketplace_width
        self.marketplace_height = marketplace_height

        # Marketplace is centered
        self.marketplace_x = (width - marketplace_width) // 2
        self.marketplace_y = (height - marketplace_height) // 2

        # Track agent positions
        self.agent_positions: Dict[int, Position] = {}

    def add_agent(self, agent_id: int, position: Position) -> None:
        """Add agent to grid at specified position.

        Args:
            agent_id: Unique agent identifier
            position: Initial position on grid

        Raises:
            ValueError: If position is outside grid boundaries
        """
        if not self._is_valid_position(position):
            raise ValueError(f"Position {position} outside grid bounds")
        self.agent_positions[agent_id] = position

    def move_agent(self, agent_id: int, new_position: Position) -> None:
        """Move agent to new position.

        Args:
            agent_id: Agent to move
            new_position: Target position

        Raises:
            ValueError: If new position is outside grid boundaries
            KeyError: If agent not found on grid
        """
        if agent_id not in self.agent_positions:
            raise KeyError(f"Agent {agent_id} not found on grid")
        if not self._is_valid_position(new_position):
            raise ValueError(f"Position {new_position} outside grid bounds")
        self.agent_positions[agent_id] = new_position

    def get_position(self, agent_id: int) -> Position:
        """Get current position of agent.

        Args:
            agent_id: Agent to locate

        Returns:
            Current position of agent

        Raises:
            KeyError: If agent not found on grid
        """
        if agent_id not in self.agent_positions:
            raise KeyError(f"Agent {agent_id} not found on grid")
        return self.agent_positions[agent_id]

    def is_in_marketplace(self, position: Position) -> bool:
        """Check if position is within marketplace boundaries.

        Args:
            position: Position to check

        Returns:
            True if position is inside marketplace
        """
        return (
            self.marketplace_x
            <= position.x
            < self.marketplace_x + self.marketplace_width
            and self.marketplace_y
            <= position.y
            < self.marketplace_y + self.marketplace_height
        )

    def get_agents_in_marketplace(self) -> Set[int]:
        """Return set of agent IDs currently in marketplace.

        Returns:
            Set of agent IDs within marketplace boundaries
        """
        return {
            agent_id
            for agent_id, pos in self.agent_positions.items()
            if self.is_in_marketplace(pos)
        }

    def distance_to_marketplace(self, position: Position) -> int:
        """Calculate Manhattan distance to nearest marketplace cell.

        Args:
            position: Position to measure from

        Returns:
            Minimum Manhattan distance to any marketplace cell
        """
        if self.is_in_marketplace(position):
            return 0

        min_dist = float("inf")
        for mx in range(
            self.marketplace_x, self.marketplace_x + self.marketplace_width
        ):
            for my in range(
                self.marketplace_y, self.marketplace_y + self.marketplace_height
            ):
                dist = position.manhattan_distance(Position(mx, my))
                min_dist = min(min_dist, dist)
        return int(min_dist)

    def get_marketplace_center(self) -> Position:
        """Get center position of marketplace (for movement targeting).

        Returns:
            Center position of marketplace region
        """
        center_x = self.marketplace_x + self.marketplace_width // 2
        center_y = self.marketplace_y + self.marketplace_height // 2
        return Position(center_x, center_y)

    def move_agent_toward_marketplace(self, agent_id: int) -> int:
        """Move agent one step toward marketplace using myopic strategy.

        Simple movement: move one cell toward marketplace center each round.
        Uses lexicographic tie-breaking (x-direction first, then y-direction)
        for deterministic behavior.

        Args:
            agent_id: Agent to move

        Returns:
            Manhattan distance traveled (0 or 1)

        Raises:
            KeyError: If agent not found on grid
        """
        if agent_id not in self.agent_positions:
            raise KeyError(f"Agent {agent_id} not found on grid")

        current_pos = self.agent_positions[agent_id]
        marketplace_center = self.get_marketplace_center()

        # If already in marketplace, don't move
        if self.is_in_marketplace(current_pos):
            return 0

        return self._move_agent_toward_target(agent_id, marketplace_center)

    def move_agent_toward_position(self, agent_id: int, target: Position) -> int:
        """Move agent one step toward an arbitrary target position.

        Args:
            agent_id: Agent to move
            target: Target grid position

        Returns:
            Manhattan distance traveled (0 or 1)

        Raises:
            KeyError: If agent not found on grid
            ValueError: If target lies outside the grid bounds
        """
        if not self._is_valid_position(target):
            raise ValueError(f"Target position {target} outside grid bounds")

        return self._move_agent_toward_target(agent_id, target)

    def _move_agent_toward_target(self, agent_id: int, target: Position) -> int:
        """Internal helper to move an agent one step toward a target position."""
        if agent_id not in self.agent_positions:
            raise KeyError(f"Agent {agent_id} not found on grid")

        current_pos = self.agent_positions[agent_id]

        # No movement needed if already at target
        if current_pos == target:
            return 0

        new_x = current_pos.x
        new_y = current_pos.y

        # Move in x-direction first (lexicographic tie-breaking)
        if current_pos.x < target.x:
            new_x += 1
        elif current_pos.x > target.x:
            new_x -= 1
        # Then move in y-direction if x-movement didn't occur
        elif current_pos.y < target.y:
            new_y += 1
        elif current_pos.y > target.y:
            new_y -= 1

        new_position = Position(new_x, new_y)
        if new_position != current_pos:
            self.move_agent(agent_id, new_position)
            return 1

        return 0

    def _is_valid_position(self, position: Position) -> bool:
        """Check if position is within grid boundaries.

        Args:
            position: Position to validate

        Returns:
            True if position is valid
        """
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def get_grid_summary(self) -> Dict[str, Any]:
        """Get summary information about current grid state.

        Returns:
            Dictionary with grid statistics and agent distribution
        """
        total_agents = len(self.agent_positions)
        marketplace_agents = len(self.get_agents_in_marketplace())

        return {
            "grid_size": (self.width, self.height),
            "marketplace_bounds": (
                self.marketplace_x,
                self.marketplace_y,
                self.marketplace_width,
                self.marketplace_height,
            ),
            "total_agents": total_agents,
            "marketplace_agents": marketplace_agents,
            "agents_outside_marketplace": total_agents - marketplace_agents,
        }


def create_random_positions(
    n_agents: int, grid_width: int, grid_height: int, seed: Optional[int] = None
) -> List[Position]:
    """Generate random starting positions for agents.

    Args:
        n_agents: Number of positions to generate
        grid_width: Grid width
        grid_height: Grid height
        seed: Random seed for reproducibility

    Returns:
        List of random positions within grid bounds
    """
    if seed is not None:
        random.seed(seed)

    positions = []
    for _ in range(n_agents):
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        positions.append(Position(x, y))

    return positions
