"""Movement policy abstraction layer.

Provides a pluggable interface so different pathfinding / movement strategies
can be swapped without touching the economic or lifecycle logic. The initial
implementation exposes only the current greedy Manhattan step toward a target
(`GreedyManhattanPolicy`). Future policies (e.g., `AStarPolicy`, stochastic
policies, congestion-aware movement) can implement the same interface.

Design goals:
- Deterministic behaviour (critical for reproducibility & tests)
- O(1) per-step cost for the greedy baseline
- Backwards compatible: wraps existing `Grid` movement helpers

Contract:
    move_agent_toward_marketplace(grid, agent_id) -> int
        Moves agent at most one cell toward marketplace. Returns Manhattan
        distance traveled (0 or 1).

    move_agent_toward_position(grid, agent_id, target) -> int
        Moves agent at most one cell toward the supplied target `Position`.

Both methods MUST:
    - Return 0 if already at target / inside marketplace
    - Raise KeyError if agent not on grid (delegated to Grid helpers)
    - Preserve determinism (no RNG inside unless policy documents it)

Future Extension Notes:
- An A* implementation can internally compute full path then cache next steps.
- A distance-field descent policy can precompute BFS distances and take local
  gradient steps (optimal under static obstacles & uniform edge costs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .grid import Grid, Position


class MovementPolicy(Protocol):  # pragma: no cover - protocol structure
    """Protocol for movement policies.

    Concrete implementations must provide deterministic single-step movement
    methods returning the distance moved (0 or 1 under current rules).
    """

    name: str

    def move_agent_toward_marketplace(self, grid: Grid, agent_id: int) -> int: ...

    def move_agent_toward_position(
        self, grid: Grid, agent_id: int, target: Position
    ) -> int: ...


@dataclass
class GreedyManhattanPolicy:
    """Deterministic greedy Manhattan descent (current canonical behaviour).

    Delegates directly to the existing `Grid` convenience methods which move
    in x-direction first, then y (lexicographic tie-breaking), ensuring
    reproducibility.
    """

    name: str = "greedy"

    def move_agent_toward_marketplace(self, grid: Grid, agent_id: int) -> int:
        return grid.move_agent_toward_marketplace(agent_id)

    def move_agent_toward_position(
        self, grid: Grid, agent_id: int, target: Position
    ) -> int:
        return grid.move_agent_toward_position(agent_id, target)


def get_movement_policy(name: str) -> MovementPolicy:
    """Factory returning a movement policy instance.

    Args:
        name: Policy identifier (case-insensitive). Currently supported:
              - "greedy" (default)

    Returns:
        MovementPolicy instance

    Raises:
        ValueError: If name is unknown.
    """
    key = name.lower().strip()
    if key in {"greedy", "greedy_manhattan", "default"}:
        return GreedyManhattanPolicy()
    raise ValueError(f"Unknown movement policy: {name}")
