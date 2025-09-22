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

from dataclasses import dataclass, field
from typing import Protocol, Dict, List, Tuple
import heapq

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
    if key in {"astar", "a*"}:
        return AStarPolicy()
    raise ValueError(f"Unknown movement policy: {name}")


@dataclass
class AStarPolicy:
    """Deterministic A* single-step movement policy.

    Characteristics:
    - Uniform edge cost (1) on 4-neighborhood grid
    - Manhattan distance heuristic (admissible & consistent)
    - Deterministic tie-breaking using (f, g, x, y, insertion_order)
    - Path caching per agent (list of remaining (x,y) steps). Recomputed when
      cache empty or target changes.
    - Lexicographic ordering of neighbor expansion ensures reproducibility across
      Python versions / platforms.

    NOTE: Environment currently has no dynamic obstacles. If future features
    introduce blocking or congestion, cache invalidation must be extended.
    """

    name: str = "astar"
    _cached_paths: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # type: ignore[type-arg]
    _cached_targets: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # type: ignore[type-arg]

    # Public API -----------------------------------------------------
    def move_agent_toward_marketplace(self, grid: Grid, agent_id: int) -> int:
        center = grid.get_marketplace_center()
        if grid.is_in_marketplace(grid.get_position(agent_id)):
            return 0
        return self._advance_one_step(grid, agent_id, center)

    def move_agent_toward_position(
        self, grid: Grid, agent_id: int, target: Position
    ) -> int:
        return self._advance_one_step(grid, agent_id, target)

    # Internal helpers ----------------------------------------------
    def _advance_one_step(
        self, grid: Grid, agent_id: int, target: Position
    ) -> int:
        current = grid.get_position(agent_id)
        if current == target:
            # Clean stale cache
            self._cached_paths.pop(agent_id, None)
            self._cached_targets.pop(agent_id, None)
            return 0

        tgt_key = (target.x, target.y)
        path = self._cached_paths.get(agent_id)
        cached_target = self._cached_targets.get(agent_id)
        if not path or cached_target != tgt_key:
            path = self._compute_path(grid, current, target)
            if path and path[0] == (current.x, current.y):
                path = path[1:]
            self._cached_paths[agent_id] = path
            self._cached_targets[agent_id] = tgt_key

        if not path:
            return 0

        next_x, next_y = path.pop(0)
        self._cached_paths[agent_id] = path
        new_pos = Position(next_x, next_y)
        if new_pos != current:
            grid.move_agent(agent_id, new_pos)
            return 1
        return 0

    def _compute_path(
        self, grid: Grid, start: Position, goal: Position
    ) -> List[Tuple[int, int]]:
        if start == goal:
            return [(start.x, start.y)]

        def h(x: int, y: int) -> int:
            return abs(x - goal.x) + abs(y - goal.y)

        open_heap: List[Tuple[int, int, int, int, int]] = []
        g_score: Dict[Tuple[int, int], int] = {(start.x, start.y): 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        tie_seq = 0
        heapq.heappush(open_heap, (h(start.x, start.y), 0, start.x, start.y, tie_seq))
        visited: set[Tuple[int, int]] = set()
        while open_heap:
            _f, g, x, y, _ = heapq.heappop(open_heap)  # _f unused beyond pop
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if x == goal.x and y == goal.y:
                return self._reconstruct_path(came_from, (x, y))
            for nx, ny in self._neighbors(grid, x, y):
                if (nx, ny) in visited:
                    continue
                tentative_g = g + 1
                prev_g = g_score.get((nx, ny))
                if prev_g is None or tentative_g < prev_g:
                    g_score[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = (x, y)
                    tie_seq += 1
                    f_score = tentative_g + h(nx, ny)
                    heapq.heappush(open_heap, (f_score, tentative_g, nx, ny, tie_seq))
        return [(start.x, start.y)]

    def _neighbors(self, grid: Grid, x: int, y: int) -> List[Tuple[int, int]]:
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid: List[Tuple[int, int]] = []
        for nx, ny in cand:
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                valid.append((nx, ny))
        valid.sort(key=lambda p: (p[0], p[1]))
        return valid

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
