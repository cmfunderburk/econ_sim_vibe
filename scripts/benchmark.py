"""Benchmark harness for Walrasian solver performance.

Usage (example):
    python scripts/benchmark.py --agents 50 100 --goods 3 --rounds 50 --seed 42

Reports aggregate timing statistics using the instrumentation provided by
`solve_walrasian_equilibrium` via `get_last_solver_metrics()`.

Design goals:
- Minimal dependency footprint (reuse existing project types)
- Deterministic agent generation (seeded Dirichlet alphas + endowments)
- Synthetic marketplace each round (all agents treated as participants)
- Optional JSON output for CI parsing / regression tracking

This script intentionally bypasses movement/spatial layers to isolate
solver performance.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, cast
import numpy as np

# Local imports (assumes editable install or PYTHONPATH setup)
try:  # Prefer installed package style
    from econ.equilibrium import solve_walrasian_equilibrium, get_last_solver_metrics  # type: ignore
except ImportError:  # Fallback for direct repo execution without editable install
    from src.econ.equilibrium import (  # type: ignore
        solve_walrasian_equilibrium,
        get_last_solver_metrics,
    )

try:
    from constants import SOLVER_TOL, MIN_ALPHA, FEASIBILITY_TOL  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    from src.constants import SOLVER_TOL, MIN_ALPHA, FEASIBILITY_TOL  # type: ignore


@dataclass
class AgentStub:  # Minimal object with required attributes
    agent_id: int
    alpha: np.ndarray
    home_endowment: np.ndarray
    personal_endowment: np.ndarray

    # Uniform interface expected by solver
    def __post_init__(self) -> None:
        # Ensure alpha sums to 1 and min alpha respected
        if not np.isclose(self.alpha.sum(), 1.0):  # pragma: no cover - deterministic path
            self.alpha = self.alpha / self.alpha.sum()


def generate_agents(n_agents: int, n_goods: int, rng: np.random.Generator) -> List[AgentStub]:
    alphas = rng.dirichlet(np.ones(n_goods), size=n_agents)
    # Enforce MIN_ALPHA clip then renormalize
    alphas = np.clip(alphas, MIN_ALPHA, None)
    alphas = alphas / alphas.sum(axis=1, keepdims=True)

    # Random positive endowments (avoid zero wealth agents)
    # Draw from gamma then scale so each good total ~= n_agents
    raw = rng.gamma(shape=2.0, scale=1.0, size=(n_agents, n_goods)) + 0.05
    scale = (n_agents / raw.sum(axis=0))
    endowments = raw * scale

    agents: List[AgentStub] = []
    for i in range(n_agents):
        omega = endowments[i]
        agents.append(
            AgentStub(
                agent_id=i + 1,
                alpha=alphas[i],
                home_endowment=omega.copy(),
                personal_endowment=np.zeros_like(omega),  # total = home + personal
            )
        )
    return agents


def run_round(agents: List[AgentStub]) -> Dict[str, Any]:
    # Prices not used further here; metrics accessed via accessor
    _prices, z_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
    metrics = cast(Dict[str, Any], get_last_solver_metrics())
    return {
        "status": status,
        "z_rest_norm": z_norm,
        "walras_dot": walras_dot,
        **{f"metric_{k}": v for k, v in metrics.items()},
    }


def benchmark(n_agents: int, n_goods: int, n_rounds: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    agents = generate_agents(n_agents, n_goods, rng)

    # Warm-up round (excluded) to mitigate one-time import / JIT overhead
    run_round(agents)

    per_round: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for _ in range(n_rounds):
        result = run_round(agents)
        per_round.append(result)
    total_time = time.perf_counter() - start

    total_solver_time = sum(r["metric_total_time"] for r in per_round)
    fallback_count = sum(1 for r in per_round if r["metric_fallback_used"])

    z_norms = [r["z_rest_norm"] for r in per_round]
    fsolve_times = [r["metric_fsolve_time"] for r in per_round]

    summary = {
        "agents": n_agents,
        "goods": n_goods,
        "rounds": n_rounds,
        "seed": seed,
        "wall_time_total": total_time,
        "wall_time_per_round": total_time / n_rounds,
        "solver_time_accumulated": total_solver_time,
        "solver_time_mean": total_solver_time / n_rounds,
        "solver_time_median": statistics.median(r["metric_total_time"] for r in per_round),
        "z_rest_norm_max": max(z_norms),
        "z_rest_norm_median": statistics.median(z_norms),
        "fsolve_time_mean": statistics.mean(fsolve_times),
        "fallback_rounds": fallback_count,
        "fallback_rate": fallback_count / n_rounds,
        "tolerance": SOLVER_TOL,
        "converged_rounds": sum(1 for r in per_round if r["status"] == "converged"),
    }
    return {"summary": summary, "rounds": per_round}


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walrasian solver benchmark harness")
    p.add_argument("--agents", type=int, nargs="+", default=[50], help="List of agent counts to benchmark")
    p.add_argument("--goods", type=int, default=3, help="Number of goods")
    p.add_argument("--rounds", type=int, default=50, help="Benchmark rounds per configuration (excludes warm-up)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    p.add_argument("--detail", action="store_true", help="Include per-round details in JSON output")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    all_results = []
    for a in args.agents:
        result = benchmark(a, args.goods, args.rounds, args.seed)
        all_results.append(result)
        summary = result["summary"]
        print(
            f"Agents={summary['agents']}, Goods={summary['goods']}, Rounds={summary['rounds']}: "
            f"solver_mean={summary['solver_time_mean']*1e3:.3f} ms, wall_per_round={summary['wall_time_per_round']*1e3:.3f} ms, "
            f"fallback_rate={summary['fallback_rate']*100:.1f}%, max_resid={summary['z_rest_norm_max']:.2e}"
        )

    if args.json:
        payload = {
            "schema": "solver_benchmark_v1",
            "results": [
                {
                    "summary": r["summary"],
                    **({"rounds": r["rounds"]} if args.detail else {}),
                }
                for r in all_results
            ],
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
