#!/usr/bin/env python3
"""
Debug script to verify Edgeworth box analytical solution.

This script manually computes the expected equilibrium for the 2x2 Edgeworth box
and compares it with our solver results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from core.agent import Agent
from econ.equilibrium import solve_walrasian_equilibrium, compute_excess_demand

def main():
    print("=== Edgeworth Box Debug ===")
    
    # Setup: Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]  
    #        Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]
    # Expected: p* = [1, 6/7], x₁* = [6/7, 2/7], x₂* = [1/7, 5/7]
    
    agent1 = Agent(
        agent_id=1,
        alpha=np.array([0.6, 0.4]),
        home_endowment=np.array([0.5, 0.0]),
        personal_endowment=np.array([0.5, 0.0]),
        position=(0, 0)
    )
    agent2 = Agent(
        agent_id=2,
        alpha=np.array([0.3, 0.7]),
        home_endowment=np.array([0.0, 0.5]),
        personal_endowment=np.array([0.0, 0.5]),
        position=(0, 0)
    )
    
    print(f"Agent 1: α={agent1.alpha}, ω_total={agent1.home_endowment + agent1.personal_endowment}")
    print(f"Agent 2: α={agent2.alpha}, ω_total={agent2.home_endowment + agent2.personal_endowment}")
    
    # Expected analytical solution
    expected_p = np.array([1.0, 6.0/7.0])
    print(f"Expected prices: {expected_p}")
    
    # Test expected solution
    agents = [agent1, agent2]
    excess_demand_expected = compute_excess_demand(expected_p, agents)
    print(f"Excess demand at expected prices: {excess_demand_expected}")
    print(f"Rest-goods norm: {np.linalg.norm(excess_demand_expected[1:], ord=np.inf)}")
    
    # Solve with our method
    prices, z_rest_norm, walras_dot, status = solve_walrasian_equilibrium(agents)
    print(f"\nSolver results:")
    print(f"Status: {status}")
    print(f"Computed prices: {prices}")
    print(f"Z_rest_norm: {z_rest_norm}")
    print(f"Walras dot: {walras_dot}")
    
    # Check excess demand at computed solution
    excess_demand_computed = compute_excess_demand(prices, agents)
    print(f"Excess demand at computed prices: {excess_demand_computed}")
    
    # Manual calculation for verification
    print(f"\n=== Manual Verification ===")
    
    # At expected prices [1, 6/7]
    p1, p2 = expected_p[0], expected_p[1]
    
    # Agent 1 demand: x1j = α1j * wealth1 / pj
    omega1 = agent1.home_endowment + agent1.personal_endowment
    wealth1 = np.dot(expected_p, omega1)
    demand1 = agent1.alpha * wealth1 / expected_p
    print(f"Agent 1: wealth={wealth1}, demand={demand1}")
    
    # Agent 2 demand
    omega2 = agent2.home_endowment + agent2.personal_endowment  
    wealth2 = np.dot(expected_p, omega2)
    demand2 = agent2.alpha * wealth2 / expected_p
    print(f"Agent 2: wealth={wealth2}, demand={demand2}")
    
    # Total demand and supply
    total_demand = demand1 + demand2
    total_supply = omega1 + omega2
    print(f"Total demand: {total_demand}")
    print(f"Total supply: {total_supply}")
    print(f"Excess demand: {total_demand - total_supply}")
    
    # Check if our analytical solution is correct
    print(f"\n=== Analytical Solution Check ===")
    x1_expected = np.array([6.0/7.0, 2.0/7.0])
    x2_expected = np.array([1.0/7.0, 5.0/7.0])
    print(f"Expected x1: {x1_expected}")
    print(f"Expected x2: {x2_expected}")
    print(f"Computed demand1: {demand1}")
    print(f"Computed demand2: {demand2}")
    print(f"Match x1? {np.allclose(demand1, x1_expected)}")
    print(f"Match x2? {np.allclose(demand2, x2_expected)}")
    

if __name__ == "__main__":
    main()