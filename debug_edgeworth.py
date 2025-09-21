#!/usr/bin/env python3
"""Debug Edgeworth box demand calculation"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Create agents for V1 scenario (from SPECIFICATION.md)
print("Creating Edgeworth box agents...")

# Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]
agent1 = Agent(
    agent_id=1, 
    alpha=np.array([0.6, 0.4]),
    home_endowment=np.array([1.0, 0.0]),
    personal_endowment=np.array([0.0, 0.0]),
    position=(0, 0)
)

print(f"Agent 1 alpha: {agent1.alpha}")
print(f"Agent 1 total endowment: {agent1.total_endowment}")

# Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]  
agent2 = Agent(
    agent_id=2,
    alpha=np.array([0.3, 0.7]),
    home_endowment=np.array([0.0, 1.0]),
    personal_endowment=np.array([0.0, 0.0]),
    position=(0, 0)
)

print(f"Agent 2 alpha: {agent2.alpha}")
print(f"Agent 2 total endowment: {agent2.total_endowment}")

# Test known equilibrium prices: p* = [1, 6/7]
equilibrium_prices = np.array([1.0, 6/7])
print(f"Equilibrium prices: {equilibrium_prices}")

# Compute demands manually and via agent
print("\n--- Agent 1 Demand Calculation ---")
wealth1 = np.dot(equilibrium_prices, agent1.total_endowment)
print(f"Agent 1 wealth: p·ω = {equilibrium_prices} · {agent1.total_endowment} = {wealth1}")

manual_demand1 = agent1.alpha * wealth1 / equilibrium_prices
print(f"Manual demand 1: α₁ * wealth / p = {agent1.alpha} * {wealth1} / {equilibrium_prices} = {manual_demand1}")

computed_demand1 = agent1.demand(equilibrium_prices)
print(f"Agent.demand() result: {computed_demand1}")

print("\n--- Agent 2 Demand Calculation ---")
wealth2 = np.dot(equilibrium_prices, agent2.total_endowment)
print(f"Agent 2 wealth: p·ω = {equilibrium_prices} · {agent2.total_endowment} = {wealth2}")

manual_demand2 = agent2.alpha * wealth2 / equilibrium_prices
print(f"Manual demand 2: α₂ * wealth / p = {agent2.alpha} * {wealth2} / {equilibrium_prices} = {manual_demand2}")

computed_demand2 = agent2.demand(equilibrium_prices)
print(f"Agent.demand() result: {computed_demand2}")

# Expected demands from analytical solution: x₁* = [6/7, 2/7], x₂* = [1/7, 5/7]
expected_demand1 = np.array([6/7, 2/7])
expected_demand2 = np.array([1/7, 5/7])

print(f"\n--- Expected vs Computed ---")
print(f"Expected demand 1: {expected_demand1}")
print(f"Computed demand 1: {computed_demand1}")
print(f"Difference 1: {computed_demand1 - expected_demand1}")

print(f"Expected demand 2: {expected_demand2}")
print(f"Computed demand 2: {computed_demand2}")
print(f"Difference 2: {computed_demand2 - expected_demand2}")

# Check market clearing
total_endowment = agent1.total_endowment + agent2.total_endowment
total_demand = computed_demand1 + computed_demand2
print(f"\n--- Market Clearing Check ---")
print(f"Total endowment: {total_endowment}")
print(f"Total demand: {total_demand}")
print(f"Difference: {total_demand - total_endowment}")