#!/usr/bin/env python3
"""Recalculate analytical Edgeworth box solution"""

import numpy as np

# Agent parameters
alpha1 = np.array([0.6, 0.4])
alpha2 = np.array([0.3, 0.7])
omega1 = np.array([1.0, 0.0])
omega2 = np.array([0.0, 1.0])

print("=== Edgeworth Box Analytical Solution ===")
print(f"Agent 1: α₁ = {alpha1}, ω₁ = {omega1}")
print(f"Agent 2: α₂ = {alpha2}, ω₂ = {omega2}")

# With numéraire p₁ = 1, let p₂ = p
# Wealth: W₁ = p₁ * ω₁¹ + p₂ * ω₁² = 1 * 1 + p * 0 = 1
# Wealth: W₂ = p₁ * ω₂¹ + p₂ * ω₂² = 1 * 0 + p * 1 = p

# Demands:
# x₁¹ = α₁¹ * W₁ / p₁ = 0.6 * 1 / 1 = 0.6
# x₂¹ = α₁² * W₁ / p₂ = 0.4 * 1 / p = 0.4/p
# x₁² = α₂¹ * W₂ / p₁ = 0.3 * p / 1 = 0.3p  
# x₂² = α₂² * W₂ / p₂ = 0.7 * p / p = 0.7

# Market clearing:
# Good 1: x₁¹ + x₁² = 1 → 0.6 + 0.3p = 1 → 0.3p = 0.4 → p = 4/3
# Good 2: x₂¹ + x₂² = 1 → 0.4/p + 0.7 = 1 → 0.4/p = 0.3 → p = 0.4/0.3 = 4/3

p2_equilibrium = 4/3
p1_equilibrium = 1.0
prices = np.array([p1_equilibrium, p2_equilibrium])

print(f"\nEquilibrium prices: p* = {prices}")

# Calculate equilibrium demands
W1 = 1.0
W2 = p2_equilibrium

x1_1 = 0.6 * W1 / p1_equilibrium  # Agent 1, good 1
x2_1 = 0.4 * W1 / p2_equilibrium  # Agent 1, good 2
x1_2 = 0.3 * W2 / p1_equilibrium  # Agent 2, good 1  
x2_2 = 0.7 * W2 / p2_equilibrium  # Agent 2, good 2

demand1 = np.array([x1_1, x2_1])
demand2 = np.array([x1_2, x2_2])

print(f"Agent 1 demand: x₁* = {demand1}")
print(f"Agent 2 demand: x₂* = {demand2}")

# Verify market clearing
total_endowment = omega1 + omega2
total_demand = demand1 + demand2

print(f"\nMarket clearing check:")
print(f"Total endowment: {total_endowment}")
print(f"Total demand: {total_demand}")
print(f"Difference: {total_demand - total_endowment}")
print(f"Market clears: {np.allclose(total_demand, total_endowment)}")

# Express as fractions for clarity
print(f"\nAs fractions:")
print(f"p₂ = {p2_equilibrium} = 4/3")
print(f"x₁¹ = {x1_1} = 3/5")  
print(f"x₂¹ = {x2_1} = 0.4/(4/3) = 0.3 = 3/10")
print(f"x₁² = {x1_2} = 0.3*(4/3) = 0.4 = 2/5")
print(f"x₂² = {x2_2} = 0.7")