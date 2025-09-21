#!/usr/bin/env python3
"""
Calculate the correct analytical solution for the Edgeworth box.
"""

import numpy as np

def solve_edgeworth_analytical():
    """
    Solve the 2x2 Edgeworth box analytically.
    
    Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]  
    Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]
    
    At equilibrium, for each agent: MRS₁₂ = p₁/p₂
    MRS₁₂ = (∂U/∂x₁)/(∂U/∂x₂) = (α₁/x₁)/(α₂/x₂) = α₁x₂/(α₂x₁)
    
    For agent 1: 0.6*x₁₂/(0.4*x₁₁) = 1/p₂  =>  1.5*x₁₂/x₁₁ = 1/p₂
    For agent 2: 0.3*x₂₂/(0.7*x₂₁) = 1/p₂  =>  (3/7)*x₂₂/x₂₁ = 1/p₂
    
    Also budget constraints:
    Agent 1: x₁₁ + p₂*x₁₂ = 1 + p₂*0 = 1
    Agent 2: x₂₁ + p₂*x₂₂ = 0 + p₂*1 = p₂
    
    Market clearing:
    x₁₁ + x₂₁ = 1
    x₁₂ + x₂₂ = 1
    """
    
    # Use Cobb-Douglas demand functions directly
    # For agent i: x_{ij} = α_{ij} * wealth_i / p_j
    
    # Agent 1 wealth = p₁*1 + p₂*0 = p₁ = 1 (numéraire)
    # Agent 2 wealth = p₁*0 + p₂*1 = p₂
    
    # Agent 1 demands: x₁₁ = 0.6*1/1 = 0.6, x₁₂ = 0.4*1/p₂ = 0.4/p₂
    # Agent 2 demands: x₂₁ = 0.3*p₂/1 = 0.3*p₂, x₂₂ = 0.7*p₂/p₂ = 0.7
    
    # Market clearing for good 1: x₁₁ + x₂₁ = 1
    # 0.6 + 0.3*p₂ = 1
    # 0.3*p₂ = 0.4
    # p₂ = 4/3
    
    p2 = 4.0/3.0
    print(f"Calculated equilibrium price: p₂ = {p2}")
    
    # Verify with demands
    x11 = 0.6 * 1 / 1
    x12 = 0.4 * 1 / p2
    x21 = 0.3 * p2 / 1  
    x22 = 0.7 * p2 / p2
    
    print(f"Agent 1 demands: x₁ = [{x11}, {x12}]")
    print(f"Agent 2 demands: x₂ = [{x21}, {x22}]")
    
    # Check market clearing
    print(f"Good 1 market clearing: {x11} + {x21} = {x11 + x21} (should be 1.0)")
    print(f"Good 2 market clearing: {x12} + {x22} = {x12 + x22} (should be 1.0)")
    
    return np.array([1.0, p2])

if __name__ == "__main__":
    correct_prices = solve_edgeworth_analytical()
    print(f"\nCorrect analytical solution: {correct_prices}")