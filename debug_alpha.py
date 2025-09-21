#!/usr/bin/env python3
"""Debug alpha normalization"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Test the failing case
alpha_input = np.array([0.001, 0.002, 0.003])
print(f"Original alpha: {alpha_input}")
print(f"Original sum: {np.sum(alpha_input)}")

home_endowment = np.ones(len(alpha_input))
personal_endowment = np.zeros(len(alpha_input))
agent = Agent(agent_id=1, alpha=alpha_input, home_endowment=home_endowment, 
              personal_endowment=personal_endowment, position=(0, 0))

print(f"Agent alpha: {agent.alpha}")
print(f"Agent alpha sum: {np.sum(agent.alpha)}")

# Expected proportions
expected_proportions = alpha_input / np.sum(alpha_input)
print(f"Expected proportions: {expected_proportions}")

# Computed proportions  
computed_proportions = agent.alpha
print(f"Computed proportions: {computed_proportions}")

# Check difference
diff = np.abs(expected_proportions - computed_proportions)
print(f"Absolute differences: {diff}")
print(f"Max difference: {np.max(diff)}")

# Check if interiority constraint is affecting it
MIN_ALPHA = 0.05
print(f"MIN_ALPHA = {MIN_ALPHA}")
print(f"Any alpha below MIN_ALPHA? {np.any(expected_proportions < MIN_ALPHA)}")