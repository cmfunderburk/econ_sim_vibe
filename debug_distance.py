#!/usr/bin/env python3
"""Debug spatial distance calculation"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

marketplace_center = (3, 3)

# Agent positions and expected distances
test_cases = [
    ((3, 3), 0),   # At center
    ((1, 3), 2),   # 2 left  
    ((10, 10), 14), # Far away: |10-3| + |10-3| = 7 + 7 = 14
    ((4, 2), 1)    # Edge: |4-3| + |2-3| = 1 + 1 = 2, not 1!
]

for pos, expected in test_cases:
    agent = Agent(
        agent_id=1,
        alpha=np.array([0.5, 0.5]),
        home_endowment=np.array([1.0, 1.0]),
        personal_endowment=np.array([0.5, 0.5]),
        position=pos
    )
    
    distance = agent.distance_to_marketplace(marketplace_center)
    manhattan_calc = abs(pos[0] - 3) + abs(pos[1] - 3)
    
    print(f"Position {pos}:")
    print(f"  Expected: {expected}")
    print(f"  Computed: {distance}")
    print(f"  Manhattan: {manhattan_calc}")
    print(f"  Match: {distance == expected}")
    print()