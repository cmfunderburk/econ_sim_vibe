"""
Test configuration module.

Common utilities and fixtures for the test suite.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

# Test configuration constants
TEST_TOLERANCE = 1e-10
SOLVER_TOL = 1e-8
FEASIBILITY_TOL = 1e-10

def load_config(config_name: str) -> dict:
    """Load a validation scenario configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# TODO: Add common test fixtures
# TODO: Add economic property validation helpers
# TODO: Add numerical comparison utilities