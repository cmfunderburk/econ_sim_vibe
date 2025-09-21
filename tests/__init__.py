"""
Test configuration module.

Common utilities and fixtures for the test suite.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

# Import constants from centralized source with proper path handling
try:
    # Try direct import for installed package
    from src.constants import SOLVER_TOL, FEASIBILITY_TOL
except ImportError:
    # Fallback for development environment
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        from constants import SOLVER_TOL, FEASIBILITY_TOL
    except ImportError:
        # Final fallback - define locally to prevent test failures
        SOLVER_TOL = 1e-8
        FEASIBILITY_TOL = 1e-10

# Test-specific constants
TEST_TOLERANCE = 1e-10


def load_config(config_name: str) -> dict:
    """Load a validation scenario configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# TODO: Add common test fixtures
# TODO: Add economic property validation helpers
# TODO: Add numerical comparison utilities
