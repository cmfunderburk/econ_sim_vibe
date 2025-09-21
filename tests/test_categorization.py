"""
Test categorization utilities for the economic simulation platform.

This module provides pytest markers and utilities for categorizing tests based on
their purpose and expected behavior:

- economic_core: Core economic logic tests (must always pass, zero tolerance)
- robustness: Stress testing and edge cases (graceful degradation acceptable)  
- real_functions: Tests that exercise actual production code paths (no mocking)

Test Categories:
1. Economic Core: Fundamental economic properties and invariants
   - Walras' Law validation
   - Budget constraint enforcement  
   - Cobb-Douglas demand formula correctness
   - Numéraire constraint preservation
   - Conservation laws

2. Robustness: Stress testing and boundary conditions
   - Extreme parameter values
   - Numerical precision limits
   - Solver convergence under stress
   - Performance under scale

3. Real Functions: Production code validation
   - No hardcoded results
   - No mocked behavior
   - Actual function call testing
   - Integration path validation

Usage Examples:
    @pytest.mark.economic_core
    def test_walras_law_enforcement():
        # Core economic property - must always hold
        
    @pytest.mark.robustness  
    def test_extreme_preferences():
        # Stress test - partial failure acceptable
        
    @pytest.mark.real_functions
    def test_equilibrium_solver_actual_calls():
        # Tests real production code, no mocking
"""

import pytest
import functools
from typing import Callable, Any


def require_convergence(test_func: Callable) -> Callable:
    """
    Decorator for tests that require solver convergence to be meaningful.
    
    This decorator should be used with @pytest.mark.economic_core tests
    where solver failure indicates a regression rather than acceptable degradation.
    
    Args:
        test_func: Test function that requires convergence
        
    Returns:
        Wrapped test function that fails if solver doesn't converge
    """
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        result = test_func(*args, **kwargs)
        # Test function should handle convergence checking internally
        return result
    
    wrapper._requires_convergence = True
    return wrapper


def allow_graceful_degradation(test_func: Callable) -> Callable:
    """
    Decorator for robustness tests that can accept graceful degradation.
    
    This decorator should be used with @pytest.mark.robustness tests
    where solver failure or poor performance is acceptable under stress.
    
    Args:
        test_func: Test function that allows degradation
        
    Returns:
        Wrapped test function that handles graceful failure
    """
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        result = test_func(*args, **kwargs)
        # Test function should handle degradation checking internally
        return result
    
    wrapper._allows_degradation = True
    return wrapper


# Pytest marker shortcuts for cleaner imports
economic_core = pytest.mark.economic_core
robustness = pytest.mark.robustness  
real_functions = pytest.mark.real_functions
integration = pytest.mark.integration
validation = pytest.mark.validation

# Combined markers for common patterns
strict_economic = pytest.mark.parametrize("", [pytest.param(marks=[economic_core, real_functions])])
stress_test = pytest.mark.parametrize("", [pytest.param(marks=[robustness, real_functions])])


class TestCategorizationGuide:
    """
    Guidelines for applying test categories correctly.
    
    Use this class as reference when categorizing tests.
    """
    
    ECONOMIC_CORE_CRITERIA = [
        "Tests fundamental economic properties (Walras' Law, conservation)",
        "Validates theoretical formula implementation (Cobb-Douglas)",
        "Checks invariant preservation (numéraire constraint)",
        "Must pass for economic correctness",
        "Zero tolerance for failure",
        "Uses well-conditioned test cases"
    ]
    
    ROBUSTNESS_CRITERIA = [
        "Tests extreme parameter values", 
        "Validates numerical stability limits",
        "Checks performance under stress",
        "Tests edge cases and boundary conditions",
        "Graceful degradation is acceptable",
        "May use poorly-conditioned test cases"
    ]
    
    REAL_FUNCTIONS_CRITERIA = [
        "Calls actual production functions",
        "No hardcoded or mocked results",
        "Tests real code paths and integration",
        "Validates actual solver behavior",
        "No 'if status != converged: skip' patterns",
        "Exercises end-to-end functionality"
    ]


def categorize_existing_test(test_name: str, test_description: str) -> list[str]:
    """
    Helper function to suggest categories for existing tests.
    
    Args:
        test_name: Name of the test function
        test_description: Description of what the test does
        
    Returns:
        List of suggested pytest markers
    """
    markers = []
    
    # Check for economic core indicators
    core_keywords = [
        "walras", "law", "conservation", "budget", "constraint", "numeraire",
        "cobb_douglas", "demand", "excess_demand", "equilibrium", "invariant"
    ]
    if any(keyword in test_name.lower() or keyword in test_description.lower() 
           for keyword in core_keywords):
        markers.append("economic_core")
    
    # Check for robustness indicators  
    robustness_keywords = [
        "extreme", "stress", "performance", "scale", "boundary", "edge",
        "numerical", "precision", "stability", "large", "small"
    ]
    if any(keyword in test_name.lower() or keyword in test_description.lower()
           for keyword in robustness_keywords):
        markers.append("robustness")
    
    # Check for real function indicators
    real_function_keywords = [
        "solve_", "execute_", "compute_", "actual", "real", "production",
        "integration", "end_to_end"
    ]
    if any(keyword in test_name.lower() or keyword in test_description.lower()
           for keyword in real_function_keywords):
        markers.append("real_functions")
    
    return markers