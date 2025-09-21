# Test Categorization Usage Guide

## Quick Reference

### Running Different Test Categories

```bash
# Core economic tests that must always pass
pytest -m "economic_core"

# Robustness tests (graceful degradation acceptable)  
pytest -m "robustness"

# Tests using real production code (no mocking)
pytest -m "real_functions"

# Integration tests (full system behavior)
pytest -m "integration"

# Validation scenario tests
pytest -m "validation"

# Combine categories
pytest -m "economic_core and real_functions"
pytest -m "robustness or integration"
```

### Adding Categories to New Tests

```python
import pytest
from tests.test_categorization import economic_core, robustness, real_functions

@pytest.mark.economic_core
@pytest.mark.real_functions
class TestFundamentalEconomicProperty:
    """Tests that validate core economic theory implementation."""
    
    def test_cobb_douglas_demand_formula(self):
        """Test that demand matches analytical Cobb-Douglas formula."""
        # This test MUST always pass - it validates fundamental economics
        pass

@pytest.mark.robustness
@pytest.mark.real_functions  
class TestSystemRobustness:
    """Tests for edge cases and stress conditions."""
    
    def test_extreme_market_conditions(self):
        """Test behavior with very large or small parameter values."""
        # Graceful degradation acceptable - we're testing limits
        pass
```

## Categorization Decision Tree

### When to use `economic_core`:
- ✅ Validates fundamental economic theory (Cobb-Douglas, Walras' Law, etc.)
- ✅ Tests analytical formula implementation 
- ✅ Verifies economic invariants that must always hold
- ✅ Checks budget constraints and wealth conservation
- ✅ Tests that compare against known analytical solutions

### When to use `robustness`:
- ✅ Edge cases with extreme parameter values
- ✅ Stress testing with large numbers of agents/goods
- ✅ Boundary conditions and degenerate cases
- ✅ Tests where numerical instability might occur
- ✅ Performance testing under load

### When to use `real_functions`:
- ✅ Tests that call actual production code functions
- ✅ Integration testing of real system components
- ✅ End-to-end functionality validation
- ❌ Avoid for unit tests with heavy mocking

### When to use `integration`:
- ✅ Full simulation runs from start to finish
- ✅ Tests that exercise multiple system components
- ✅ Market clearing with actual agent behavior
- ✅ Multi-round trading simulations

### When to use `validation`:
- ✅ Scenario-based validation tests (V1, V2, etc.)
- ✅ Benchmark comparisons against known results
- ✅ Research validation and academic verification
- ✅ Publication-quality result validation

## Example Classifications

### Economic Core Examples:
```python
# Demand function accuracy
@pytest.mark.economic_core
def test_demand_matches_analytical_formula():
    """x_j = α_j * wealth / p_j must be exact."""
    
# Market clearing accuracy  
@pytest.mark.economic_core
def test_walras_law_conservation():
    """Σ p_j * excess_demand_j = 0 must hold."""
    
# Budget constraint enforcement
@pytest.mark.economic_core
def test_budget_constraint_exact():
    """Σ p_j * x_j ≤ wealth must be satisfied."""
```

### Robustness Examples:
```python
# Extreme preferences
@pytest.mark.robustness
def test_corner_solution_preferences():
    """α ≈ [0.99, 0.01] - may have poor convergence."""
    
# Large scale stress test
@pytest.mark.robustness  
def test_thousand_agent_simulation():
    """Performance and stability with many agents."""
    
# Numerical edge cases
@pytest.mark.robustness
def test_very_small_prices():
    """Behavior when prices approach numerical limits."""
```

## CI/CD Integration Recommendations

### Development Workflow:
1. **Fast feedback**: Run `economic_core` tests on every commit
2. **Pre-merge**: Run `economic_core and real_functions` 
3. **Nightly**: Run full suite including `robustness`
4. **Release**: Run all categories including `validation`

### Build Pipeline Configuration:
```yaml
# Fast feedback (< 30 seconds)
quick_tests:
  command: pytest -m "economic_core"
  
# Pre-merge validation (< 2 minutes)  
core_tests:
  command: pytest -m "economic_core and real_functions"
  
# Full validation (< 10 minutes)
all_tests:
  command: pytest
```

## Failure Handling Guidelines

### Economic Core Failures:
- 🚨 **BLOCK**: All economic_core failures must be fixed before merge
- 🔍 **INVESTIGATE**: These indicate fundamental economic logic errors
- 📊 **PRIORITY**: Fix these before any other test failures

### Robustness Failures:
- ⚠️ **REVIEW**: Assess if failure indicates real problem or expected limit
- 🔧 **IMPROVE**: Consider making system more robust if practical
- 📝 **DOCUMENT**: Note known limitations in robustness tests

### Real Functions Failures:
- 🛠️ **FIX**: These test actual production code behavior
- 🔗 **INTEGRATION**: May indicate integration issues between components
- 🎯 **ACCURACY**: Verify the test correctly represents real usage

## Migration Strategy for Existing Tests

1. **Start with obvious economic core tests**: Equilibrium solving, demand computation
2. **Identify clear robustness tests**: Edge cases, stress tests, boundary conditions  
3. **Mark real function tests**: Tests that don't use mocks
4. **Leave uncategorized initially**: Focus on most important classifications first
5. **Gradual expansion**: Add categories to remaining tests over time

This systematic approach ensures that the most critical economic properties are always validated while allowing for appropriate handling of edge cases and performance testing.