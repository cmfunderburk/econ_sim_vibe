# Contributing to Economic Simulation

Welcome to the economic simulation project! This guide will help you get started with development and ensure your contributions align with our research-grade standards.

## 🚨 Current Development Status

### Critical Issue: Import System Currently Broken

**DEVELOPMENT ENVIRONMENT BROKEN**:
- ❌ **Cannot run any tests**: ModuleNotFoundError prevents pytest execution
- ❌ **Missing package configuration**: No setup.py for proper installation
- ❌ **Setup instructions fail**: Fresh environment setup produces non-functional environment
- ❌ **All test claims unverifiable**: Cannot validate any functionality until imports work

### Setup That Will Fail (Reality Check)

```bash
# This setup procedure produces a broken development environment:
git clone https://github.com/cmfunderburk/econ_sim_vibe.git
cd econ_sim_vibe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (insufficient for proper imports)
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools

# CRITICAL MISSING STEP (not yet implemented):
# pip install -e .  # This should work but setup.py doesn't exist

# These commands will fail with import errors:
make test          # FAILS: ImportError: No module named 'src'
make validate      # FAILS: Cannot import modules
pytest tests/     # FAILS: ModuleNotFoundError
```

### Required Fixes to Enable Development

**BEFORE any real development can begin**:
1. **Create Package Configuration**: Add missing setup.py to enable `pip install -e .`
2. **Fix Import System**: Enable proper development imports of src/ modules
3. **Test in Fresh Environment**: Verify setup actually works for new contributors
4. **Update Instructions**: Document working setup procedure

Only after fixing import system can we validate any claimed functionality.

## Development Workflow

## Development Workflow

### Core Commands
We've provided a Makefile for common development tasks:

```bash
# Run full test suite
make test

# Run tests with parallel execution
make test-fast

# Run validation scenarios
make validate

# Format code
make format

# Check code quality (lint + test)
make check

# Run simulation with config
make run CONFIG=config/edgeworth.yaml SEED=42

# Generate analysis figures
make figures

# Clean temporary files
make clean
```

### Manual Commands
```bash
# Run specific simulation
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42

# Run validation suite
pytest tests/validation/ -v

# Format code
black src/ tests/ scripts/
ruff format src/ tests/ scripts/

# Type checking
mypy src/ --ignore-missing-imports

# Linting
flake8 src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

## Project Structure

```
econ_sim_vibe/
├── src/                    # Core implementation
│   ├── core/              # Agent framework (COMPLETE ✅)
│   ├── econ/              # Economic engine (COMPLETE ✅)
│   │   ├── equilibrium.py # Walrasian equilibrium solver 
│   │   └── market.py      # Market clearing mechanisms
│   └── spatial/           # Spatial extensions (Phase 2)
├── tests/
│   ├── unit/              # Unit tests (74/74 passing ✅)
│   └── validation/        # Economic validation tests (V1-V10)
├── scripts/
│   ├── run_simulation.py  # Main simulation runner (needs implementation)
│   └── validate_scenario.py # Validation runner (needs implementation)
├── config/               # YAML configuration files (10 scenarios ready)
├── copilot_summaries/    # Implementation tracking and Human Summary
└── .github/              # AI assistant instructions and workflows
```

### Current Implementation Status
- ❌ **Critical Blocker**: Import system broken - no tests can run until fixed
- ⚠️ **Agent Framework**: Code exists but cannot verify functionality due to import failures
- ⚠️ **Economic Engine**: Code exists but cannot verify test claims due to import failures  
- ❌ **Testing Suite**: Cannot run any tests due to ModuleNotFoundError
- ❌ **Development Environment**: Non-functional setup prevents contribution

## Contribution Guidelines

### Current Blockers

**BEFORE attempting any development**:
1. **Fix Import System**: Cannot run any code until `setup.py` created and `pip install -e .` works
2. **Verify Test Claims**: All "74/74 tests passing" claims unverifiable due to import failures
3. **Test Fresh Environment**: Setup instructions don't work in clean environments
4. **Document Working Procedure**: Need verified setup instructions that actually work

### Code Standards (When Environment Works)

1. **Economic Correctness**: All code must preserve economic invariants
   - Walras' Law: p·Z_market(p) ≡ 0
   - Conservation: Total goods conserved across all operations
   - Budget constraints: p·x ≤ p·ω for all agents
   - Value feasibility: buy_value ≤ sell_value per round per agent

2. **Code Quality**
   - Use type hints throughout
   - Follow PEP 8 style (enforced by black/ruff)
   - Write docstrings for all public functions
   - Add unit tests for new functionality

3. **Performance**
   - Use vectorized numpy operations where possible
   - Target 100+ agents with <30 seconds per 1000 rounds
   - Profile performance-critical sections

4. **Reproducibility**
   - All experiments must be configurable via YAML
   - Use fixed random seeds: `np.random.seed(seed)` and `random.seed(seed)`
   - Set deterministic environment: `OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1`

### Economic Testing (Cannot Be Verified - Import System Broken)

#### Test Status Claims (Unverifiable)
The project claims comprehensive test coverage but cannot verify:
- **Import failures**: ModuleNotFoundError prevents any test execution
- **Missing package config**: No setup.py or pytest.ini for proper test discovery
- **Broken setup**: Fresh environment setup produces non-functional development environment

#### What Should Work (When Import System Fixed)
```bash
# These should work after fixing imports:
# pip install -e .        # Install package in development mode (MISSING: setup.py)
# make test               # Run full test suite (FAILS: import errors)
# pytest tests/unit/ -v   # Run unit tests (FAILS: ModuleNotFoundError)
```  
- **V6 (Price Normalization)**: p₁ ≡ 1 and convergence criteria
#### Priority Validation Scenarios (When Import System Works)
- **V1 (Edgeworth 2×2)**: Analytical verification against known solution
- **V2 (Spatial Null)**: κ=0 should match Phase-1 exactly  
- **V6 (Price Normalization)**: p₁ ≡ 1 and convergence criteria
- **V10 (Integration Test)**: Fast unit test for economic pipeline

### Pull Request Process

1. **Branch**: Create feature branch from `main`
2. **Implement**: Follow coding standards and add tests
3. **Test**: Ensure all tests pass locally
4. **Document**: Update relevant documentation
5. **Review**: Submit PR with clear description

#### PR Checklist
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Economic invariants are preserved
- [ ] New features have tests
- [ ] Documentation is updated

### Common Implementation Patterns

#### Agent Management
```python
# Always use vectorized operations
endowments = np.array([agent.total_endowment for agent in agents])
positions = np.array([agent.position for agent in agents])

# Market participants only
market_agents = [agent for agent in agents if agent.in_marketplace]
```

#### Numerical Stability
```python
# Use tolerance constants from SPECIFICATION.md
from src.constants import SOLVER_TOL, FEASIBILITY_TOL

# Primary convergence test
assert np.linalg.norm(Z_market[1:], ord=np.inf) < SOLVER_TOL

# Conservation checks
assert abs(total_buys - total_sells) < FEASIBILITY_TOL
```

#### Configuration (When Imports Work)
```python
# Always load from YAML
import yaml
with open('config/scenario.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configurable marketplace size
market_width = config.get('market_width', 2)
market_height = config.get('market_height', 2)
```

## Economic Theory Guidelines (When Environment Works)

### Phase 1: Pure Walrasian
- Focus on equilibrium computation and analytical verification
- No spatial frictions (κ = 0)
- Perfect market clearing

### Phase 2: Spatial Extensions
- Local-participants equilibrium (post-move marketplace agents)
- Constrained execution (personal inventory limits)
- Money-metric welfare measurement

### Phase 3: Future Extensions
- Local price formation
- Bilateral bargaining
- Market microstructure

## Research Standards (Cannot Be Applied - Import System Broken)

### Data Products
- Log all simulation data to Parquet with schema versioning
- Include git SHA and config hash for reproducibility
- Use standardized sign conventions:
  - `z_market[g] = demand - endowment` (+ = excess demand)
  - `executed_net[g] = buys - sells` (+ = net buyer)

### Welfare Measurement
- Use money-metric utilities (equivalent variation)
- Pin to Phase-1 prices for interpersonal comparability
- Report in units of good 1 (numéraire)

### Validation
- Economic scenarios test theoretical properties
- Unit tests verify implementation correctness
- Performance tests ensure scalability targets

## Getting Help

### Documentation
- **[Human Summary](copilot_summaries/Human%20Summary)**: Quick contributor orientation guide
- **[SPECIFICATION.md](SPECIFICATION.md)**: Complete technical specification (825 lines)
- **[README.md](README.md)**: Project overview and contributor quick start
- **AI Instructions**: Comprehensive AI development assistant configuration

### Key Files to Understand (Cannot Be Imported Until Fixed)
- **`src/core/agent.py`**: Economic agents with Cobb-Douglas preferences (UNVERIFIABLE)
- **`src/econ/equilibrium.py`**: Market-clearing price computation (UNVERIFIABLE)
- **`src/econ/market.py`**: Trade execution with inventory constraints (UNVERIFIABLE)
- **`tests/unit/test_components.py`**: Core functionality tests (CANNOT RUN)

### Issues and Discussions
- **Bug reports**: Use GitHub Issues - **PRIORITY**: Import system failure
- **Feature requests**: Must wait until basic development environment works
- **Questions**: Use GitHub Discussions for research collaboration
- **AI Assistant**: Project includes comprehensive AI development instructions

### Code Review Focus
Our reviews prioritize:
1. **Import System Fix**: Priority one - nothing else matters until this works
2. **Environment Verification**: Setup instructions must work in fresh environments
3. **Test Validation**: Verify claimed test results in independent environments
4. **Economic correctness**: Do the economics work as specified? (when testable)
5. **Reproducibility**: Can others reproduce your results? (when possible)

## Release Process (Cannot Be Applied - Environment Broken)

### Validation Requirements
Before any release:
- **CRITICAL**: Fix import system and verify development environment works
- All validation scenarios (V1-V10) must pass (currently untestable)
- Performance benchmarks must meet targets (currently unmeasurable)
- Documentation must be up-to-date (currently contains false claims)
- Cross-platform testing (Linux, macOS, Windows) (currently impossible)

### Versioning
We use semantic versioning:
- **Major**: Breaking changes to economic model or API
- **Minor**: New features, additional validation scenarios
- **Patch**: Bug fixes, performance improvements

---

**IMPORTANT**: Before any meaningful development can occur, the import system must be fixed and setup instructions must work in fresh environments. Thank you for your patience as we address these fundamental development environment issues! 🚨
- **Minor**: New features, additional validation scenarios
- **Patch**: Bug fixes, performance improvements

---

Thank you for contributing to economic simulation research! Your work helps advance our understanding of spatial market mechanisms and agent-based economic modeling.