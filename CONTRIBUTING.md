# Contributing to Economic Simulation

Welcome to the economic simulation project! This guide will help you get started with development and ensure your contributions align with our research-grade standards.

**Current Status**: Economic Engine complete with 74/74 unit tests passing. **Next Priority**: Validation scenarios V1-V2.

## Quick Start

### Prerequisites
- Python 3.12.3+
- Git
- Virtual environment manager

### Setup
```bash
# Clone the repository
git clone https://github.com/cmfunderburk/econ_sim_vibe.git
cd econ_sim_vibe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools

# Verify everything works
make test          # Should show: 74 tests passing (100% success rate)
make validate      # Run economic validation scenarios
```

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
- ✅ **Agent Framework**: Production-ready with comprehensive testing
- ✅ **Economic Engine**: Complete Walrasian equilibrium solver + market clearing
- ✅ **Testing Suite**: 74 unit tests covering all economic invariants
- 🔄 **Validation Scenarios**: V1-V2 implementation needed (NEXT PRIORITY)
- ⚠️ **Simulation Engine**: Configuration loading and runtime implementation needed

## Contribution Guidelines

### Code Standards

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

### Economic Testing

#### Current Test Status
The project has comprehensive test coverage with **74/74 unit tests passing**:
- **Agent Framework**: 15 unit tests (creation, utilities, inventory management)
- **Equilibrium Solver**: 28 unit tests (price computation, convergence, edge cases)  
- **Market Clearing**: 31 unit tests (trade execution, rationing, economic invariants)

#### Required Tests for New PRs
Every PR must pass:
```bash
# Full unit test suite (should show 74/74 passing)
make test

# Economic invariant validation
pytest tests/unit/test_equilibrium.py -v
pytest tests/unit/test_market_clearing.py -v

# Core validation scenarios (when implemented)
pytest tests/validation/test_scenarios.py -k "V1 or V2"
```

#### Priority Validation Scenarios (Next Implementation Target)
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

#### Configuration
```python
# Always load from YAML
import yaml
with open('config/scenario.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configurable marketplace size
market_width = config.get('market_width', 2)
market_height = config.get('market_height', 2)
```

## Economic Theory Guidelines

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

## Research Standards

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

### Key Files to Understand
- **`src/core/agent.py`**: Economic agents with Cobb-Douglas preferences
- **`src/econ/equilibrium.py`**: Market-clearing price computation
- **`src/econ/market.py`**: Trade execution with inventory constraints
- **`tests/unit/test_components.py`**: Core functionality tests

### Issues and Discussions
- **Bug reports**: Use GitHub Issues with reproduction steps
- **Feature requests**: Use GitHub Issues with economic motivation
- **Questions**: Use GitHub Discussions for research collaboration
- **AI Assistant**: Project includes comprehensive AI development instructions

### Code Review Focus
Our reviews prioritize:
1. **Economic correctness**: Do the economics work as specified?
2. **Reproducibility**: Can others reproduce your results?
3. **Performance**: Does it scale to research targets?
4. **Clarity**: Is the code self-documenting?

## Release Process

### Validation Requirements
Before any release:
- All validation scenarios (V1-V10) must pass
- Performance benchmarks must meet targets
- Documentation must be up-to-date
- Cross-platform testing (Linux, macOS, Windows)

### Versioning
We use semantic versioning:
- **Major**: Breaking changes to economic model or API
- **Minor**: New features, additional validation scenarios
- **Patch**: Bug fixes, performance improvements

---

Thank you for contributing to economic simulation research! Your work helps advance our understanding of spatial market mechanisms and agent-based economic modeling.
We use semantic versioning:
- **Major**: Breaking changes to economic model or API
- **Minor**: New features, additional validation scenarios
- **Patch**: Bug fixes, performance improvements

---

Thank you for contributing to economic simulation research! Your work helps advance our understanding of spatial market mechanisms and agent-based economic modeling.