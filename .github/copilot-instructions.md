# Economic Simulation Vibe Project - AI Coding Instructions

## Project Overview
This is a research-grade economic simulation implementing agent-based modeling with spatial frictions in market economies. The simulation models rational agents trading goods on a configurable spatial grid with centralized marketplace access and movement costs.

**📋 Key References**: 
- [SPECIFICATION.md](../SPECIFICATION.md) for complete technical specification (825+ lines)
- [CONTRIBUTING.md](../CONTRIBUTING.md) for development standards and workflows
- [README.md](../README.md) for project overview and quick-start guide

## 🚨 CRITICAL PROJECT STATUS: Development Environment Broken

### Reality Check for AI Assistants
- ❌ **Import System Failure**: ModuleNotFoundError prevents any test execution
- ❌ **Missing Package Configuration**: No setup.py for proper package installation
- ❌ **Setup Instructions Don't Work**: Fresh environment setup produces non-functional development
- ❌ **All Functionality Claims Unverifiable**: Cannot validate any previous achievements due to import failures

### Previous Claims Cannot Be Verified
The following claims exist in documentation but cannot be verified:
- "Complete Validation Framework ACHIEVED" - UNTESTABLE due to import failures
- "84/84 tests passing" - UNTESTABLE due to ModuleNotFoundError
- "Production-Ready Research Platform" - UNTESTABLE due to non-functional setup
- "All V1-V10 scenarios validated" - UNTESTABLE due to pytest execution failures

### What Code Actually Exists (Unverified)
**Code files present but cannot execute**:
- `src/core/agent.py` - Agent class implementation (appears complete, unverifiable)
- `src/econ/equilibrium.py` - Economic solver code (appears complete, unverifiable)
- `src/econ/market.py` - Market clearing mechanisms (appears complete, unverifiable)
- `tests/unit/test_components.py` - Unit test definitions (cannot run due to imports)
- `tests/validation/test_scenarios.py` - Validation scenarios (cannot run due to imports)

**🔄 Critical Implementation Needed:**
- Package configuration (`setup.py`, `pytest.ini`) to enable imports
- Verified setup instructions that work in fresh environments
- Reality-based documentation that reflects actual functionality
## Implementation Patterns & Standards (When Environment Works)

### File Organization
```
src/
├── core/
│   ├── agent.py          # Agent class with Cobb-Douglas utilities
│   ├── simulation.py     # SimulationEngine and round management
│   └── types.py          # Trade, SimulationState, core dataclasses
├── econ/
│   ├── equilibrium.py    # Walrasian solver with analytical forms
│   ├── market.py         # Market clearing and trade execution
│   └── welfare.py        # Utility measurement and welfare analysis
├── spatial/              # Phase 2: spatial extensions (deferred)
│   ├── grid.py          # Spatial grid and marketplace detection
│   └── movement.py      # A* pathfinding and movement costs
└── config/
    └── loader.py        # YAML configuration loading
```

### Development Workflow & Commands (Cannot Use Until Import System Fixed)

```bash
# These commands will fail until setup.py created:
make install-dev       # FAILS: No package configuration
make test             # FAILS: ImportError: No module named 'src'
make format           # FAILS: Cannot import for checking
make run CONFIG=config/edgeworth.yaml SEED=42  # FAILS: Import errors

# These should work after fixing imports:
# pytest tests/unit/ -v          # Test individual components
# pytest tests/validation/ -v    # Economic validation scenarios V1-V10
```

### Key Configuration Constants (From SPECIFICATION.md)
```python
# Numerical tolerances - USE THESE EXACT VALUES
SOLVER_TOL = 1e-8        # Primary convergence: ||Z_rest||_∞ < SOLVER_TOL  
FEASIBILITY_TOL = 1e-10  # Conservation and feasibility checks
RATIONING_EPS = 1e-10    # Prevent division by zero in rationing

# Economic parameters
NUMERAIRE_GOOD = 0       # Good 1 is numéraire (p[0] ≡ 1.0)
MIN_ALPHA = 0.05         # Minimum preference weight (ensures interior solutions)
```

### Implementation Guidelines

#### Code Quality Standards
- **Type hints**: All functions must have proper type annotations
- **Docstrings**: Include mathematical formulas and economic interpretation
- **Error handling**: Graceful degradation with informative error messages
- **Performance**: Vectorized numpy operations for 100+ agents

#### Economic Correctness Checks
```python
# Every function that modifies economic state should validate invariants
def transfer_goods(agent: Agent, goods: np.ndarray, to_personal: bool = True):
    """Transfer goods between home and personal inventory"""
    initial_total = agent.home_endowment + agent.personal_endowment
    
    if to_personal:
        agent.home_endowment -= goods
        agent.personal_endowment += goods
    else:
        agent.personal_endowment -= goods
        agent.home_endowment += goods
    
    # Invariant: total endowment conserved
    final_total = agent.home_endowment + agent.personal_endowment
    assert np.allclose(initial_total, final_total, atol=FEASIBILITY_TOL)
    
    # Invariant: non-negativity
    assert np.all(agent.home_endowment >= -FEASIBILITY_TOL)
    assert np.all(agent.personal_endowment >= -FEASIBILITY_TOL)
```

## Validation Scenarios (Implementation Targets)

### Priority Scenarios for Phase 1
**V1: Edgeworth Box Verification**
```python
def test_edgeworth_2x2_analytical():
    """Verify against known analytical solution for 2 agents, 2 goods"""
    # Agent 1: α₁ = [0.6, 0.4], ω₁ = [1, 0]  
    # Agent 2: α₂ = [0.3, 0.7], ω₂ = [0, 1]
    # Analytical equilibrium: p* = [1, 6/7], x₁* = [6/7, 2/7], x₂* = [1/7, 5/7]
    
    agents = create_edgeworth_agents()
    prices, z_norm = solve_walrasian_equilibrium(agents)
    
    expected_prices = np.array([1.0, 6/7])
    assert np.allclose(prices, expected_prices, atol=1e-8)
    assert z_norm < SOLVER_TOL
```

**V2: Spatial Null Test**
```python
def test_spatial_null_efficiency():
    """With κ=0 and all agents at marketplace, Phase 2 should equal Phase 1"""
    # This tests that spatial extensions don't break baseline economics
    phase1_result = run_phase1_simulation(agents, n_rounds=1)
    phase2_result = run_phase2_simulation(agents, movement_cost=0.0, n_rounds=1)
    
    assert np.allclose(phase1_result.allocation, phase2_result.allocation)
    assert abs(phase1_result.welfare - phase2_result.welfare) < FEASIBILITY_TOL
```

### Common Implementation Pitfalls ⚠️
```python
# WRONG: Using length of prices array to infer n_goods
n_goods = len(prices)  # ❌ Breaks with normalization

# CORRECT: Use agent data structure
n_goods = agents[0].alpha.size  # ✅ Always reliable

# WRONG: Using Walras' Law for primary convergence test
converged = abs(np.dot(prices, excess_demand)) < tolerance  # ❌ Can be misleading

# CORRECT: Use rest-goods norm as primary test
z_rest_norm = np.linalg.norm(excess_demand[1:], ord=np.inf)
converged = z_rest_norm < SOLVER_TOL  # ✅ Proper convergence test

# WRONG: Forgetting numéraire constraint
prices = scipy.optimize.fsolve(excess_demand, initial_guess)  # ❌ No normalization

# CORRECT: Solve for rest goods with numéraire constraint
def excess_demand_rest(p_rest):
    prices = np.concatenate([[1.0], p_rest])  # p₁ ≡ 1
    return excess_demand(prices)[1:]  # Return rest goods only

p_rest = scipy.optimize.fsolve(excess_demand_rest, initial_guess[1:])
prices = np.concatenate([[1.0], p_rest])  # ✅ Proper normalization
```

## Dependencies & Environment

### Production Dependencies
```python
# requirements.txt (minimal for Phase 1)
numpy>=1.24.0          # Vectorized operations and linear algebra
scipy>=1.10.0          # Optimization for equilibrium solver  
pyyaml>=6.0            # Configuration file loading
```

### Development Dependencies  
```python
# requirements-dev.txt
pytest>=7.0.0          # Testing framework
pytest-cov>=4.0.0      # Coverage reporting
black>=23.0.0          # Code formatting
isort>=5.12.0          # Import sorting
mypy>=1.0.0            # Static type checking
flake8>=6.0.0          # Linting
```

### Performance Targets (Phase 1)
- **Agent scale**: 20-50 agents for initial validation, design for 100+
- **Convergence**: Solver should converge in <10 iterations for typical cases
- **Memory**: O(n_agents × n_goods) memory complexity, avoid quadratic growth
- **Timing**: Complete V1-V2 validation in <5 seconds on modern hardware
## Key Reference Documents
- **[SPECIFICATION.md](../SPECIFICATION.md)**: Complete 600+ line technical specification (authoritative source)
- **[CONTRIBUTING.md](../CONTRIBUTING.md)**: Development standards and economic theory guidelines  
- **[README.md](../README.md)**: Project overview and current implementation status
- **[Makefile](../Makefile)**: Development workflow commands and automation

## Phase 2+ Future Extensions (Deferred)

### Spatial Extensions (Phase 2)
- **Grid-based movement**: A* pathfinding with Manhattan distance
- **Marketplace access**: Only agents in central marketplace can trade
- **Movement costs**: κ·distance budget deduction in numéraire units
- **Local-participants equilibrium**: Prices computed from marketplace agents only

### Local Price Formation (Phase 3)  
- **Bilateral bargaining**: Nash bargaining for co-located agents
- **Dynamic pricing**: Prices emerge from local trading, not global computation
- **Spatial arbitrage**: Price differences drive agent movement decisions

### Advanced Features (Phase 4+)
- **Production**: Firms, technology, factor markets
- **Money & credit**: Monetary economics, banking systems
- **Behavioral economics**: Learning, bounded rationality, social preferences

## AI Assistant Guidelines 🤖

### When Working on This Project
1. **Always prioritize economic correctness** over coding convenience
2. **Validate invariants** in every function that modifies economic state
3. **Use proper mathematical notation** from SPECIFICATION.md (Z_market, not Z)
4. **Test against known analytical solutions** (Edgeworth box, simple cases)
5. **Follow .gitignore rules** - exclude venv/, __pycache__/, *.pyc, etc.
6. **Focus on Phase 1 implementation** - spatial features come later

### Current Implementation Context
- **Working code base**: Agent framework complete, economic engine complete and comprehensively tested, validation scenarios V1-V2 needed next
- **Mathematical foundation**: Complete and validated in SPECIFICATION.md
- **Testing framework**: 84/84 unit tests passing with comprehensive validation across 6 categories
- **Configuration**: YAML configs exist, need runtime loading implementation
- **Development tools**: Complete (Makefile, requirements, formatting, etc.)
- **Recent completion**: Complete Validation Framework with comprehensive testing validation (Sep 20, 2025)

### Summary Documentation (Optional)
When completing significant work, you may optionally add a summary to `copilot_summaries/` for future reference:

```
Date: [Current Date]
Title: [Brief Description]

- What was changed
- Files affected  
- Current status
- Next steps
```

### Current Implementation Context
- **Working code base**: Code files exist but cannot verify functionality due to broken import system
- **Critical blocker**: Missing setup.py prevents any development or testing
- **Development environment**: Non-functional - setup instructions don't work in fresh environments
- **Priority**: Fix import system before any other development

### AI Assistant Guidelines 🤖

### When Working on This Project
1. **Priority**: Fix development environment before economic features
2. **Reality check**: Verify setup works in fresh environments before claiming success
3. **Honest assessment**: Don't claim functionality that cannot be tested due to import failures
4. **Focus**: Package configuration and basic development setup over advanced features

### Common Requests & Responses
- **"Add configuration integration"** → First fix import system, then verify if economic code actually works
- **"Fix economic bug"** → Cannot debug without functional development environment
- **"Add spatial features"** → Must create working development environment first
- **"Test the implementation"** → Cannot run any tests until import system fixed
- **"Optimize performance"** → Cannot measure performance until tests can run

This project requires basic development environment functionality before any economic simulation work can proceed.
- **Research integration**: Design for testing economic hypotheses and policy experiments

## Git Workflow & Commit Protocol (Simplified) 🔄

### Automatic Commit/Push Behavior
When the user requests to "commit" or "push" changes, automatically execute git operations with descriptive messages. Do not ask for confirmation - proceed directly with staging, committing, and pushing.

### Simple Commit Message Format
Use this concise format for commits:

```
[type]: [Brief summary of changes]

- [What was done]
- [Key files changed]
- [Testing status or known issues]

[Next steps if applicable]
```

### Commit Types
- `feat:` New feature implementation
- `docs:` Documentation updates
- `test:` Testing additions
- `fix:` Bug fixes
- `config:` Configuration changes
- `env:` Development environment fixes

### Quick Commit Guidelines
1. **Be specific**: What actually changed?
2. **Note blockers**: Any import failures or setup issues?
3. **Test status**: Can tests run? Do they pass?
4. **Next steps**: What should be done next?

### Example Simplified Commit Messages
```
fix: Replace false test claims with honest import failure status

- Updated README.md, CONTRIBUTING.md to reflect broken import system
- Removed unverifiable "84/84 tests passing" claims
- Added setup.py requirement and development environment blockers

Next: Create setup.py to fix imports, then verify actual test results
```
### Special Cases
- **Breaking changes**: Mark clearly with BREAKING: prefix and migration guidance
- **Validation completion**: Reference specific scenarios (V1-V10) and economic metrics
- **Performance improvements**: Include before/after benchmarks
- **Bug fixes**: Explain economic implications and edge cases resolved
- **Documentation updates**: Note which sections help future AI agent comprehension

This protocol ensures every commit provides maximum context for project continuity and AI agent understanding of the economic simulation research progression.
- Ready for Phase 1 implementation: equilibrium computation and analytical verification
```

### Special Cases
- **Breaking changes**: Mark clearly with BREAKING: prefix and migration guidance
- **Validation completion**: Reference specific scenarios (V1-V10) and economic metrics
- **Performance improvements**: Include before/after benchmarks
- **Bug fixes**: Explain economic implications and edge cases resolved
- **Documentation updates**: Note which sections help future AI agent comprehension

This protocol ensures every commit provides maximum context for project continuity and AI agent understanding of the economic simulation research progression.