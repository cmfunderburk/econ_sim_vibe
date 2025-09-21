# Economic Simulation Vibe Project - AI Coding Instructions

## Project Overview
This is a research-grade economic simulation implementing agent-based modeling with spatial frictions in market economies. The simulation models rational agents trading goods on a configurable spatial grid with centralized marketplace access and movement costs.

*This project is forked from econ_sim_base with enhanced "vibe" features and customizations.*

**📋 Key References**: 
- [SPECIFICATION.md](../SPECIFICATION.md) for complete technical specification (825+ lines)
- [CONTRIBUTING.md](../CONTRIBUTING.md) for development standards and workflows
- [README.md](../README.md) for project overview and quick-start guide

## Current Project Status ✅ AGENT FRAMEWORK COMPLETE, EQUILIBRIUM SOLVER NEXT
- ✅ **Specification Phase Complete**: Bulletproof technical design with configurable marketplace
- ✅ **Developer Tooling Complete**: Comprehensive development environment and standards
- ✅ **Project Scaffolding Complete**: All module stubs, configs, and test structure ready
- ✅ **Agent Framework Complete**: Production-ready Agent class with comprehensive testing (Sep 20, 2025)
- 🔄 **Phase 1 Implementation Active**: Equilibrium solver and market clearing needed
- 📋 **Immediate Priority**: Walrasian equilibrium solver implementation
- 🎯 **Implementation Goal**: Working Phase 1 baseline with validation scenarios V1-V2

### What Exists vs What's Needed
**✅ Complete & Ready:**
- Module structure (`src/core/`, `src/econ/`, `src/spatial/`) with detailed docstrings
- Configuration files for 10 validation scenarios (V1-V10) in `config/`
- Testing framework with comprehensive test suite in `tests/unit/` and `tests/validation/`
- Economic theory specification and mathematical foundations in SPECIFICATION.md
- Development tooling (Makefile, linting, formatting, CI setup)
- **AGENT FRAMEWORK COMPLETE**: `src/core/agent.py` with Cobb-Douglas utilities, inventory management, spatial positioning
- **COMPREHENSIVE TESTING**: 15 unit tests, edge cases, specification compliance, integration tests, performance validation
- Implementation summaries tracking in `copilot_summaries/Implementation Summaries`

**🔄 Implementation Needed:**
- Walrasian equilibrium solver with numéraire normalization (`src/econ/equilibrium.py`)
- Market clearing mechanisms with economic invariants (`src/econ/market.py`) 
- Grid and movement implementation (`src/spatial/`)
- Validation test implementations for scenarios V1-V2 (`tests/validation/`)
- Configuration loading and simulation engine (`scripts/run_simulation.py`)

## Phase 1 Implementation Priorities 🎯

### Immediate Implementation Tasks
1. **✅ Agent Framework COMPLETE** (`src/core/agent.py`)
   - ✅ Agent class with Cobb-Douglas utility functions
   - ✅ Home and personal inventory management
   - ✅ Position tracking and market access detection
   - ✅ Comprehensive testing suite (15 unit tests, edge cases, integration, performance)
   - ✅ All economic invariants validated (conservation, budget constraints, Walras' Law)
   - ✅ Performance targets met (3.12μs per demand calculation, 8.71KB per agent)

2. **🔄 Economic Engine NEXT** (`src/econ/equilibrium.py`)
   - Walrasian equilibrium solver with numéraire normalization (p₁ ≡ 1)
   - Closed-form Cobb-Douglas demand functions
   - Excess demand computation for marketplace participants

3. **Market Clearing** (`src/econ/market.py`)
   - Pure exchange clearing (Phase 1: no spatial constraints yet)
   - Economic invariant validation (Walras' Law, conservation)
   - Trade execution and state updates

4. **Validation Implementation** (`tests/validation/test_scenarios.py`)
   - V1: Edgeworth box 2×2 analytical verification
   - V2: Spatial null test (κ=0 should equal Phase 1)
   - Economic invariant checking framework

### Mathematical Foundations (Implementation Ready)

#### Cobb-Douglas Implementation Pattern
```python
class CobbDouglasAgent:
    """Agent with Cobb-Douglas utility: U(x) = ∏_j x_j^α_j where ∑_j α_j = 1"""
    
    def __init__(self, alpha: np.ndarray, endowment: np.ndarray):
        self.alpha = alpha / np.sum(alpha)  # Ensure normalization
        self.total_endowment = endowment.copy()
        
    def demand(self, prices: np.ndarray) -> np.ndarray:
        """Optimal demand: x_j = α_j * wealth / p_j"""
        wealth = np.dot(prices, self.total_endowment)
        return self.alpha * wealth / prices
        
    def utility(self, consumption: np.ndarray) -> float:
        """Cobb-Douglas utility with safety guards"""
        consumption = np.maximum(consumption, 1e-10)  # Avoid log(0)
        return np.prod(consumption ** self.alpha)
```

#### Equilibrium Solver Pattern
```python
def solve_walrasian_equilibrium(agents: List[Agent]) -> Tuple[np.ndarray, float]:
    """Solve for market-clearing prices with numéraire normalization"""
    n_goods = agents[0].alpha.size
    
    def excess_demand_rest_goods(p_rest: np.ndarray) -> np.ndarray:
        """Excess demand for goods 2,...,n (good 1 is numéraire)"""
        prices = np.concatenate([[1.0], p_rest])  # p₁ ≡ 1
        total_demand = np.zeros(n_goods)
        total_endowment = np.zeros(n_goods)
        
        for agent in agents:
            demand = agent.demand(prices)
            total_demand += demand
            total_endowment += agent.total_endowment
            
        excess_demand = total_demand - total_endowment
        return excess_demand[1:]  # Return only rest goods (exclude numéraire)
    
    # Solve using scipy
    p_rest_initial = np.ones(n_goods - 1)  # Initial guess
    p_rest_solution = scipy.optimize.fsolve(excess_demand_rest_goods, p_rest_initial)
    prices = np.concatenate([[1.0], p_rest_solution])
    
    # Compute convergence metric
    z_rest_norm = np.linalg.norm(excess_demand_rest_goods(p_rest_solution), ord=np.inf)
    
    return prices, z_rest_norm
```

## Critical Economic Invariants 🚨

All implementations MUST preserve these invariants (failing these breaks the simulation):

```python
# 1. Conservation Law: Goods cannot be created or destroyed
def test_conservation(agents_before: List[Agent], agents_after: List[Agent]):
    total_before = sum(agent.total_endowment for agent in agents_before)
    total_after = sum(agent.total_endowment for agent in agents_after)
    assert np.allclose(total_before, total_after, atol=1e-10)

# 2. Walras' Law: Price vector dot excess demand must equal zero
def test_walras_law(prices: np.ndarray, excess_demand: np.ndarray):
    walras_dot = np.dot(prices, excess_demand)
    assert abs(walras_dot) < 1e-8, f"Walras' Law violated: {walras_dot}"

# 3. Numéraire Constraint: Good 1 price always equals 1
def test_numeraire(prices: np.ndarray):
    assert abs(prices[0] - 1.0) < 1e-12, f"Numéraire violated: p[0]={prices[0]}"

# 4. Market Clearing: Primary convergence test
def test_market_clearing(excess_demand: np.ndarray):
    z_rest_norm = np.linalg.norm(excess_demand[1:], ord=np.inf)
    assert z_rest_norm < 1e-8, f"Poor convergence: ||Z_rest||_∞ = {z_rest_norm}"

# 5. Non-negativity: All consumption bundles must be non-negative
def test_nonnegativity(consumption: np.ndarray):
    assert np.all(consumption >= -1e-10), f"Negative consumption: {consumption}"
```

## Implementation Patterns & Standards

### File Organization
```
src/
├── core/
│   ├── agent.py          # ✅ COMPLETE: Agent class with Cobb-Douglas utilities
│   ├── simulation.py     # SimulationEngine and round management
│   └── types.py          # Trade, SimulationState, core dataclasses
├── econ/
│   ├── equilibrium.py    # 🔄 NEXT: Walrasian solver with analytical forms
│   ├── market.py         # Market clearing and trade execution
│   └── welfare.py        # Utility measurement and welfare analysis
├── spatial/              # Phase 2: spatial extensions (deferred)
│   ├── grid.py          # Spatial grid and marketplace detection
│   └── movement.py      # A* pathfinding and movement costs
└── config/
    └── loader.py        # YAML configuration loading
```

### Development Workflow & Commands (Essential for Implementation)

```bash
# Primary development commands (use these!)
make install-dev       # Setup virtual environment with all dependencies
make test             # Run full test suite with economic invariants
make format           # Auto-format code (black, ruff) - run before commits
make run CONFIG=config/edgeworth.yaml SEED=42  # Test simulation

# Implementation workflow
pytest tests/unit/ -v          # Test individual components
pytest tests/validation/ -v    # Economic validation scenarios V1-V10

# Critical: All commands set deterministic environment
# OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 (prevents solver jitter)
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
- **Working code base**: Agent framework complete, equilibrium solver needed next
- **Mathematical foundation**: Complete and validated in SPECIFICATION.md
- **Testing framework**: Comprehensive suite implemented, scenarios V1-V2 are priority
- **Configuration**: YAML configs exist, need runtime loading implementation
- **Development tools**: Complete (Makefile, requirements, formatting, etc.)
- **Recent completion**: Agent class with full testing validation (Sep 20, 2025)

### Implementation Summary Protocol
When completing major implementation tasks, append summaries to `copilot_summaries/Implementation Summaries` using this format:

```
================================================================================
DATE: [Current Date]
TITLE: [Brief Descriptive Title]
================================================================================
[Detailed summary content including:
- What was implemented
- Testing results
- Performance metrics
- Files created/modified
- Next steps]
================================================================================
```

This creates a historical record of implementation progress and achievements for future reference and project continuity.

### Common Requests & Responses
- **"Implement equilibrium solver"** → Focus on Cobb-Douglas with numéraire normalization
- **"Add spatial features"** → Defer to Phase 2, complete Phase 1 first
- **"Fix economic bug"** → Always validate against economic theory in SPECIFICATION.md
- **"Add new utility function"** → Ensure plugin architecture compatibility
- **"Optimize performance"** → Vectorize with numpy, avoid premature optimization

This project represents cutting-edge research in computational economics. Every line of code should reflect the highest standards of both software engineering and economic theory.
- **Research integration**: Design for testing economic hypotheses and policy experiments