# Economic Simulation Vibe - AI Development Instructions

## 📋 Project Overview
This is a research-grade economic simulation platform implementing agent-based modeling with spatial frictions. The platform studies **spatial deadweight loss** in market economies using Walrasian equilibrium theory with spatial extensions.

**Key References:**
- [SPECIFICATION.md](../SPECIFICATION.md) - Complete technical specification (825+ lines)
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development standards and workflows
- [README.md](../README.md) - Project overview and current implementation status

## 🎯 Current Development Status

### ✅ Phase 1: Production-Ready Economic Engine
**COMPLETE & VERIFIED (179/191 tests passing - 93.5%):**
- **Walrasian Solver**: Cobb-Douglas closed forms with numerical fallbacks
- **Agent Framework**: Complete inventory management (home/personal split)
- **Market Clearing**: Constrained execution with proportional rationing
- **Validation Suite**: V1-V10 scenarios covering all economic properties
- **Package Setup**: Working `setup.py`, `pytest.ini`, development environment

### ✅ Phase 2: Basic Spatial Implementation  
**FUNCTIONAL WITH SIMPLE MOVEMENT:**
- **Basic Movement**: Simple one-step toward marketplace (not A* pathfinding)
- **Spatial Grid**: Position tracking and marketplace detection working
- **Simulation Runner**: Works with YAML configurations and travel cost integration
- **Travel Cost Integration**: Actually implemented with proper budget adjustment

### ⚠️ Known Implementation Limitations
**Documented Gaps:**
1. **Movement Algorithm**: Simple greedy movement only (no A* pathfinding implemented)
2. **Test Status**: 12 tests fail expecting unlimited credit behavior (correctly failing)
3. **Advanced Features**: No Parquet logging or pygame visualization implemented
4. **Documentation Accuracy**: Some docs claim features not actually implemented

## 🏗️ Architecture & Data Flow

### Core Architecture
```
Home ↔ Personal ↔ Market
 ω_h     ω_p      prices
   ↘       ↓        ↓
    total_endowment → price_computation (LTE from marketplace participants)
           ↓
    personal_inventory → execution (constrained by personal stock)
           ↓
        rationing → carry-over orders
```

**Key Insight**: Prices computed from **total endowments** but execution limited by **personal inventory**.

### File Organization
```
src/
├── constants.py          # SOLVER_TOL, FEASIBILITY_TOL, numerical constants
├── core/
│   ├── agent.py          # Agent class with Cobb-Douglas utilities  
│   └── types.py          # Trade, SimulationState, core dataclasses
├── econ/
│   ├── equilibrium.py    # solve_walrasian_equilibrium() 
│   └── market.py         # execute_constrained_clearing()
└── spatial/
    └── grid.py          # Grid, Position, basic movement
```

### Critical Data Flows
1. **Agent Movement**: `agent.move_one_step_toward_marketplace()` → position update
2. **Price Discovery**: `solve_walrasian_equilibrium(marketplace_agents)` → price vector
3. **Order Generation**: `agent.compute_optimal_trade(prices)` → buy/sell orders  
4. **Market Clearing**: `execute_constrained_clearing(orders)` → executed trades
5. **State Update**: Apply trades, update inventories, record results

## 🛠️ Development Workflow

### Setup & Environment
```bash
# Required setup (critical for tests)
pip install -e .          # Must install package in editable mode
source venv/bin/activate   # Virtual environment required

# Verified working commands:
make test                 # 179/191 tests pass (~93.5% success rate)
make validate            # V1-V10 scenarios pass 
make format              # black + isort code formatting
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 --no-gui  # Full simulation with travel costs
```

### Development Commands
```bash
# Testing (all verified working)
pytest tests/unit/ -v                    # Unit tests (mix of passing and documented failures)
pytest tests/validation/ -v              # 10 validation scenarios (all pass)
python scripts/validate_scenario.py V1   # Individual scenario testing

# Quality checks
make lint                # flake8 + mypy type checking
make check              # Full quality pipeline
```

## 🧮 Economic Implementation Patterns

### Critical Constants (Use These Exact Values)
```python
# From src/constants.py - DO NOT MODIFY
SOLVER_TOL = 1e-8        # Primary convergence: ||Z_rest||_∞ < SOLVER_TOL
FEASIBILITY_TOL = 1e-10  # Conservation checks
RATIONING_EPS = 1e-10    # Prevent division by zero
NUMERAIRE_GOOD = 0       # Good 1 is numéraire (p[0] ≡ 1.0)
MIN_ALPHA = 0.05         # Minimum preference weight
```

### Economic Correctness Patterns
```python
# CORRECT: Primary convergence test
z_rest_norm = np.linalg.norm(excess_demand[1:], ord=np.inf)
converged = z_rest_norm < SOLVER_TOL  # ✅ Proper test

# WRONG: Using Walras' Law for convergence  
converged = abs(np.dot(prices, excess_demand)) < tolerance  # ❌ Misleading

# CORRECT: Numéraire normalization
def excess_demand_rest(p_rest):
    prices = np.concatenate([[1.0], p_rest])  # p₁ ≡ 1
    return excess_demand(prices)[1:]  # Return rest goods only

# CORRECT: Conservation validation in every trade function
def transfer_goods(agent: Agent, goods: np.ndarray):
    initial_total = agent.home_endowment + agent.personal_endowment
    # ... perform transfer ...
    final_total = agent.home_endowment + agent.personal_endowment
    assert np.allclose(initial_total, final_total, atol=FEASIBILITY_TOL)
```

### Cobb-Douglas Implementation
```python
# Agent utility: U(x) = ∏_j x_j^α_j where ∑α_j = 1
def utility(self, consumption: np.ndarray) -> float:
    """Cobb-Douglas utility with numerical safety"""
    consumption = np.maximum(consumption, MIN_ALPHA * 1e-6)  # Prevent log(0)
    return np.exp(np.sum(self.alpha * np.log(consumption)))

# Optimal demand: x_j = (α_j / p_j) * (p · ω)  
def compute_optimal_demand(self, prices: np.ndarray) -> np.ndarray:
    wealth = np.dot(prices, self.total_endowment)
    return (self.alpha / prices) * wealth
```

## 🧪 Validation Framework

### V1-V10 Scenarios (All Must Pass)
```python
# V1: Edgeworth 2×2 - Analytical verification
# Expected: p* = [1, 6/7], allocation error < 1e-8

# V2: Spatial Null - Phase equivalence  
# Expected: spatial_welfare == walrasian_welfare when κ=0

# V3: Market Access - Spatial efficiency loss
# Expected: efficiency_loss > 0.1 with movement restrictions  

# V5: Spatial Dominance - Welfare bounds
# Expected: spatial_welfare ≤ walrasian_welfare always
```

### Test Patterns for New Features
```python
def test_new_feature():
    """Template for economic validation"""
    # 1. Setup agents with known preferences/endowments
    agents = create_test_agents()
    
    # 2. Run simulation or economic computation
    result = run_economic_computation(agents)
    
    # 3. Validate economic properties
    assert_conservation_laws(result)
    assert_feasibility_constraints(result)
    assert_economic_theory_prediction(result)
    
    # 4. Check numerical precision
    assert result.convergence_norm < SOLVER_TOL
```

## 🚨 Priority Implementation Gaps

### Immediate Fixes Required (High Impact)
```python
# 1. Travel cost deduction (scripts/run_simulation.py:141)
# CURRENT: distance_moved = agent.move_one_step_toward_marketplace()
# MISSING: agent.wealth -= movement_cost * distance_moved

# 2. Budget-constrained orders (src/econ/market.py)
# CURRENT: Uses personal inventory only
# NEEDED: w_i = max(0, p·ω_total - κ·d_i) for travel-adjusted budget

# 3. Position unification 
# CONFLICT: Two incompatible Position classes
# SOLUTION: Use src/spatial/grid.py as canonical, remove src/core/types.py version
```

### Priority Implementation Gaps

### Immediate Fixes Required (High Impact)
```python
# 1. Budget-constrained orders (src/econ/market.py)
# CURRENT: Uses personal inventory only
# NEEDED: w_i = max(0, p·ω_total - κ·d_i) for travel-adjusted budget

# 2. Position unification 
# CONFLICT: Two incompatible Position classes
# SOLUTION: Use src/spatial/grid.py as canonical, remove src/core/types.py version

# 3. Advanced pathfinding implementation
# CURRENT: Simple greedy movement only
# NEEDED: A* algorithm for optimal pathfinding (if required by research)
```

### Movement Policy Decision
**Current State**: Single greedy step toward marketplace  
**Documentation Claims**: "myopic A* pathfinding"

**Options for AI agents:**
- **Option A**: Implement full A* (2-3 hours, research-grade pathfinding)
- **Option B**: Update docs to match greedy implementation (15 minutes)
- **Option C**: Keep simple movement, focus on Phase 1 research applications

## 🤖 AI Agent Guidelines

### When Working on Economic Features
1. **Economic correctness first** - Validate all theoretical properties
2. **Use analytical benchmarks** - V1 Edgeworth box provides exact solutions  
3. **Preserve invariants** - Conservation, feasibility, and numéraire constraints
4. **Follow mathematical notation** - Use SPECIFICATION.md symbols (Z_market, not Z)
5. **Test comprehensively** - Unit tests + validation scenarios + edge cases

### When Implementing Spatial Features
**Priority Order:**
1. Travel cost integration (core spatial friction)
2. Position type unification (architectural cleanup)  
3. Movement policy decision (A* vs documentation alignment)
4. Advanced spatial features (only after core gaps resolved)

### Code Quality Standards
- **Type hints**: All functions must have proper type annotations
- **Docstrings**: Include mathematical formulas and economic interpretation  
- **Error handling**: Graceful degradation with informative messages
- **Performance**: Vectorized numpy operations for scalability

### Integration Points to Understand
- **Agent-Market Interface**: Personal inventory limits vs total endowment pricing
- **Spatial-Economic Integration**: Travel costs affect budgets but prices computed globally
- **Configuration Loading**: YAML configs drive simulation parameters
- **Test-Code Coupling**: V1-V10 scenarios validate specific economic properties

## 📝 Summary Documentation Protocol

When completing significant work, optionally add to `copilot_summaries/`:
```
Date: [Current Date]  
Title: [Brief Description]

Changes:
- What was implemented/fixed
- Files affected
- Economic properties validated

Status:
- Current implementation state
- Known limitations  
- Next priorities
```

This ensures continuity for future AI agents and maintains project knowledge base.
