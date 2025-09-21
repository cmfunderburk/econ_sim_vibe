# Economic Simulation Vibe: Spatial Walrasian Markets

A research-grade economic simulation platform for studying spatial frictions in market economies. This project implements agent-based modeling of economic exchange with spatial constraints, movement costs, and centralized marketplace trading.

*Forked from the original econ_sim_base project.*

## � Current Development Status

### Critical Issue: Development Environment Currently Broken

**KNOWN BLOCKERS**:
- ❌ **Import System Failure**: ModuleNotFoundError prevents any tests from running
- ❌ **Missing Package Configuration**: No setup.py or proper package installation method
- ❌ **Setup Instructions Don't Work**: Fresh environment setup fails with import errors
- ❌ **Test Claims Unverifiable**: Cannot validate "84/84 tests passing" due to import failures

### Quick Reality Check

```bash
# Current setup attempts will fail:
git clone https://github.com/cmfunderburk/econ_sim_vibe.git
cd econ_sim_vibe
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# These commands fail with ModuleNotFoundError:
make test          # FAILS: ImportError: No module named 'src'
pytest tests/     # FAILS: Cannot import src.core.agent
python scripts/run_simulation.py  # FAILS: Module import errors
```

### Required Fixes for Functional Development Environment

**To make this project actually usable**:
1. **Add Package Configuration**: Create setup.py for proper package installation
2. **Fix Import System**: Enable `pip install -e .` for development imports
3. **Verify Test Claims**: Actually run tests in fresh environment before claiming success
4. **Update Setup Instructions**: Document working setup procedure

### What Code Actually Exists (Honest Assessment)

**✅ Code Files Present**:
- `src/core/agent.py` - Agent class implementation (appears complete)
- `src/econ/equilibrium.py` - Economic solver code (appears complete)
- `src/econ/market.py` - Market clearing mechanisms (appears complete)
- `tests/unit/test_components.py` - Unit test definitions (unverified due to import failures)
- `tests/validation/test_scenarios.py` - Validation scenarios (unverified due to import failures)

**❌ Functionality Status UNKNOWN**:
- Cannot verify if any tests pass due to broken import system
- Cannot run any simulation code due to missing package configuration
- Cannot validate economic claims due to non-functional development environment

**⚠️ Recent Spatial Implementation**:
- Spatial grid and simulation runner may have been implemented recently
- Cannot verify functionality until import system is fixed

## Project Status: Development Environment Broken ❌

**Reality Check**: Cannot verify any functionality claims due to broken import system.

### ❌ Critical Blockers
- **Import System**: All test commands fail with ModuleNotFoundError
- **Package Configuration**: Missing setup.py prevents proper installation
- **Setup Instructions**: Don't work in fresh environments
- **Test Verification**: Cannot run any tests to validate claims

### � What May Be Complete (Unverified)
Based on file presence but unable to verify due to import failures:
- **Economic Theory Implementation**: Code exists for Walrasian equilibrium solver
- **Agent Framework**: Code exists for agent class with inventory management
- **Test Definitions**: Test files exist but cannot execute
- **Configuration Files**: YAML configs exist but cannot load

### 🚨 Previous Documentation Claims (Unverifiable)
Earlier versions claimed:
- "84/84 tests passing" - **Cannot verify due to import failures**
- "Production-ready platform" - **Cannot verify due to setup failures**
- "Complete validation framework" - **Cannot verify due to test execution failures**
- "Comprehensive testing" - **Cannot verify any test results**
### 🚨 Next Steps to Fix Development Environment

**To make this project functional**:
1. **Create Package Configuration** (`setup.py`) for proper package installation
2. **Add Development Setup** (`pip install -e .`) to fix import paths
3. **Test in Fresh Environment** to verify actual functionality
4. **Document Working Setup** with verified instructions

### 🎯 Once Environment is Fixed
If/when import system works, need to verify and then complete:
- Validation scenarios implementation and testing
- Spatial grid and simulation runner functionality 
- End-to-end pipeline from YAML config to results
- Reproducible research experiment tools

## Quick Start (Currently Non-Functional)

### Prerequisites
- Python 3.12.3+
- Git

### Setup (Known to Fail)
```bash
# Clone and enter directory
git clone <repository-url>
cd econ_sim_vibe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (insufficient for imports)
pip install -r requirements.txt

# REQUIRED FIX (not yet implemented):
# pip install -e .  # This should work but setup.py doesn't exist
```

### Current Status Testing
```bash
# These will fail until import system is fixed:
make test          # FAILS: ImportError: No module named 'src'
make validate      # FAILS: Cannot import modules

# The following cannot be verified:
# python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42
# pytest tests/unit/ -v
# pytest tests/validation/ -v
```

## Research Focus

This simulation studies **spatial deadweight loss** in economic markets:

- **Research Question**: How do movement costs (κ > 0) and marketplace access restrictions quantitatively reduce allocative efficiency compared to frictionless Walrasian outcomes?
- **Key Innovation**: Local-participants equilibrium pricing with constrained execution
- **Measurement**: Money-metric welfare loss (equivalent variation in numéraire units)

### Three-Phase Development

1. **Phase 1: Pure Walrasian** - Frictionless baseline with perfect market clearing
2. **Phase 2: Spatial Extensions** - Movement costs, marketplace access, spatial efficiency analysis
3. **Phase 3: Local Price Formation** - Bilateral bargaining, spatial price variation, advanced market mechanisms

## Technical Architecture

### Core Components
- **Walrasian Solver**: Cobb-Douglas closed forms with numerical fallbacks
- **Spatial Grid**: Agent movement with A* pathfinding
- **Market Clearing**: Constrained execution with carry-over order management
- **Welfare Measurement**: Money-metric utilities for interpersonal comparability

### Architecture Flow
```
Home ↔ Personal ↔ Market
 ω_h     ω_p      prices
   ↘       ↓        ↓
    total_endowment → price_computation (theoretical clearing)
           ↓
    personal_inventory → execution (constrained by personal stock)
           ↓
        rationing → carry-over
```
*Key insight: Prices reflect total endowments; execution limited by personal stock*

### Key Features
- **Reproducible**: Deterministic simulations with configurable random seeds (NumPy PCG64 + random.seed() for complete coverage)
- **Scalable**: Target: 100+ agents with <30 seconds per 1000 rounds (on reference hardware, G≤5; see [performance analysis](SPECIFICATION.md#scalability-targets) for details)
- **Extensible**: Plugin architecture for utility functions and movement policies
- **Research-Grade**: Parquet logging, git SHA tracking, comprehensive validation (see [data types](SPECIFICATION.md#repository-scaffolding) in SPEC)

### Initialization Guarantees
To prevent "my p blew up" issues on first runs:
- **Interiority conditions**: Dirichlet preferences clipped at 0.05 and renormalized to sum to 1, positive supply for all goods, no zero wealth
- **Grid scaling**: ≈ 2.5√N per side for N agents, goods count G=3-5 independent of agent count  
- **Goods divisibility**: Goods are perfectly divisible (ℝ⁺), so proportional rationing introduces no rounding
- **Units & distance**: See [Numerical Constants & Units](SPECIFICATION.md#numerical-constants--units-source-of-truth) for canonical κ scaling and L1 distance conventions
- **Numerical stability**: See [SPECIFICATION.md](SPECIFICATION.md) for complete details on solver parameterization and convergence criteria
- **Tolerance constants**: SOLVER_TOL=1e-8 and FEASIBILITY_TOL=1e-10 are defined in [SPECIFICATION.md](SPECIFICATION.md) as the single source of truth

## Documentation

- **[SPECIFICATION.md](SPECIFICATION.md)** - Complete technical specification (825 lines)
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI development assistant configuration
- **requirements.txt** - Python dependencies
- **config/** - Validation scenarios and simulation configurations (when implemented)

### Logging Sign Conventions
- **`z_market[g] = demand - endowment`** (+ = excess demand)
- **`executed_net[g] = buys - sells`** (+ = net buyer)
- **Distance & units**: See [Numerical Constants & Units](SPECIFICATION.md#numerical-constants--units-source-of-truth) in SPECIFICATION.md

## Dependencies

### Core Libraries
- **numpy** - Mathematical operations and array handling
- **scipy** - Optimization (Walrasian equilibrium solver)

### Visualization (Optional)
- **pygame** - Real-time visualization of agent movement (optional import for `--no-gui` CI compatibility)

### Development Tools
- **pytest** - Testing framework
- **black** - Code formatting
- **mypy** - Type checking
- **numba** - Optional JIT compilation for performance

## Simulation Protocol

Each round follows the spatial Walrasian protocol:
1. **Agent Movement**: Move toward marketplace (Manhattan/L1; tie-break lexicographic by (x,y), then agent ID)
2. **Price Discovery**: Compute Local Theoretical Equilibrium (LTE) using **post-move** marketplace participants' total endowments (theoretical clearing; execution constrained by personal stock; track liquidity_gap[g] = z_market[g] - executed_net[g], positive = constrained by personal inventory)
3. **Order Generation**: Each marketplace agent computes buy/sell orders:
   - **Budget-Constrained Wealth**: $w_i = \max(0, p \cdot \omega_i^{\text{total}} - \kappa \cdot d_i) = \max(0, p \cdot (\omega_i^{\text{home}} + \omega_i^{\text{personal}}) - \kappa \cdot d_i)$
   - **LTE exclusion**: Agents with $p \cdot \omega_i^{\text{total}} \leq \epsilon$ are excluded from LTE computation to avoid singular Jacobians (travel costs do not affect pricing participation)
   - **Order quantity**: $\Delta_i = x_i^*(p,w_i) - \omega_i^{\text{personal}}$ (positive = buy, negative = sell)
4. **Order Matching**: Execute trades constrained by personal inventory with proportional rationing
5. **State Update**: Record results, update positions and carry-over queues

```python
# Extract agents physically inside the 2×2 marketplace
# CRITICAL: Use post-move positions as the snapshot for market_agents
# to ensure price computation matches the participant set for this round
market_agents = grid.get_agents_in_marketplace()
n_goods = market_agents[0].alpha.size if market_agents else 0

# Edge-gates prevent partial pricing in Phase-2
if not market_agents or len(market_agents) < 2 or n_goods < 2:
    prices, trades = None, []
else:
    prices, z_rest_inf, walras_dot, status = solve_equilibrium(market_agents, normalization="good_1", endowment_scope="total")
    trades = execute_constrained_clearing(market_agents, prices)
```

**Termination**: Simulation stops at T ≤ 200 rounds, when all agents reach marketplace with total unmet demand/supply below `RATIONING_EPS` for 5 consecutive rounds, or after `max_stale_rounds` rounds without meaningful progress (default: 50). Log `termination_reason` as "horizon", "market_cleared", or "stale_progress".

## Validation Framework (Cannot Be Verified - Import System Broken)

The project claims **10 comprehensive validation scenarios** but import failures prevent verification:

**V1-V10 Scenarios (Cannot Verify - Import Failures Prevent Testing)**:
| Scenario | Purpose | Status | Expected Outcome |
|----------|---------|---------|------------------|
| **V1: Edgeworth 2×2** | Analytic verification | ❌ UNTESTABLE | `‖p_computed - p_analytic‖ < 1e-8` |
| **V2: Spatial Null** | Friction-free baseline | ❌ UNTESTABLE | `efficiency_loss < 1e-10` |
| **V3: Market Access** | Spatial efficiency loss | ❌ UNTESTABLE | `efficiency_loss > 0.1` |
| **V4: Throughput Cap** | Market rationing effects | ❌ UNTESTABLE | `uncleared_orders > 0` |
| **V5: Spatial Dominance** | Welfare bounds | ❌ UNTESTABLE | `spatial_welfare ≤ walrasian_welfare` |
| **V6: Price Normalization** | Numerical stability | ❌ UNTESTABLE | `p₁ ≡ 1 and ||Z_market(p)||_∞ < 1e-8` |
| **V7: Empty Marketplace** | Edge case handling | ❌ UNTESTABLE | `prices = None, trades = []` |
| **V8: Stop Conditions** | Termination logic | ❌ UNTESTABLE | Proper termination detection |
| **V9: Scale Invariance** | Numerical robustness | ❌ UNTESTABLE | Price scaling consistency |
| **V10: Spatial Null (Unit Test)** | Regression testing | ❌ UNTESTABLE | Phase equivalence validation |

**Validation Status Reality Check**:
- **Import failures prevent any test execution** - cannot verify if any tests pass
- **Setup instructions don't work** in fresh environments due to missing package configuration
- **Test files exist but cannot run** due to ModuleNotFoundError in import system
- **All functionality claims unverifiable** until import system is fixed

## Contributing

This project requires **immediate environment fixes** before any meaningful development:

1. **Fix Import System**: Create setup.py and enable `pip install -e .` for development
2. **Verify Tests Actually Work**: Run tests in fresh environment to validate claims  
3. **Document Working Setup**: Update instructions with actual working procedure
4. **Economic Development**: Once environment works, continue with economic validation

**Current Reality**: Cannot run any tests or verify any functionality due to broken import system.

See [SPECIFICATION.md](SPECIFICATION.md) for theoretical guidelines, but actual implementation must wait for functional development environment.

## Questions? Need Help?

**For New Contributors**:
1. **CURRENT BLOCKER**: Cannot run `make test` due to import system failures
2. **Cannot Explore**: Import errors prevent loading any source files  
3. **Cannot Validate**: All test execution fails with ModuleNotFoundError

**Next Steps**:
1. **Fix Development Environment**: Add missing setup.py and package configuration
2. **Test in Fresh Environment**: Verify setup instructions actually work
3. **Validate Functionality**: Once imports work, verify if claimed features exist
**Documentation & Support**:
- **[Human Summary](copilot_summaries/Human%20Summary)** - Quick contributor orientation guide  
- **[SPECIFICATION.md](SPECIFICATION.md)** - Complete technical specification (825 lines)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for research questions and collaboration
- **AI Assistant**: Project includes comprehensive AI development instructions

**Development Commands (Will Fail Until Import System Fixed)**:
```bash
# THESE CURRENTLY FAIL - Import system must be fixed first:
make test              # FAILS: ImportError: No module named 'src'
make validate          # FAILS: Cannot import modules
make format            # Format code
make check             # Quality checks (lint + test)
```

## License

MIT License - See LICENSE file for details.

---

**Ready to contribute?** **FIRST**: Fix the import system by creating setup.py and enabling `pip install -e .`, then verify tests actually work in a fresh environment. Only then can we validate the economic implementation! �