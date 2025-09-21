# Economic Simulation Vibe: Spatial Walrasian Markets

A research-grade economic simulation platform for studying spatial frictions in market economies. This project implements agent-based modeling of economic exchange with spatial constraints, movement costs, and centralized marketplace trading.

*Forked from the original econ_sim_base project.*

## 🎯 Current Development Status

### Production-Ready Economic Engine with Spatial Infrastructure

**PHASE 1 - COMPLETE ✅**:
- **Economic Engine**: Walrasian equilibrium solver with Cobb-Douglas utilities
- **Agent Framework**: Simplified inventory management with full economic correctness
- **Market Clearing**: Constrained execution with proportional rationing
- **Test Suite**: 197/197 tests passing (185 unit + 12 validation scenarios)
- **Package Configuration**: Working setup.py and pytest.ini for development

**PHASE 2 - BASIC IMPLEMENTATION 🚧**:
- **Spatial Grid**: Complete positioning and marketplace detection
- **Agent Movement**: Simple one-step movement toward marketplace (greedy pathfinding)
- **Simulation Runner**: Functional with YAML configuration support
- **Travel Cost Integration**: Basic implementation with budget adjustment

### Implementation Status & Limitations

**✅ Complete & Functional**:
- **Economic Engine**: Core agent framework, equilibrium solver, market clearing mechanisms
- **Test Framework**: 197/197 tests passing (185 unit + 12 validation; V1–V10 plus enhanced real-function variants)
- **Simplified Inventory Management**: Agents load full home inventory at cycle start, eliminating strategic withholding complexity
- **Package Configuration**: Working setup.py, pytest.ini, requirements.txt
- **Spatial Infrastructure**: Basic grid movement and marketplace detection working
- **Travel Cost Integration**: Implemented with proper budget adjustment w_i = max(0, p·ω_total - κ·d_i)

**⚠️ Simple Implementation**:
- **Movement System**: Basic greedy movement toward marketplace (not A* pathfinding)
- **Pathfinding**: Simple one-step movement with lexicographic tie-breaking
- **Local Price Formation**: Uses global Walrasian pricing (Phase 2 uses LTE on marketplace participants)

**📋 Planned / In-Progress Advanced Features**:
- **A* Pathfinding**: Optimal pathfinding with obstacle avoidance (not yet implemented)
- **Data Persistence**: Parquet logging with schema versioning (hooks exist, not fully implemented)
- **Real-time Visualization**: pygame visualization system (basic --no-gui mode works)
- **Phase 3 Features**: Local price formation, bilateral bargaining, market microstructure
- **Financing Mode TOTAL_WEALTH**: Enum placeholder present; distinct execution semantics not yet activated

## Quick Start

### Prerequisites
- Python 3.12+
- Git

### Setup
```bash
# Clone and enter directory
git clone https://github.com/cmfunderburk/econ_sim_vibe.git
cd econ_sim_vibe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Running Tests and Simulations
```bash
# Run full test suite (197/197 tests pass - 100% success rate)
make test

# Run validation scenarios (all 12 scenarios pass)
make validate

# Run specific simulation (travel costs implemented)
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 --no-gui
```

## Research Focus

This simulation studies **spatial deadweight loss** in economic markets:

- **Research Question**: How do movement costs (κ > 0) and marketplace access restrictions quantitatively reduce allocative efficiency compared to frictionless Walrasian outcomes?
- **Key Innovation**: Local-participants equilibrium pricing with constrained execution
- **Measurement**: Money-metric welfare loss (equivalent variation in numéraire units)

### Three-Phase Development

1. **Phase 1: Pure Walrasian** - ✅ Implemented: Frictionless baseline with perfect market clearing
2. **Phase 2: Spatial Extensions** - 🚧 Partial: Basic movement, marketplace access (planned: movement costs, full spatial analysis)
3. **Phase 3: Local Price Formation** - 📋 Planned: Bilateral bargaining, spatial price variation, advanced market mechanisms

## Technical Architecture

### Core Components
- **Walrasian Solver**: Cobb-Douglas closed forms with numerical fallbacks
- **Spatial Grid**: Basic one-step movement on configurable grid (simple greedy, not A*)
- **Market Clearing**: Constrained execution with carry-over order management
- **Travel Cost System**: Budget adjustment for movement costs (implemented)
- **Welfare Measurement**: Money-metric utilities for interpersonal comparability

### Architecture Flow
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
*Key insight: Prices computed from **total endowments** but execution limited by **personal inventory***

### Key Features
- **Reproducible**: Deterministic simulations with configurable random seeds
- **Scalable**: Target: 100+ agents with <30 seconds per 1000 rounds
- **Extensible**: Plugin architecture for utility functions and movement policies
- **Research-Grade**: Parquet logging, git SHA tracking, comprehensive validation

### Financing Model
Current Phase‑2 baseline uses **personal‑inventory financing** (FinancingMode.PERSONAL):
- Agents may only finance purchases with goods they physically carried to the marketplace (personal inventory). No borrowing against home inventory.
- Value feasibility per round: p·(executed buys) ≤ p·(executed sells) + ε.
- Prices are computed from participants' total endowments (home + personal) to form the Local Theoretical Equilibrium (LTE); execution is constrained by personal inventory, generating a measurable liquidity gap.
- Travel costs reduce theoretical wealth diagnostically but do not expand financing capacity (future modes may deduct explicit numéraire units on movement).

Implemented extension introduces a `FinancingMode` toggle (regression‑guarded):
```
FinancingMode = Enum('FinancingMode', ['PERSONAL', 'TOTAL_WEALTH'])
```
Mode semantics:
- PERSONAL (default, active): Pure barter constraint with proportional rationing (enforced: p·executed_buys ≤ p·executed_sells + ε). Backward compatibility guaranteed by a regression test that compares default vs explicit PERSONAL mode.
- TOTAL_WEALTH (placeholder): Enum entry present; future behavior will size orders off total (travel‑adjusted) wealth with modified feasibility rule p·buys ≤ p·ω_total.

Rationale: Decoupling price formation (total endowment) from financing scope (personal vs total) enables experiments on liquidity frictions without altering equilibrium fundamentals.

## Simulation Protocol

Each round follows the spatial Walrasian protocol:
1. **Agent Movement**: Move toward marketplace (Manhattan/L1; tie-break lexicographic by (x,y), then agent ID)
2. **Price Discovery**: Compute Local Theoretical Equilibrium (LTE) using **post-move** marketplace participants' total endowments
3. **Order Generation**: Each marketplace agent computes buy/sell orders
4. **Order Matching**: Execute trades constrained by personal inventory with proportional rationing
5. **State Update**: Record results, update positions and carry-over queues

**Termination**: Simulation stops at T ≤ 200 rounds, when all agents reach marketplace with total unmet demand/supply below tolerance for 5 consecutive rounds, or after max_stale_rounds without meaningful progress.

### Validation Framework

The project includes **10 core validation scenarios + 2 enhanced real-function validation tests** with all tests passing:

**V1-V10 Scenarios (All Verified ✅)**:
| Scenario | Purpose | Status | Expected Outcome |
|----------|---------|---------|------------------|
| **V1: Edgeworth 2×2** | Analytic verification | ✅ PASS | `‖p_computed - p_analytic‖ < 1e-8` |
| **V2: Spatial Null** | Friction-free baseline | ✅ PASS | `efficiency_loss < 1e-10` |
| **V3: Market Access** | Spatial efficiency loss | ✅ PASS | `efficiency_loss > 0.1` |
| **V4: Throughput Cap** | Market rationing effects | ✅ PASS | `uncleared_orders > 0` |
| **V5: Spatial Dominance** | Welfare bounds | ✅ PASS | `spatial_welfare ≤ walrasian_welfare` |
| **V6: Price Normalization** | Numerical stability | ✅ PASS | `p₁ ≡ 1 and ||Z_market(p)||_∞ < 1e-8` |
| **V7: Empty Marketplace** | Edge case handling | ✅ PASS | `prices = None, trades = []` |
| **V8: Stop Conditions** | Termination logic | ✅ PASS | Proper termination detection |
| **V9: Scale Invariance** | Numerical robustness | ✅ PASS | Price scaling consistency |
| **V10: Spatial Null (Unit Test)** | Regression testing | ✅ PASS | Phase equivalence validation |

**Complete Validation Framework**:
- **197/197 tests passing** (185 unit tests + 12 validation scenarios)
- **Complete economic validation** covering all fundamental properties
- **Comprehensive edge case handling** for robust real-world deployment
- **Research-grade validation** suitable for publication-quality experiments

## Documentation

- **[SPECIFICATION.md](SPECIFICATION.md)** - Complete technical specification (825 lines)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development workflow and standards
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI development assistant configuration
- **requirements.txt** - Python dependencies
- **config/** - Validation scenarios and simulation configurations

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

## Contributing

This project provides a complete, functional development environment for economic simulation research:

1. **Working Setup**: Follow the verified setup instructions above
2. **Test Everything**: All 84 tests pass with proper package installation  
3. **Economic Validation**: Complete V1-V10 validation scenarios ready for research
4. **Development Ready**: Full economic engine with spatial extensions implemented

**Development Commands**:
```bash
make test              # ✅ WORKS: 197/197 tests passing
make validate          # ✅ WORKS: All validation scenarios pass
make format            # Format code
make check             # Quality checks (lint + test)
```

See [SPECIFICATION.md](SPECIFICATION.md) for theoretical guidelines and [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow.

## Questions? Need Help?

**Documentation & Support**:
- **[SPECIFICATION.md](SPECIFICATION.md)** - Complete technical specification (825 lines)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development workflow and standards
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for research questions and collaboration
- **AI Assistant**: Project includes comprehensive AI development instructions

## License

MIT License - See LICENSE file for details.

---

**Ready to contribute?** Follow the setup instructions above to get started with the complete economic simulation platform! 🚀