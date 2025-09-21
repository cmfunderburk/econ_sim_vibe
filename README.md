# Economic Simulation Vibe: Spatial Walrasian Markets

A research-grade economic simulation platform for studying spatial frictions in market economies. This project implements agent-based modeling of economic exchange with spatial constraints, movement costs, and centralized marketplace trading.

*Forked from the original econ_sim_base project.*

## 🎯 Current Development Status

### Production-Ready Economic Engine with Spatial Infrastructure

**PHASE 1 - COMPLETE ✅**:
- **Economic Engine**: Walrasian equilibrium solver with Cobb-Douglas utilities
- **Agent Framework**: Simplified inventory management with full economic correctness
- **Market Clearing**: Constrained execution with proportional rationing
- **Test Suite**: 217/217 tests passing (205 unit + 12 validation scenarios)
- **Package Configuration**: Working setup.py and pytest.ini for development

**PHASE 2 - BASIC IMPLEMENTATION 🚧**:
- **Spatial Grid**: Complete positioning and marketplace detection
- **Agent Movement**: Simple one-step movement toward marketplace (greedy pathfinding)
- **Simulation Runner**: Functional with YAML configuration support
- **Travel Cost Integration**: Basic implementation with budget adjustment

### Implementation Status & Limitations

**✅ Complete & Functional**:
- **Economic Engine**: Core agent framework, equilibrium solver, market clearing mechanisms
- **Test Framework**: 217/217 tests passing (205 unit + 12 validation; V1–V10 plus enhanced real-function variants)
- **Simplified Inventory Management**: Agents load full home inventory at cycle start, eliminating strategic withholding complexity
- **Package Configuration**: Working setup.py, pytest.ini, requirements.txt
- **Spatial Infrastructure**: Basic grid movement and marketplace detection working
- **Travel Cost Integration**: Implemented with proper budget adjustment w_i = max(0, p·ω_total - κ·d_i)

**⚠️ Simple Implementation**:
- **Movement System**: Basic greedy movement toward marketplace (not A* pathfinding)
- **Pathfinding**: Simple one-step movement with lexicographic tie-breaking
- **Local Price Formation**: Uses global Walrasian pricing (Phase 2 uses LTE on marketplace participants)

**📋 Planned / In-Progress Advanced Features**:
## ✨ Recent Enhancements (2025-09-21)
Robustness and observability improvements have been added since the initial Phase 2 baseline:

- Solver fallback: Adaptive tâtonnement engages automatically if primary `fsolve` convergence is poor, guaranteeing non-worsening residuals.
- Enhanced invariants: Movement monotonicity, randomized conservation fuzz, per-agent value feasibility proxy.
- Structured logging hardening: Canonical schema guard test ensures field stability (`SCHEMA_VERSION=1.1.0`). Added enriched per-agent diagnostics (requested/executed buys & sells, unmet components, fill rates) as additive fields.
- Compression support: Optional gzip for JSONL and Parquet outputs via `RunLogger(compress=True)`.
- Financing mode tagging: All records now populate `financing_mode="PERSONAL"` (foundation for future multi-mode analysis).
- Configuration validation: Early aggregated validation (`validate_simulation_config`) catches invalid grid / marketplace / agent parameter combinations.

Planned follow-ups: effective budget logging, warm-started solver hints, CLI flag for compressed logging, console entry points, and performance benchmark harness.

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
# Run full test suite (217/217 tests pass - 100% success rate)
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
       - Schema safety net: Logging schema guard test prevents silent breaking changes.
       - Config safety net: Early validation rejects inconsistent or degenerate scenarios.

### Financing Model (Summary)
Personal-inventory financing enforced (PERSONAL mode). See `docs/STATUS.md` for mode semantics and future TOTAL_WEALTH notes.

## Simulation Protocol

Each round follows the spatial Walrasian protocol:
1. **Agent Movement**: Move toward marketplace (Manhattan/L1; tie-break lexicographic by (x,y), then agent ID)
2. **Price Discovery**: Compute Local Theoretical Equilibrium (LTE) using **post-move** marketplace participants' total endowments
3. **Order Generation**: Each marketplace agent computes buy/sell orders
4. **Order Matching**: Execute trades constrained by personal inventory with proportional rationing
5. **State Update**: Record results, update positions and carry-over queues

**Termination**: Simulation stops at T ≤ 200 rounds, when all agents reach marketplace with total unmet demand/supply below tolerance for 5 consecutive rounds, or after max_stale_rounds without meaningful progress.

### Validation Framework (Summary)
217/217 tests passing (205 unit + 12 validation). Scenario descriptions and expected metrics live in `docs/STATUS.md` and full economic detail in `SPECIFICATION.md`.

## Documentation Index
Primary references:
- `docs/STATUS.md` (current state)
- `docs/ROADMAP.md` (priorities & backlog)
- `docs/DEVELOPER_GUIDE.md` (developer workflow)
- `SPECIFICATION.md` (theory & algorithms)

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
See `docs/DEVELOPER_GUIDE.md` for setup & commands, and `CONTRIBUTING.md` for contribution standards.

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