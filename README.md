# Economic Simulation Vibe: Spatial Walrasian Markets

A research-grade economic simulation platform for studying spatial frictions in market economies. This project implements agent-based modeling of economic exchange with spatial constraints, movement costs, and centralized marketplace trading.
# Economic Simulation Vibe: Spatial Walrasian Markets

 **Test Suite**: 250/250 tests passing (unit + validation + replay, spatial fidelity, HUD & playback controls)
## 🎯 Current Development Status

- **Test Suite**: 247/247 tests passing (unit + validation + replay & spatial fidelity)
- **Package Configuration**: Working setup.py and pytest.ini for development

**PHASE 2 - BASIC IMPLEMENTATION (CORE VISUAL + LOGGING COMPLETE) ✅**:
 **Test Framework**: 250/250 tests passing (V1–V10 + logging schema guard, replay parity, spatial distance fidelity, frame hash & HUD regression tests)
- **Simulation Runner**: YAML-driven; supports deterministic replay (`--replay`)
- **Travel Cost Integration**: Budget-side deduction active
- **Visualization**: Pygame HUD + ASCII renderer + snapshot support

**✅ Complete & Functional**:
- **Economic Engine**: Core agent framework, equilibrium solver, market clearing mechanisms
- **Test Framework**: 247/247 tests passing (V1–V10 + logging schema guard, replay parity, spatial distance fidelity tests)
- **Simplified Inventory Management**: Agents load full home inventory at cycle start, eliminating strategic withholding complexity
- **Movement System**: Greedy movement (A* not yet implemented; docs aligned)
- **Local Price Formation**: Uses Walrasian LTE (participants-only) pricing

**📋 Planned / In-Progress Advanced Features**:
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
# Run full test suite (250/250 tests pass - 100% success rate)
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
# Run validation scenarios (all 12 scenarios pass)
make validate

# Run specific simulation (travel costs implemented)
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 --no-gui
```

## Research Focus

This simulation studies **spatial deadweight loss** in economic markets:
### Three-Phase Development

1. **Phase 1: Pure Walrasian** - ✅ Implemented
2. **Phase 2: Spatial Extensions** - ✅ Core spatial movement, logging, visualization implemented
3. **Phase 3: Local Price Formation** - 📋 Planned

## Technical Architecture

### Core Components
- **Walrasian Solver**: Cobb-Douglas closed forms with numerical fallbacks
- **Spatial Grid**: Basic one-step movement on configurable grid (simple greedy, not A*)
- **Market Clearing**: Constrained execution with carry-over order management
- **Travel Cost System**: Budget adjustment for movement costs (implemented)
- **Welfare Measurement**: Money-metric utilities for interpersonal comparability

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
247/247 tests passing (unit + validation + logging/replay integrity). Scenario descriptions and expected metrics live in `docs/STATUS.md` and full economic detail in `SPECIFICATION.md`.

### Playback Controls (GUI)
Space: play/pause | Right Arrow: step forward | Up/Down: speed ± (doubling/halving) | Esc/q: quit.
HUD displays current status (PLAY/PAUSE/LIVE), speed (r/s), convergence index, solver residual, fill & unmet shares, distances, efficiency, and a digest snippet.

### Known Numerical Notes
Adaptive solver fallback (tâtonnement) engages when `fsolve` convergence is poor. You may see informational warnings about fallback activation; final reported `resid` (rest-goods norm) in HUD should satisfy `||Z_rest||_∞ < 1e-8` after fallback.

## Logging & Replay Overview
Per-round JSONL (or Parquet if optional deps installed) records one row per agent per round. A geometry sidecar (market bounds, grid size, movement policy) plus deterministic frame hash supports spatial & HUD fidelity replay. An integrity digest (sha256) validates price path & participant identity sets.

Optional Parquet engine install:
```bash
pip install pyarrow  # or fastparquet
```

Replay example:
```bash
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 --output runs/example
python scripts/run_simulation.py --replay runs/example/Edgeworth\ Economy_seed42_round_log.jsonl --gui
```

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