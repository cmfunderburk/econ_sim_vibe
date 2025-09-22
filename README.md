# Economic Simulation Vibe: Spatial Walrasian Markets

Research-grade agent-based economic simulation platform for studying spatial frictions, movement costs, and centralized marketplace trading under a spatial Walrasian protocol.

**Test Suite:** 250/250 tests passing (unit + validation + replay, spatial fidelity, HUD & playback controls)

## 🎯 Current Development Status

### Phase Summary
- **Phase 1 (Pure Walrasian)**: COMPLETE – equilibrium solver (Cobb-Douglas), conservation & welfare validation.
- **Phase 2 (Spatial Extensions)**: CORE IMPLEMENTATION COMPLETE – movement, travel costs, visualization, structured logging, replay integrity.

### Core Capabilities (✅)
- Economic engine: Walrasian LTE pricing (participants’ total endowments) with constrained execution (personal inventory financing).
- Market clearing: Proportional rationing; conservation & value-feasibility invariants enforced.
- Travel cost integration: Budget-side deduction w_i = max(0, p·ω_total − κ·d_i).
- Movement: Deterministic greedy Manhattan step (A* planned).
- Structured logging: JSONL + geometry sidecar + integrity digest + schema guard (1.3.0).
- Replay & HUD: Frame hash, convergence index, clearing efficiency, unmet shares.
- Rationing diagnostics: Per-agent unmet buys/sells, fill rates, liquidity gaps.

### Enhanced Instrumentation
- Geometry sidecar for reproducible spatial reconstruction.
- Frame digest & hashing for regression protection.
- Config validation preflight (grid/marketplace/agent consistency).
- Financing mode enum scaffold (`PERSONAL` active, `TOTAL_WEALTH` placeholder).

### In Progress / Planned (📋)
- A* pathfinding (movement optimality under static additive costs).
- TOTAL_WEALTH financing semantics & comparative liquidity regime analysis.
- Performance harness & warm-start heuristics for multi-round solver reuse.
- Optional compressed logging & CLI convenience flags.

### Deferred / Future (🚧 Phase 3+)
- Local price formation (bargaining / order book microstructure).
- Spatial price dispersion & arbitrage dynamics.
- Production, credit, institutions, and behavioral extensions.

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
1. Phase 1: Pure Walrasian (complete)
2. Phase 2: Spatial Extensions (current baseline + instrumentation)
3. Phase 3: Local Price Formation (planned)

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
- Deterministic: Seeded RNG, geometry sidecar, frame hash
- Scalable target: 100+ agents (<30s per 1000 rounds typical goal)
- Extensible: Utility / movement policies / financing modes
- Research-grade telemetry: Schema guard, integrity digest, replay parity tests
- Config safety: Early validation rejects inconsistent scenarios

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

### Validation Framework
250/250 tests passing (unit + validation + replay + visualization). Economic theory invariants, spatial distance fidelity, logging schema stability, and replay reconstruction all under regression test.

### Playback Controls (GUI)
Space: play/pause | Right Arrow: step forward | Up/Down: speed ± (doubling/halving) | Esc/q: quit.
HUD displays current status (PLAY/PAUSE/LIVE), speed (r/s), convergence index, solver residual, fill & unmet shares, distances, efficiency, and a digest snippet.

### Numerical Notes
Adaptive fallback (tâtonnement) engages if `fsolve` struggles; final rest-goods norm enforced at < 1e-8. HUD surfaces solver residual & status for transparency.

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