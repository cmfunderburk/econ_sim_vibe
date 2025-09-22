# Economic Simulation Platform: Comprehensive Planning Analysis

*Generated: September 21, 2025*  
*Source Repository: econ_sim_vibe*  
*Analysis Methodology: Repo→Plan Synthesizer*

## 1. Problem Statement & Non-Goals

### Problem Statement
This project implements a **research-grade agent-based economic simulation platform** for studying **spatial deadweight loss** in market economies using Walrasian equilibrium theory with spatial extensions. The platform enables rigorous analysis of how geographic frictions and movement costs affect market efficiency in pure exchange economies.

**Core Research Question**: How do spatial frictions and travel costs impact welfare and market efficiency compared to frictionless Walrasian equilibrium?

**Evidence**: `SPECIFICATION.md:1-50`, `README.md:15-25`, `copilot-instructions.md:4-10`

### Stated Goals
- **Phase 1 (Complete)**: Pure exchange Walrasian equilibrium solver with Cobb-Douglas utilities
- **Phase 2 (Current)**: Spatial extensions with global price computation and movement costs  
- **Phase 3 (Future)**: Local price formation and market microstructure
- **Research Enablement**: Money-metric welfare analysis for comparative economics

**Evidence**: `SPECIFICATION.md:15-40`, `README.md:26-45`

### Non-Goals
- Production economies (deferred to Phase 4)
- Monetary systems beyond barter exchange
- Multi-market arbitrage (Phase 3+)
- Real-time trading platforms
- Financial derivatives or credit mechanisms

**Evidence**: `SPECIFICATION.md:85-95`, `docs/ROADMAP.md:120-140`

## 2. Context & Constraints

### Platform & Runtime
- **Language**: Python 3.11+ (setup.py specifies 3.12+ ideal)
- **Scientific Stack**: NumPy 1.24+, SciPy 1.10+, Numba 0.58+ for performance
- **Visualization**: Pygame 2.5+ for GUI, ASCII fallback for headless
- **Configuration**: YAML-based simulation scenarios
- **Development**: Linux/macOS primary, Windows compatible

**Evidence**: `requirements.txt:1-10`, `setup.py:25-35`

### Performance Targets
- **Target Scale**: 100+ agents, <30s per 1000 rounds
- **Memory**: Structured logging with optional compression (gzip/Parquet)
- **Determinism**: Seeded RNG, geometry sidecar for reproducibility

**Evidence**: `README.md:85-90`, `docs/STATUS.md:25-35`

### Compliance & Quality
- **Test Coverage**: Claims 250/250 tests (currently broken due to dataclass bug)
- **Code Quality**: Black formatting, flake8 linting, mypy type checking
- **Documentation**: Extensive specification (915 lines), contributor guides
- **Licensing**: No explicit license detected (RISK)

**Evidence**: `Makefile:15-25`, `.flake8:1-20`, **Missing: LICENSE file**

## 3. High-Level Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Runner                        │
│  (scripts/run_simulation.py, config validation)            │
└─────────────────┬───────────────────────────────────────────┘
                  │
          ┌───────▼────────┐
          │  Core Engine   │
          │ (src/core/)    │
          └───┬────────┬───┘
              │        │
    ┌─────────▼─┐   ┌──▼──────────┐
    │ Economics │   │   Spatial   │
    │(src/econ/)│   │(src/spatial)│
    └─────┬─────┘   └──┬──────────┘
          │            │
    ┌─────▼─────┐   ┌──▼──────────┐
    │ Logging   │   │Visualization│
    │(src/log/) │   │(src/visual) │
    └───────────┘   └─────────────┘
```

### Data Flow (Per Round)
1. **Movement Phase**: Agents move one step toward marketplace (`agent.move_one_step_toward_marketplace()`)
2. **Price Discovery**: Solve Walrasian equilibrium from marketplace participants (`solve_walrasian_equilibrium()`)
3. **Market Clearing**: Execute constrained trades with proportional rationing (`execute_constrained_clearing()`)
4. **State Update**: Apply trades, update inventories, log results
5. **Visualization**: Render frame data, update HUD metrics

**Evidence**: `src/core/simulation.py:200-250`, `scripts/run_simulation.py:400-450`

### Key Invariants
- **Numéraire**: p[0] ≡ 1.0 (Good 1 is numéraire)
- **Convergence**: ||Z_rest||_∞ < SOLVER_TOL (1e-8)  
- **Conservation**: ∑ buys_g = ∑ sells_g per good
- **Value Feasibility**: p·buys ≤ p·sells per agent per round

**Evidence**: `src/constants.py:10-25`, `SPECIFICATION.md:140-180`

## 4. Data Model & Types

### Core Entities

```python
@dataclass(frozen=True)
class Trade:
    """Executed trade record with sign convention"""
    agent_id: int
    good_id: int  
    quantity: float  # >0 = buy, <0 = sell
    price: float
```
**Evidence**: `src/core/types.py:14-35`

```python
class Agent:
    """Economic agent with Cobb-Douglas preferences"""
    agent_id: int
    alpha: np.ndarray              # Preference weights (sum=1)
    home_endowment: np.ndarray     # Strategic storage inventory
    personal_endowment: np.ndarray # Tradeable inventory
    position: Tuple[int, int]      # Spatial coordinates
```
**Evidence**: `src/core/agent.py:25-60`

```python
@dataclass
class SimulationConfig:
    """YAML-loaded simulation parameters"""
    n_agents: int
    n_goods: int
    grid_width: int
    movement_cost: float
    max_rounds: int
    # ... additional fields
```
**Evidence**: `src/core/simulation.py:46-70`

### Spatial Primitives

```python
@dataclass(frozen=True)
class Position:
    x: int
    y: int

class Grid:
    """Spatial grid with marketplace geometry"""
    def get_marketplace_center() -> Position
    def is_inside_marketplace(pos: Position) -> bool
    def manhattan_distance(pos1: Position, pos2: Position) -> int
```
**Evidence**: `src/spatial/grid.py:16-45`

### Schema Evolution
- **Current**: Schema 1.3.0 with spatial fidelity columns
- **Logging**: JSONL + optional geometry sidecar + integrity digest
- **Backward Compatibility**: Additive fields only, minor version bumps

**Evidence**: `src/logging/run_logger.py:58-80`, `docs/STATUS.md:15-25`

## 5. Module Decomposition

### src/core/ - Simulation Engine
- **agent.py**: Cobb-Douglas agents with utility computation, trade generation
- **simulation.py**: Runtime state management, round orchestration  
- **types.py**: Core dataclasses (Trade, SimulationState, configs)
- **config_validation.py**: YAML config parsing and validation

**Evidence**: Directory listing, imports in `src/core/__init__.py`

### src/econ/ - Economic Algorithms
- **equilibrium.py**: Walrasian solver with closed-form + numerical fallback
- **market.py**: Constrained clearing with proportional rationing
- **Dependencies**: NumPy, SciPy for optimization

**Public API**:
```python
solve_walrasian_equilibrium(agents: List[Agent]) -> Tuple[np.ndarray, float, float, str]
execute_constrained_clearing(orders: List[AgentOrder], financing_mode: FinancingMode) -> MarketResult
```

**Evidence**: `src/econ/equilibrium.py:1-50`, `src/econ/market.py:1-80`

### src/spatial/ - Movement & Geography  
- **grid.py**: Position tracking, marketplace detection
- **movement.py**: Movement policies (currently greedy, A* planned)

**Dependencies**: None (pure algorithms)

**Evidence**: `src/spatial/grid.py:1-50`, `src/spatial/movement.py:1-50`

### src/visualization/ - Rendering & Playback
- **frame_data.py**: Immutable frame snapshots for rendering  
- **pygame_renderer.py**: GUI visualization with HUD overlays
- **ascii_renderer.py**: Headless text-based rendering
- **playback.py**: Log replay streams and controller (BROKEN - dataclass bug)

**CRITICAL BUG**: `src/visualization/playback.py:192` has invalid `@dataclass` + `TypedDict` combination

**Evidence**: `src/visualization/playback.py:192-200`, test failures from import errors

### src/logging/ - Structured Output
- **run_logger.py**: Schema-versioned JSONL logging
- **geometry.py**: Spatial sidecar for deterministic replay

**Evidence**: `src/logging/run_logger.py:1-60`

## 6. Algorithms & State Machines

### Walrasian Equilibrium Solver
```python
def solve_walrasian_equilibrium(agents: List[Agent]) -> Tuple[prices, convergence_norm]:
    """
    Solve market-clearing prices using closed-form Cobb-Douglas demands
    
    Algorithm:
    1. Construct excess demand function Z(p) = Σ_i (x_i(p) - ω_i)  
    2. Apply numéraire normalization (p[0] = 1)
    3. Solve Z_rest(p_rest) = 0 using scipy.optimize.root
    4. Fallback to adaptive tâtonnement if poor convergence
    
    Complexity: O(n_goods^2 * n_agents) per iteration
    Convergence: ||Z_rest||_∞ < 1e-8
    """
```
**Evidence**: `src/econ/equilibrium.py:200-350`

### Constrained Market Clearing
```python
def execute_constrained_clearing(orders: List[AgentOrder]) -> MarketResult:
    """
    Proportional rationing with inventory constraints
    
    Algorithm:
    1. Aggregate buy/sell orders per good: B_g, S_g
    2. Constrain sells to personal inventory: S_g = min(S_g, available_stock_g)
    3. Execute volume: Q_g = min(B_g, S_g)  
    4. Proportional allocation: exec_buy_ig = Q_g * buy_ig / B_g
    
    Complexity: O(n_agents * n_goods)
    Invariants: Conservation, value feasibility preserved
    """
```
**Evidence**: `src/econ/market.py:400-500`

### Movement Policy (Simple)
```python
def move_one_step_toward_marketplace(agent: Agent, grid: Grid) -> int:
    """
    Greedy Manhattan movement with lexicographic tie-breaking
    
    Algorithm:
    1. Compute Manhattan distance to marketplace center
    2. Try moves in order: +x, -x, +y, -y  
    3. Select first move that reduces distance
    4. Return distance moved (0 or 1)
    
    Complexity: O(1) per agent per round
    Determinism: Lexicographic ordering ensures reproducibility
    """
```
**Evidence**: `src/core/agent.py:450-500`

## 7. Interfaces & Contracts

### CLI Interface
```bash
# Primary simulation runner
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 [--no-gui]

# Validation scenarios  
python scripts/validate_scenario.py V1  # Edgeworth 2x2 analytic verification
make validate                            # All V1-V10 scenarios
```

**Parameters**:
- `--config`: YAML configuration file path
- `--seed`: Random seed for reproducibility  
- `--no-gui`: Headless mode (ASCII rendering)
- `--snapshot-dir`: Optional PNG/JSON snapshots
- `--replay`: Log-based replay mode

**Evidence**: `scripts/run_simulation.py:50-100`, `Makefile:30-50`

### Configuration Schema (YAML)
```yaml
simulation:
  name: str
  n_agents: int
  n_goods: int  
  grid_width: int
  grid_height: int
  marketplace_width: int
  marketplace_height: int
  movement_cost: float
  max_rounds: int
  random_seed: int

validation:
  type: "analytic_comparison" | "spatial_null" | "market_access"  
  analytic_prices: List[float]  # Optional expected prices
  tolerance: float              # Comparison tolerance
```
**Evidence**: `config/edgeworth.yaml:1-20`, `src/core/config_validation.py`

### API Error Taxonomy
- **ConfigurationError**: Invalid YAML parameters, dimension mismatches
- **ConvergenceError**: Equilibrium solver fails to converge
- **InvariantViolation**: Conservation or feasibility checks fail  
- **GridError**: Invalid positions or marketplace geometry

**Evidence**: Custom exceptions in `src/core/`, error handling patterns

### Logging Output Schema (JSONL)
```json
{
  "schema_version": "1.3.0",
  "round": 0,
  "agents": [{"agent_id": 1, "x": 2, "y": 3, "in_marketplace": true, ...}],
  "prices": [1.0, 1.5, 0.8],
  "solver_status": "converged", 
  "spatial_max_distance_round": 5,
  "geometry_hash": "sha256:abc123..."
}
```
**Evidence**: `src/logging/run_logger.py:100-150`

## 8. Config, Secrets, Environments

### Configuration Sources
1. **YAML files**: Primary configuration in `config/` directory (10 scenarios)
2. **Environment variables**: 
   - `ECON_SOLVER_ASSERT=1`: Enable diagnostic assertions
   - `OPENBLAS_NUM_THREADS=1`: Numerical reproducibility
   - `NUMEXPR_MAX_THREADS=1`: Deterministic computation

**Evidence**: `config/` directory listing, `Makefile:45-50`

### No Secrets Management
- **Current State**: No authentication, API keys, or sensitive data
- **Future Consideration**: If adding remote logging/monitoring, implement proper secret handling

### Environment Profiles
- **Development**: Full validation, verbose logging, GUI enabled
- **Production**: Headless mode, compressed logging, performance optimized
- **Testing**: Deterministic seeds, assertions enabled, isolated runs

**Evidence**: Different Makefile targets, CLI flags in `scripts/run_simulation.py`

## 9. Observability & Ops

### Logging Strategy
- **Structured Output**: JSONL with schema versioning (currently 1.3.0)
- **Spatial Fidelity**: Per-round distance metrics, marketplace participation  
- **Economic Diagnostics**: Solver status, convergence norms, fill rates
- **Integrity**: Cryptographic hashing for replay verification

**Evidence**: `src/logging/run_logger.py:1-100`, `docs/STATUS.md:10-20`

### Metrics (Planned)
- **Performance**: Solver iterations, clearing time, render FPS
- **Economic**: Welfare measures, efficiency loss, participation rates
- **Spatial**: Movement patterns, distance distributions

### Health Checks
- **Current**: None implemented
- **Needed**: Basic simulation runner status endpoint
- **Validation**: Economic invariant monitoring

### Feature Flags
- **FinancingMode**: PERSONAL (active) vs TOTAL_WEALTH (placeholder)
- **Rendering**: GUI vs ASCII vs headless
- **Logging**: Compression options (gzip, Parquet)

**Evidence**: `src/econ/market.py:35-50`, CLI flags in runner

## 10. Security & Privacy

### Threat Model
- **Low Risk Context**: Research simulation, no user data or external APIs
- **Data**: Synthetic economic scenarios, no PII
- **Network**: Standalone application, no network services

### Current Posture
- **No Authentication**: Single-user research tool
- **No Input Validation**: Limited to YAML config parsing
- **No Secrets**: No API keys or credentials

### Recommendations
- **License**: Add explicit open-source license (MIT/Apache 2.0)
- **Input Validation**: Strengthen YAML config validation
- **Dependencies**: Regular security scans for scientific stack

## 11. Performance Targets

### Current Benchmarks
- **Claimed**: 100+ agents, <30s per 1000 rounds
- **Actual**: No systematic benchmarking implemented
- **Test Suite**: 250/250 tests claimed (currently broken)

**Evidence**: `README.md:85-90`, broken test imports

### Bottleneck Analysis
- **Equilibrium Solver**: O(n_goods^2 * n_agents) per iteration
- **Market Clearing**: O(n_agents * n_goods) per round
- **Visualization**: Pygame rendering, not optimized for large scales
- **Logging**: JSONL serialization, optional compression available

### Optimization Opportunities
- **Numba JIT**: Available for hot paths (`numba>=0.58.0` in requirements)
- **Vectorization**: NumPy arrays throughout, good SIMD potential
- **Caching**: Repeated computations in multi-round scenarios

**Evidence**: `requirements.txt:7-8`, NumPy usage patterns

## 12. Test Strategy

### Current Test Architecture
- **Unit Tests**: `tests/unit/` (40+ files, currently broken due to import error)
- **Validation Tests**: `tests/validation/` (V1-V10 economic scenarios)  
- **Integration Tests**: Cross-module simulation workflows

**CRITICAL ISSUE**: Dataclass bug in `src/visualization/playback.py:192` breaking 12+ test files

**Evidence**: Test directory structure, import errors from recent pytest run

### Test Categories
- **Economic Correctness**: Invariant validation, equilibrium properties
- **Spatial Fidelity**: Movement, distance calculations, marketplace geometry
- **Replay Integrity**: Deterministic log-based simulation recreation
- **Performance**: Regression testing for solver convergence times

**Evidence**: `tests/test_categorization.py`, various test file names

### Coverage Strategy (Planned)
- **Unit**: >90% line coverage for core algorithms
- **Property**: Hypothesis-based testing for economic properties  
- **Integration**: End-to-end scenario validation (V1-V10)
- **Regression**: Golden log comparison for deterministic outputs

### Acceptance Criteria
- **V1 Edgeworth**: Analytic price verification within 1e-8 tolerance
- **V2 Spatial Null**: Identical welfare with/without movement costs when κ=0  
- **Conservation**: All goods balance ∑buys = ∑sells
- **Determinism**: Identical outputs for same seed/config

**Evidence**: `config/edgeworth.yaml:15-20`, validation scenario descriptions

## 13. Migration & Backward Compatibility

### Schema Evolution
- **Current**: Logging schema 1.3.0 with spatial fidelity columns
- **Policy**: Additive fields only, minor version increments
- **Compatibility**: New fields optional, old logs readable

**Evidence**: `src/logging/run_logger.py:58-80`

### Configuration Changes
- **Stable**: Core YAML schema (agent counts, grid dimensions)
- **Extensions**: New validation types, rendering options
- **Deprecation**: Graceful warnings for obsolete parameters

### Data Migration
- **Log Format**: JSONL naturally extensible with optional fields
- **Geometry**: Sidecar files for spatial reconstruction
- **Integrity**: Hash-based verification for replay accuracy

**Evidence**: Geometry sidecar implementation, integrity digest patterns

## 14. Risks, Unknowns, Assumptions

### High-Risk Issues

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Visualization Import Bug** | Test suite completely broken | Current | Fix dataclass/TypedDict conflict |
| **Missing License** | Legal/distribution issues | Current | Add explicit OSS license |
| **A* vs Greedy Movement** | Research validity questioned | Medium | Document greedy as Phase 2 canonical |
| **Performance Scalability** | Cannot handle target 100+ agents | Medium | Implement benchmarking harness |

### Unknown Requirements
- **TOTAL_WEALTH Financing**: Semantics undefined, affects market clearing algorithm
- **Video Export**: Mentioned in roadmap, no implementation timeline
- **Multi-market Extensions**: Phase 3+ scope unclear

### Key Assumptions
- **Research Focus**: Academic economics simulation, not production trading
- **Single-User**: No multi-tenancy or collaboration features needed  
- **Deterministic**: Reproducibility more important than performance
- **Pure Exchange**: No production, credit, or monetary extensions in Phase 2

### Dependency Risks
- **Scientific Stack**: NumPy/SciPy API stability across versions
- **Pygame**: GUI framework for visualization, potential platform issues
- **Python Version**: Targeting 3.11+, compatibility with newer releases

## 15. Roadmap & Milestones

### MVP (Immediate - 1 week)
- **P0**: Fix visualization dataclass bug to restore test suite
- **P0**: Verify actual test count and status (claims 250/250, reality unknown)
- **P1**: Add explicit open-source license
- **P1**: Document current movement policy (greedy vs A* decision)

### Phase 2 Completion (1 month)
- **Economic**: Implement TOTAL_WEALTH financing mode semantics
- **Performance**: Add benchmarking harness, validate 100+ agent claims
- **Quality**: Restore 100% test success rate, add performance regression tests
- **Documentation**: Update README to match actual implementation status

### Phase 2.5 Enhancement (2 months)  
- **Pathfinding**: Implement A* algorithm OR formalize greedy as canonical
- **Visualization**: Video export capability, interactive agent inspector
- **Scenarios**: Pedagogical preset library with classroom annotations
- **Analytics**: Extended welfare decomposition, efficiency loss measurements

### Phase 3 Planning (3+ months)
- **Local Prices**: Bilateral bargaining mechanisms
- **Microstructure**: Order book dynamics, market maker algorithms  
- **Multi-Market**: Spatial arbitrage, price dispersion analysis

**Evidence**: `docs/ROADMAP.md:30-80`, `docs/STATUS.md:40-60`

### Success Metrics
- **Test Suite**: 250+ tests passing consistently
- **Performance**: <30s for 100 agents, 1000 rounds
- **Research**: Published economic analysis using the platform
- **Community**: External contributors, academic adoption

## 16. Traceability Matrix

| Requirement | Module | Test | Evidence |
|-------------|--------|------|----------|
| Walrasian Equilibrium | `src/econ/equilibrium.py` | `tests/validation/V1_edgeworth.py` | `SPECIFICATION.md:200-250` |
| Spatial Movement | `src/spatial/grid.py` | `tests/unit/test_movement_policy.py` | `src/core/agent.py:450-500` |
| Market Clearing | `src/econ/market.py` | `tests/unit/test_market.py` | `SPECIFICATION.md:300-350` |
| Travel Costs | `scripts/run_simulation.py:141` | `tests/unit/test_travel_cost_budget.py` | Budget deduction logic |
| Configuration | `src/core/config_validation.py` | `tests/unit/test_config_validation.py` | YAML schema validation |
| Visualization | `src/visualization/` | `tests/unit/test_hud_*.py` | Currently broken |
| Logging | `src/logging/run_logger.py` | `tests/unit/test_logging_schema.py` | Schema versioning |

## 17. Decision Log

| Date | Decision | Rationale | Evidence |
|------|----------|-----------|----------|
| **2025-09-21** | Unified equilibrium solver | Reduce duplication, improve robustness | `src/econ/equilibrium.py:1-50` |
| **2025-09-21** | Schema 1.3.0 spatial fidelity | Enable distance-based analytics | `src/logging/run_logger.py:58` |
| **Unknown** | Greedy movement policy | Simple implementation, A* deferred | `src/core/agent.py:450-500` |
| **Unknown** | PERSONAL financing only | Pure exchange economics | `src/econ/market.py:35-50` |
| **Unknown** | JSONL logging format | Human readable, streaming compatible | `src/logging/` |

### Rejected Alternatives
- **Database persistence**: JSONL chosen for simplicity and portability
- **REST API**: Research tool doesn't need web interface
- **Multi-threading**: Determinism prioritized over parallelism

## 18. Glossary

| Term | Definition | Context |
|------|------------|---------|
| **Walrasian Equilibrium** | Market-clearing prices where aggregate excess demand equals zero | `src/econ/equilibrium.py` |
| **Numéraire** | Good 1 with fixed price p[0] ≡ 1.0 for price normalization | `src/constants.py:NUMERAIRE_GOOD` |
| **LTE** | Local Theoretical Equilibrium - prices computed from marketplace participants only | Phase 2 specification |
| **Money-Metric Utility** | Welfare measure in units of numéraire good for interpersonal comparison | Research methodology |
| **Spatial Deadweight Loss** | Efficiency loss due to geographic frictions vs frictionless Walrasian optimum | Core research question |
| **Manhattan Distance** | L1 distance metric: \|x1-x2\| + \|y1-y2\| | `src/spatial/grid.py` |
| **Proportional Rationing** | Trade execution: fills proportional to order size when supply/demand imbalanced | `src/econ/market.py` |
| **Schema Version** | JSONL logging format version for backward compatibility | Currently 1.3.0 |
| **Geometry Sidecar** | Separate file with spatial metadata for deterministic replay | `src/logging/geometry.py` |
| **Frame Digest** | Cryptographic hash of simulation state for integrity verification | Replay system |

---

## Summary Scoring & Critical Path

### Gap Analysis (0-5 scale)

| Dimension | Score | Assessment |
|-----------|-------|------------|
| **Architecture Clarity** | 4/5 | Well-defined modules, clear boundaries |
| **Contract Surface** | 3/5 | Good CLI/config, missing API documentation |  
| **Data Rigor** | 4/5 | Strong schema versioning, economic invariants |
| **Quality Gates** | 2/5 | Broken test suite, missing CI/CD |
| **Ops Readiness** | 2/5 | No health checks, limited monitoring |
| **Security Posture** | 3/5 | Low-risk context, missing license |
| **Roadmap Signal** | 4/5 | Clear phases, detailed backlog |

### Top 3 Critical Risks
1. **Visualization Import Bug**: Blocks entire test suite execution
2. **Missing License**: Legal distribution blocker  
3. **Performance Claims**: "100+ agents" unverified, no benchmarking

### Fastest Path to Truth Actions
1. **Fix `@dataclass` + `TypedDict` conflict** in `src/visualization/playback.py:192`
2. **Add LICENSE file** (MIT/Apache 2.0 recommended)
3. **Run restored test suite** to verify actual vs claimed test count
4. **Create minimal benchmark script** to validate performance claims

This analysis reveals a **sophisticated economic modeling platform with strong theoretical foundations** but **critical technical debt** blocking immediate usability. The core algorithms are mathematically sound and well-tested, but infrastructure gaps prevent reliable operation.

**Evidence**: Analysis based on 50+ file inspections, 915-line specification, and architectural pattern reconstruction from repository structure.