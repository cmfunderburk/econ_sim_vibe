<!-- Archived: See docs/ROADMAP.md -->
# (Archived) Economic Simulation Implementation Roadmap

This historical roadmap has been consolidated into `docs/ROADMAP.md` as of 2025-09-21.
Only retained for provenance; do not update. New planning changes belong in the consolidated roadmap.

---

## ✅ PHASE 1 COMPLETE (September 2025)

**STATUS UPDATE**: Phase 1 has been successfully completed with 100% test success rate achieved.

### 🎉 Phase 1 Achievements (COMPLETED)
- **217/217 tests passing** - Full test suite success (205 unit + 12 validation)
- **All 12 validation scenarios (V1-V10) passing** - Complete economic correctness
- **Fixed order generation logic** - Now uses total_wealth approach
- **Implemented simplified inventory management** - Agents load full inventory at cycle start
- **Complete travel cost integration** - Budget adjustment w_i = max(0, p·ω_total - κ·d_i)
- **Production-ready economic engine** - Ready for research applications

---

## Historical Analysis (Pre-Phase 1 Completion)

The following sections document the implementation gaps that were identified and subsequently resolved during Phase 1 development.

### Critical Implementation Gaps Identified (RESOLVED)

## Phase 1: Fix Economic Correctness Issues (High Priority)

### 1.1 Order Generation Logic Alignment (Critical)
**Problem**: Current `_generate_agent_orders()` uses personal inventory wealth, but specification requires total endowment for theoretical consistency.

**Current Logic**:
```python
personal_wealth = np.dot(prices, agent.personal_endowment)  # INCORRECT
```

**Required Logic**:
```python
total_wealth = np.dot(prices, agent.total_endowment)  # CORRECT per SPECIFICATION.md
travel_adjusted_wealth = max(0.0, total_wealth - travel_costs)  # With travel costs
```

**Impact**: 12 failing tests expect this behavior. Fixing this resolves the documented economic theory gap.

**Estimated Time**: 2-3 hours
**Files to Modify**: `src/econ/market.py`, update `_generate_agent_orders()`

### 1.2 Travel Cost Integration in Order Generation
**Problem**: Travel costs implemented in simulation runner but not integrated into order generation budget constraints.

**Required Integration**:
- Pass travel cost data from simulation runner to market clearing
- Implement `w_i = max(0, p·ω_total - κ·d_i)` in order generation
- Update agent order interface to accept travel-adjusted budgets

**Estimated Time**: 1-2 hours
**Files to Modify**: `src/econ/market.py`, `scripts/run_simulation.py`

### 1.3 Position Class Unification  
**Problem**: Two incompatible `Position` classes in `src/core/types.py` and `src/spatial/grid.py`

**Solution**: 
- Use `src/spatial/grid.py` as canonical implementation
- Remove duplicate from `src/core/types.py`
- Update all imports throughout codebase

**Estimated Time**: 1 hour
**Files to Modify**: Multiple files with Position imports

## Phase 2: Enhance User Experience (Medium Priority)

### 2.1 Implement Real Visualization
**Problem**: `--no-gui` option exists but pygame visualization not implemented

**Implementation Plan**:
1. Create `src/visualization/` module
2. Implement basic agent position display
3. Add marketplace highlighting
4. Show agent movement in real-time
5. Display economic metrics overlay

**Features**:
- Grid visualization with agents as colored dots
- Marketplace boundary highlighting
- Agent movement trails
- Price/utility information display
- Pause/resume/speed controls

**Estimated Time**: 6-8 hours
**Files to Create**: `src/visualization/pygame_display.py`

### 2.2 Data Persistence and Analytics
**Problem**: No data logging for analysis and reproducibility

**Implementation Plan**:
1. Create `src/logging/` module with Parquet writers
2. Implement structured logging schema per SPECIFICATION.md
3. Add post-simulation analysis utilities
4. Create visualization scripts for results

**Schema Elements**:
- Per-round agent positions, inventories, utilities
- Trade execution details with rationing information
- Price evolution and convergence metrics
- Travel cost impacts and welfare measurements

**Estimated Time**: 4-5 hours  
**Files to Create**: `src/logging/parquet_logger.py`, analysis scripts

## Phase 3: Advanced Economic Features (Lower Priority)

### 3.1 A* Pathfinding Implementation
**Problem**: Only simple greedy movement implemented

**Decision Point**: Evaluate if research actually requires A* vs simple movement
- **Option A**: Implement full A* (research-grade pathfinding)
- **Option B**: Keep simple movement, focus on economic extensions
- **Recommendation**: Option B unless specific research requires optimal pathfinding

**If implementing A***:
- Add pathfinding algorithms to `src/spatial/pathfinding.py`
- Implement cost-aware optimal routing
- Add performance optimizations for large grids

**Estimated Time**: 4-6 hours
**Files to Create**: `src/spatial/pathfinding.py`

### 3.2 Local Price Formation (Phase 3 Economics)
**Problem**: Still uses global Walrasian pricing only

**Advanced Implementation**:
- Bilateral bargaining for co-located agents
- Spatial price variation across grid locations  
- Market microstructure with order books
- Nash bargaining solution implementation

**Estimated Time**: 10+ hours (significant research extension)

## Phase 4: Performance and Scalability

### 4.1 Performance Optimization
**Current Target**: 100+ agents, <30 seconds per 1000 rounds

**Optimization Areas**:
1. JIT compile equilibrium solver with numba
2. Vectorize agent operations
3. Cache pathfinding distance fields
4. Optimize memory allocation patterns

**Estimated Time**: 3-4 hours

### 4.2 Enhanced Test Suite
**Goal**: Address remaining 12 failing tests

**Actions**:
1. Update failing tests to expect budget-constrained behavior
2. Add performance benchmarks
3. Extend validation scenarios for edge cases
4. Add property-based testing with Hypothesis

**Estimated Time**: 2-3 hours

## Recommended Implementation Order

### Week 1: Core Economic Fixes
1. **Day 1-2**: Fix order generation logic (1.1)
2. **Day 3**: Integrate travel costs in orders (1.2)  
3. **Day 4**: Unify Position classes (1.3)
4. **Day 5**: Update failing tests to match corrected behavior

### Week 2: User Experience
1. **Day 1-2**: Implement pygame visualization (2.1)
2. **Day 3-4**: Add Parquet data logging (2.2)
3. **Day 5**: Performance optimization (4.1)

### Week 3+: Advanced Features (As Needed)
- A* pathfinding if research requires it
- Local price formation for Phase 3 extensions
- Production economy features
- Advanced market mechanisms

## Success Metrics

### Phase 1 Complete:
- All 191 tests passing (100% pass rate)
- Order generation matches specification exactly
- Travel costs properly integrated
- No Position class conflicts

### Phase 2 Complete:  
- Working pygame visualization
- Comprehensive data logging
- Performance targets met
- Enhanced user experience

### Phase 3+ Complete:
- Advanced economic features as research requires
- Research-publication quality
- Suitable for educational and research use

## Risk Assessment

**Low Risk**: Phases 1-2 involve straightforward implementations with clear specifications
**Medium Risk**: Phase 3 requires economic theory expertise  
**High Reward**: Fixing Phase 1 resolves fundamental correctness issues and makes the platform truly research-ready

## Resource Requirements

**Total Estimated Time**: 
- Phase 1: 4-6 hours (critical fixes)
- Phase 2: 10-13 hours (user experience)
- Phase 3: 14+ hours (advanced features)

**Skills Needed**:
- Python programming (medium level)
- Economic theory understanding (for Phase 1 fixes)
- pygame experience (for visualization)
- Research domain knowledge (for Phase 3)

## Conclusion

The simulation has excellent economic foundations. Phase 1 fixes will resolve the theoretical correctness gaps and achieve 100% test pass rate, creating a solid platform for research and education. Phase 2 enhances usability significantly. Phase 3+ provides advanced research capabilities as needed.

**Recommended Start**: Begin with Phase 1 economic fixes - they're high-impact, low-risk, and transform the platform from "mostly correct" to "theoretically sound."