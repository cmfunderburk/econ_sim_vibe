# Economic Simulation Project

The goal of this project is to create a modular simulation of an economy using agent-based modeling with rigorous economic theory foundations. This project follows a pedagogical progression from pure Walrasian general equilibrium theory to realistic spatial market mechanisms.

## Project Vision

This project builds a comprehensive economic modeling platform step-by-step from economic first principles:
1. **Phase 1**: Pure exchange economy (Walrasian equilibrium)
2. **Phase 2**: Spatial extensions with global prices
3. **Phase 3**: Local price formation and market mechanisms
4. **Future**: Production, money, institutions, and behavioral economics

## Economic Theory Framework

### Phase 1: Pure Exchange Economy (Walrasian)
- **Pure barter economy**: No money, all trades are goods-for-goods exchanges
- **Global price vector**: Computed by solving aggregate excess demand = 0
- **Perfect information**: All agents know all prices instantly  
- **No spatial frictions**: Focus on utility maximization and efficiency
- **Market clearing**: ∑ᵢ zᵢ(p) = 0 where zᵢ is agent i's excess demand
- **Budget constraint**: p·x ≤ p·ω (consumption value ≤ endowment value)
- **Welfare theorems**: Competitive equilibrium is Pareto efficient

### Phase 2: Spatial Extensions (Current Implementation)
- **Static intra-round pricing**: Walrasian prices computed fresh each round, no intra-round dynamics
- **Inter-round price evolution**: Prices may change across rounds as participant set and endowments evolve
- **Grid-based movement**: Agents travel with explicit movement costs affecting utility
- **Market access restriction**: Only agents inside marketplace can trade
- **LTE (Local Theoretical Equilibrium)**: Prices computed only from agents currently in marketplace
- **Research objective**: Measure spatial deadweight loss via money-metric welfare analysis

### Phase 3: Local Price Formation (Future)
- **Dynamic price discovery**: Prices emerge from local trading mechanisms
- **Bilateral bargaining**: Nash bargaining solution for co-located agents
- **Marketplace auctions**: Continuous double auction mechanism
- **Spatial price variation**: Prices differ across locations and time
- **Market microstructure**: Order books, bid-ask spreads, market makers

## Notation Reference

**Core Symbols:**
- $x$ = consumption bundle (post-trade)
- $\omega^{\text{home}}$ = home inventory (strategic storage)
- $\omega^{\text{personal}}$ = personal inventory (tradeable at marketplace)
- $\omega^{\text{total}} = \omega^{\text{home}} + \omega^{\text{personal}}$ = total endowment
- $p$ = price vector with $p_1 \equiv 1$ (numéraire)
- $Z$ = excess demand vector ($Z = \text{demand} - \text{endowment}$)
- $d$ = distance traveled (Manhattan/L1 metric)
- $\kappa$ = movement cost per step (units of good 1)

## Quick Reference for Contributors

**Implementation Status (September 2025):**
- ✅ **Phase 1 Economic Engine COMPLETE**: Walrasian equilibrium solver + market clearing mechanisms
- ✅ **Complete Validation Framework ACHIEVED**: 247/247 tests passing (100%), all V1-V10 scenarios validated (unit + validation + replay & spatial fidelity)
- ✅ **Mathematical Foundation**: All economic invariants satisfied, robust error handling, excellent performance
- ✅ **Basic Spatial Implementation**: Grid movement, travel cost integration, simulation runner functional
- ✅ **Visualization & Replay**: Pygame HUD + ASCII renderer, deterministic log-based replay (schema 1.3.0) with integrity digest
- ⚠️ **Simple Movement System**: Greedy movement (A* pathfinding planned)
- 📋 **Next Priority**: Advanced pathfinding, financing mode extensions, performance benchmarks

**Key Invariants (Never Violate):**
- p₁ ≡ 1 (numéraire constraint)
- ||Z_market(p)_{2:n}||_∞ < SOLVER_TOL (primary convergence criterion)
- ∑ᵢ executed_buys_g = ∑ᵢ executed_sells_g (conservation per good)
- Per-agent: buy_value ≤ sell_value (value feasibility per round)

**Critical Round Order:**
1. Move → 2. Price (post-move participants) → 3. Clear (constrained by personal inventory)

**Common Implementation Pitfalls:**
- Using `len(prices)` instead of `agents[0].alpha.size` for n_goods
- Pricing before movement (violates local-participants principle)
- Forgetting personal inventory constraints in clearing algorithm
- Using Walras' Law |p·Z_market(p)| instead of rest-goods norm ||Z_market(p)_{2:n}||_∞ for convergence

## Numerical Constants & Units (Source of Truth)

### Convergence & Numerical Tolerances
- **SOLVER_TOL = 1e-8**: Primary convergence criterion for rest-goods norm ||Z_market(p)_{2:n}||_∞
- **FEASIBILITY_TOL = 1e-10**: All feasibility and conservation checks (value feasibility, market balance)
- **RATIONING_EPS = 1e-10**: Prevent division by zero in proportional rationing (max(B_g, ε) denominators)
- **Walras validation**: Uses SOLVER_TOL for sanity check |p·Z_market(p)| < 1e-8

### Distance & Travel Cost Units
- **Distance metric**: Manhattan/L1 metric on grid (1 unit per step, 4-directional movement)
- **κ units**: Denominated in units of good 1 per grid step (budget-side deduction only)
- **Tie-breaking**: Lexicographic by (x,y) coordinates then agent ID for deterministic pathfinding
- **EV reporting**: All equivalent variation values in units of good 1 (numéraire)

### Solver & Runtime Diagnostics (September 2025 Update)

Phase A unified the equilibrium solving logic into a single robust implementation
`solve_walrasian_equilibrium` (legacy minimal variant removed). Public API
unchanged: `(prices, z_rest_norm, walras_dot, status)` plus status labels
(`converged`, `poor_convergence`, `no_participants`, `insufficient_viable_agents`,
`insufficient_participants`, `failed`).

Environment flags:

| Flag | Default | Effect |
|------|---------|--------|
| `ECON_SOLVER_ASSERT` | `0` | When `1`, enables runtime assertions (shape, positivity, finite residual checks) for development/CI. |
| `ECON_ENABLE_TRAVEL_BUDGET` | `1` | When `0`, disables travel-cost deduction so wealth uses `p·ω_total` (comparison / legacy mode). |

Travel-cost adjusted wealth (when enabled):
\[ w_i = \max(0, p \cdot \omega_i^{total} - \kappa d_i) \]

Flags are environment-based (not YAML) to allow quick A/B experiments and
diagnostic hardening without mutating stored scenario configurations.

## Architecture Overview

- **Performance Target**: Scalable to 100+ agents with vectorized numpy operations
- **Extensible Framework**: Plugin system for utility functions, market mechanisms, and economic institutions
- **Theoretical Grounding**: All components connect to established economic theory
- **Real-time Visualization**: Implemented (Pygame HUD + ASCII) with deterministic replay pipeline
- **System Flow**: See [README.md](README.md) for ASCII diagram of Home ↔ Personal ↔ Market architecture

## Budget Constraints and Market Structure

### Pure Barter Economy with Numéraire
- **No physical money**: All trades are direct goods-for-goods exchanges
- **Numéraire choice**: Good 1 serves as numéraire with p₁ ≡ 1 for price normalization
- **Budget constraint**: ∑ⱼ pⱼxᵢⱼ ≤ ∑ⱼ pⱼωᵢⱼ (expenditure ≤ income in numéraire units)
- **Price vector**: p = (1, p₂, p₃, ..., pₙ) where only relative prices matter
- **Market clearing**: ∑ᵢ xᵢⱼ = ∑ᵢ ωᵢⱼ for all goods simultaneously
- **Walrasian auctioneer**: Marketplace facilitates multilateral clearing, holds no inventory

### Home Inventory Specification (Simplified Model - Phase 1 Complete)
- **Simplified inventory management**: Agents load entire home inventory into personal at cycle start via `load_inventory_for_travel()`, travel with full inventory, then deposit back via `deposit_inventory_at_home()` at cycle end
- **No strategic withholding**: Simplified model eliminates strategic withholding complexity while maintaining spatial friction effects
- **LTE price formation**: Uses **total endowment (home + personal)** for theoretical clearing; execution constrained by personal stock  
- **Execution constraint**: Trading limited by **personal inventory** present in marketplace (which equals total inventory under simplified model)
- **Rationing**: Unmet demand/supply rationed proportionally and logged as carry-over; track liquidity_gap[g] = z_market[g] - executed_net[g]. **Critical**: Carry-over is diagnostic only.
- **Carry-over repricing**: Unexecuted orders repriced at next round's equilibrium vector. **Critical**: Carry-over queues **never auto-execute**; orders are recomputed from scratch each round and carry-over is diagnostic only.

### Financing Model (Phase 2 Baseline)
The platform currently operates under a strict **personal-inventory financing** regime:

1. **Financing Source**: Only the goods an agent physically brings into the marketplace (their personal inventory) can finance purchases. There is no abstract borrowing against home inventory, and home endowment not carried is economically inert for that round.
2. **Value Feasibility Invariant**: For each agent each round: \( p \cdot b_i^{exec} \le p \cdot s_i^{exec} + \varepsilon \) (executed buy value cannot exceed executed sell value at equilibrium prices). With full-inventory loading this typically binds only when preferences imply strong rebalancing.
3. **Total vs Personal Distinction**: Prices (LTE) are still computed using participants' total endowments (home + personal) per specification, but order generation and feasibility constraints recognize only personal inventory as a financing source. This creates an intentional wedge between theoretical clearing and executable liquidity.
4. **Travel Costs Interaction**: Travel costs reduce effective wealth conceptually, but under personal financing they do NOT grant additional purchasing power; instead they are diagnostic deductions (reported, not monetized via additional financing). Future modes may treat travel costs as immediate numéraire consumption.
5. **Liquidity Gap Metric**: The difference between theoretical excess demand (using total endowment) and executed net trades (personal constrained) is recorded as a liquidity/financing gap and should never be negative for goods where the agent is constrained on the sell side.

#### Planned Extension: Financing Modes
To support research comparing institutional regimes, a forthcoming toggle will introduce:

```
FinancingMode = Enum('FinancingMode', ['PERSONAL', 'TOTAL_WEALTH'])
```

Mode semantics:
- PERSONAL (default, current behavior): Orders scaled so that executed buy value cannot exceed executed sell value; sell capacity limited by personal inventory snapshot.
- TOTAL_WEALTH (future optional): Agents may issue buy orders backed by the value of their total endowment (or travel-cost-adjusted wealth) regardless of physical composition, subject still to sell capacity constraints for goods they dispose of.

Implementation notes:
- The clearing algorithm will receive a `financing_mode` parameter; invariant validation will branch on value feasibility rule.
- Backward compatibility: Default mode remains PERSONAL; existing tests rely on the stricter barter constraint.
- Additional logging fields: `financing_mode`, `personal_wealth_entry`, `total_wealth_LTE`, and (future) `travel_cost_deduction`.

Economic rationale: Separating price formation scope (total endowment) from financing scope (personal vs total) isolates the effect of liquidity constraints on welfare and rationing dynamics without changing equilibrium pricing fundamentals.


### Movement Cost Model
We adopt a **budget-side travel cost model** for Phase 2:
- **Travel budget**: $w_i = \max(0, p \cdot \omega_i^{total} - \kappa \cdot d_i)$ where $d_i$ is distance traveled this round
- **Travel cost treatment**: Movement costs are deducted from wealth as consumption of good 1 (numéraire), preserving utility's ordinal nature
- **Distance metric**: Manhattan/L1 metric on grid (1 unit per step, 4-directional movement)
- **Tie-breaking**: Lexicographic by (x,y) coordinates then agent ID for deterministic pathfinding across platforms
- **Unit consistency**: $\kappa$ is denominated in units of good 1 per grid step, ensuring budget constraint consistency
- **Default settings**: $\kappa = 0$ in validation scenarios V1-V3 (spatial null tests)
- **Movement policy**: Default movement is **simple greedy** toward marketplace center (one step per round). Movement uses lexicographic tie-breaking (x-direction first, then y-direction) for deterministic behavior. A* pathfinding is planned but not implemented. $\kappa$ is applied as a budget reduction in the simulation runner.

### Welfare Measurement
- **Money-metric welfare**: Report equivalent variation (EV) in units of good 1 (numéraire) using expenditure functions to ensure interpersonal comparability
- **Baseline comparison**: EV measured relative to Phase 1 frictionless allocation
- **EV formula**: EV computed in money space at Phase-1 numéraire-normalized price vector p* (with p*₁=1): EVᵢ = e(p*, x_i^{Phase2}) - e(p*, x_i^{Phase1}) where Phase-2 consumption bundles computed using budget-constrained demand with w_i = max(0, p·ω_i^{total} - κ·d_i). Efficiency loss is ∑ᵢ EVᵢ.
- **Budget constraint**: Phase 2 demand uses budget w_i = max(0, p·ω_i^{total} - κ·d_i) ensuring travel costs are incorporated as budget-side deductions in numéraire units
- **Interpretation**: "Spatial deadweight loss = X units of good 1 foregone"

### Walrasian Solver Implementation

#### Price Normalization:
- **Numéraire choice**: Good 1 as numéraire (p₁ ≡ 1)
- **Solver target**: Find p₂, p₃, ..., pₙ such that excess demand = 0
- **Homogeneity**: All agents have homogeneous degree 0 preferences (price level irrelevant)
- **Residual threshold**: **Primary stop** uses ||Z_market(p)_{2:n}||_∞ < SOLVER_TOL (rest-goods system). The Walras dot |p·Z_market(p)| is a **sanity check on the theoretical system** using participants' **total** endowments; executed trades may differ due to personal-inventory constraints and rationing.

#### Cobb-Douglas Closed Forms:
For agent i with utility Uᵢ(x) = ∏ⱼ xⱼ^αᵢⱼ and ∑ⱼ αᵢⱼ = 1:
- **Demand**: xᵢⱼ(p, ωᵢ) = αᵢⱼ · (p·ωᵢ) / pⱼ
- **Excess demand system**: Z(p) = ∑ᵢ [x_i(p, ω_i^total) - ω_i^total] where ω_i^total = ω_i^home + ω_i^personal

```python
def compute_excess_demand(prices, participant_agents):
    assert participant_agents, "No participants in market this round"
    eps = 1e-10
    prices = np.maximum(prices, eps)  # Guard against division by zero/negative prices (p₁≡1 by normalization, floors apply to j>1)
    n_goods = participant_agents[0].alpha.size
    total_demand = np.zeros(n_goods)
    total_endowment = np.zeros(n_goods)
    
    for agent in participant_agents:  # Only marketplace participants
        omega_total = agent.home_endowment + agent.personal_endowment
        wealth = float(np.dot(prices, omega_total))
        
        # Cobb-Douglas demand: x_ij = alpha_ij * wealth / p_j
        demand = agent.alpha * wealth / prices
        
        total_demand += demand
        total_endowment += omega_total
    
    return total_demand - total_endowment  # Z(p)
```

#### Phase-1 Optimizations:
- **Closed-form expenditure function**: For Cobb-Douglas, $e(p,u) = u \prod_j p_j^{\alpha_j}$ avoids per-agent numerical minimization during EV validation
- **Enhanced solver return**: Optionally return `(p, Z)` so validation can log rest-goods residual without recomputation

#### Fallback for General Utilities:
If a utility plugin doesn't provide `demand()`, solver falls back to numerical utility maximization per agent with tolerance 1e-6 and box constraints x ≥ 0.

#### Solver Stability:
- **Walras' Law**: p·Z(p) ≡ 0 by construction (budget constraints sum to zero)
- **Gross substitutes**: Cobb-Douglas satisfies this (ensures unique equilibrium)
- **Fallback**: Tâtonnement with adaptive step size if fsolve fails

- **Price floors**: Enforce pⱼ ≥ ε via log-price optimization or simplex projection with p₁≡1 and pⱼ>ε for j>1
- **Warm starts**: Use previous round solution p^(t-1) as initial guess for fsolve
- **Endowment scaling**: Scale endowments to mean=1 per good to stabilize Jacobians
### Trading Mechanism (Phase 2: Clean Spatial Protocol)

#### Static Equilibrium Process Per Round:
1. **Agent Movement**: Each agent moves one grid square toward marketplace (simple greedy movement, Manhattan/L1; lexicographic tie-break by (x,y), then agent ID)

2. **Local Theoretical Equilibrium (LTE)**: Compute the Walrasian price vector using **POST-MOVE agents inside the marketplace** and their **total endowments (home + personal)**. This establishes the theoretical clearing prices; actual execution will be constrained by personal inventory. Agents outside the marketplace are excluded from this round's Z_market(p) = 0 system.

3. **Market Access Gate**: Only agents physically inside the 2×2 marketplace can submit trading orders

4. **Order Generation**: Each POST-MOVE marketplace agent submits demand-based orders: For Cobb-Douglas agents, order quantity for good j is `x_ij = alpha_ij * w_i / p_j` minus current personal inventory. Positive = buy order, negative = sell order.

   **Budget-Constrained Wealth**: $w_i = \max(0, p \cdot \omega_i^{\text{total}} - \kappa \cdot d_i) = \max(0, p \cdot (\omega_i^{\text{home}} + \omega_i^{\text{personal}}) - \kappa \cdot d_i)$; **travel-adjusted budget sizes orders only**. **LTE exclusion** is based on **total wealth**: agents with $p \cdot \omega_i^{\text{total}} \leq \epsilon$ are excluded from LTE computation to avoid singular Jacobians (travel costs do not affect pricing participation). Orders use $\Delta_i = x_i^*(p,w_i) - \omega_i^{\text{personal}}$ to determine buy/sell quantities, with execution still limited by personal stock and value feasibility.

5. **Constrained Clearing**: Execute trades at equilibrium prices, limited by personal inventory present in marketplace. Ration unmet orders proportionally by requested quantity.

6. **Carry-over Management**: Unexecuted orders logged as carry-over (quantities only), repriced at next round's equilibrium. **Important**: Carry-over queues do NOT finance current buys; they are informational/diagnostic and repriced next round only.

7. **State Update**: New endowment distribution becomes input for next round

#### Why This Design:
- **Local Theoretical Equilibrium (LTE)**: Prices reflect what full clearing would be using participants' total endowments, establishing theoretical benchmark
- **Liquidity constraints**: Execution limited by personal inventory creates measurable liquidity gap between theoretical and actual trades
- **Spatial friction bites**: Distance to marketplace creates real efficiency costs  
- **Clean measurement**: Compare money-metric welfare with/without movement costs, tracking both LTE and execution gaps
- **No arbitrage confusion**: Single trading mechanism with clear participation rules
- **Pedagogical clarity**: Students see pure effect of spatial access constraints

### Market Throughput & Rationing (Optional Extensions)

#### Throughput Caps:
- **Market capacity**: Auctioneer can clear at most Qₘₐₓ units per good per round
- **Rationing rule**: Proportional to requested quantity; deterministic tie-breakers by agent ID
- **Carry-over persistence**: Quantities only, repriced at next round's equilibrium vector. **Settlement timing**: Carry-over does NOT finance current trades; purely informational until repriced next round.
- **Creates time pressure**: Agents must consider market congestion in movement decisions
- **Configuration requirement**: Set `enable_capacity: false` by default and require `regime: "static_additive_no_congestion"` when true, with tests that relax A* optimality claims accordingly

**⚠️ Note**: Throughput caps create disequilibrium conditions, pushing toward Phase 3 territory. **Enabling caps introduces queueing and non-clearing markets** - avoid during baseline Phase-2 experiments.

## Core Components (Phase 2 Implementation)

The spatial extension maintains Walrasian equilibrium pricing while adding movement and location constraints:

1. **The Agents ("Homo Economicus")**
    i. Each agent has a preference relation represented by a utility function
    ii. Each agent has a personal inventory (carried goods, used for trading)
    iii. Each agent has a home with storage inventory (cannot be traded remotely)
    iv. Each agent has a unique numerical ID, starting at 1
    v. **Trading Rules**: 
        - **Market access only**: Agents can trade only when physically inside the 2×2 marketplace
        - **Local equilibrium prices**: Trades execute at prices computed from current marketplace participants
        - **No bilateral trading**: Eliminates arbitrage channels and path-dependence
        - **Movement costs**: $\kappa \cdot d_i$ budget-side deduction in good-1 per distance unit traveled

2. Agent's Home
    i. Each agent's home is initialized with an inventory described in section 5.
    ii. While in their home, agents can freely transfer items between their own inventory and their home inventory.

3. **Marketplace (Walrasian Auctioneer)**
    i. The marketplace is the center-most rectangle of configurable size (default: 2×2 portion of the NxN grid)
    
    **Configuration**:
    ```yaml
    [config]
    market_width: 2
    market_height: 2
    # The marketplace is the central rectangle of size market_width × market_height.
    # Edge-gates, access checks, and distance_to_market use these dimensions.
    ```
    
    ii. **Local auctioneer**: Computes equilibrium using only current marketplace participants
    iii. **Clearing mechanism**: All buy/sell orders execute simultaneously at equilibrium prices (trades may be rationed by inventory constraints)
    iv. **Rationing**: Proportional allocation when personal inventory insufficient

4. **NxN Grid**
    i. Grid dimensions: Configurable (default: ≈ 2.5√N per side for N agents)
    ii. Agents move one square per turn with movement cost $\kappa$ per unit distance

5. **Goods**
    i. Number of good types: Configurable (default: G=3 to G=5, independent of agent count)
    ii. Total quantity per good: Configurable (default: balanced allocation ensuring interior solutions)
    iii. Initial allocation: Randomly distributed to agent homes at initialization

6. **Preference Generator & Utility Functions**
    i. **Extensible Framework**: Plugin system for different utility functional forms
    ii. **Initial Implementation**: Cobb-Douglas with randomized preference weights
    iii. **Closed-form optimization**: Use analytical demand functions when available

### Initialization Guarantees (Interiority Conditions)
- **Cobb-Douglas preferences**: Draw α from Dirichlet(1,...,1) with clip αⱼ ≥ 0.05 to ensure all goods valued; **renormalize α to sum to 1 after clipping** to preserve Cobb-Douglas formulas
- **Positive supply**: Ensure each good has positive total supply and at least one unit held by some marketplace-reachable agent
- **Endowment distribution**: Random allocation to agent homes ensuring no agent starts with zero wealth at Phase-1 equilibrium prices

## Simulation Flow (Phase 2: Spatial Walrasian)

### Static Equilibrium Approach
The simulation implements a spatial extension of Walrasian equilibrium using a clean protocol that makes spatial frictions measurable:

### Each Simulation Round:
1. **Agent Movement**: Advance toward the marketplace using the configured movement policy. Supported policies:
    - `greedy`: One-step Manhattan descent (reduce |Δx| until zero, then |Δy|) with lexicographic tie-breaking by (x,y) then agent ID for determinism.
    - `astar`: Deterministic A* pathfinding (Manhattan heuristic) generating a full shortest L1 path to the selected marketplace cell. The open-set priority ordering `(f_score, g_score, x, y, sequence)` yields stable tie resolution; neighbor expansion order is fixed (±x first, then ±y). Paths are cached per agent and recomputed only if the goal or blocking geometry changes.
    Both policies are deterministic; A* provides globally shortest routes under the static additive, no-congestion regime assumed in Phase 2.
2. **Local-Participants Price Computation**: Solve ∑ᵢ∈marketplace zᵢ(p) = 0 using **post-move** marketplace agents' total endowments only
3. **Market Order Submission**: Only marketplace agents can submit buy/sell orders
4. **Constrained Clearing**: Execute at equilibrium prices, ration by personal inventory constraints
5. **Carry-over Logging**: Record unexecuted quantities for next round (repriced). **Critical**: Carry-over is diagnostic only, repriced next round, never finances current buys.
6. **State Update**: Update positions, inventories, and welfare measurements

**Termination**: Simulation stops at T ≤ 200 rounds, when all agents reach marketplace with total unmet demand/supply below `RATIONING_EPS` for 5 consecutive rounds, or after `max_stale_rounds` rounds without meaningful progress (default: 50). Log `termination_reason` as "horizon", "market_cleared", or "stale_progress".

### Key Properties:
- **Consistent economics**: LTE price formation uses marketplace participants' total endowments for theoretical clearing; execution constrained by personal inventory
- **Pure spatial friction**: Efficiency loss comes solely from movement costs and market access
- **Measurable welfare effects**: Money-metric equivalent variation in numéraire units
- **No strategic behavior**: Phase 2 neutralizes inventory management via total endowment pricing for LTE computation

### Simulation Objective: Spatial Efficiency Analysis
**Research Question**: How do movement costs and marketplace access restrictions reduce allocative efficiency compared to frictionless Walrasian outcome?

**Measurements**:
- **Efficiency loss**: Money-metric welfare loss (equivalent variation in numéraire units)
- **Travel patterns**: Analyze agent movement strategies and convergence to marketplace
- **Welfare distribution**: Measure how spatial costs affect different agent types
- **Market utilization**: Track marketplace occupancy and queueing patterns

**What Phase 2 Studies**: Pure deadweight loss from spatial separation, optimal market placement, movement cost sensitivity

**What Phase 2 Does NOT Study**: Price discovery, learning, disequilibrium dynamics, strategic behavior (reserved for Phase 3)

## Economic Validation Framework

### Phase 1 (Pure Walrasian) Validation:
- **Equilibrium prices**: Primary convergence: ||Z(p)_{2:n}||_∞ < 1e-8 (rest-goods system); Sanity check: |p·Z(p)| < 1e-8 (Walras' Law validation)
- **Walras' Law**: p·Z(p) < 1e-8 (budget constraints sum to zero)
- **Individual rationality**: Each agent maximizes utility subject to budget constraint
- **Pareto efficiency**: No allocation exists that improves someone without harming others
- **Conservation**: Total goods conserved: ∑ᵢ xᵢ = ∑ᵢ ωᵢ (across all executed trades)

*Primary stop uses ||Z_market(p)_{2:n}||_∞ < SOLVER_TOL. The Walras dot |p·Z_market(p)| is a **sanity check on the theoretical system** using participants' **total** endowments; executed trades may differ due to personal-inventory constraints and rationing.*

### Phase 2 (Spatial Extension) Validation:
- **Spatial efficiency**: Money-metric welfare loss ≥ 0 compared to Phase 1
- **Access restriction**: Only marketplace agents execute trades (verify no bilateral trades)
- **Price consistency (theoretical)**: **Primary stop** uses ||Z_market(p)_{2:n}||_∞ < SOLVER_TOL. The Walras dot |p·Z_market(p)| is a **sanity check on the theoretical system** using participants' **total** endowments; executed trades may differ due to personal-inventory constraints and rationing.
- **Movement optimality (static regime only)**: Deterministic A* pathfinding (implemented) is optimal (shortest L1 path) under current cost model (additive, static, nonnegative, no congestion). **Warning**: If future features introduce dynamic congestion, heterogeneous per-step costs, stochastic obstacles, or throughput caps, previously cached or precomputed A* paths may no longer be optimal; regression tests must then gate claims of optimality.
    - Greedy movement remains available for pedagogical clarity and profiling; it is not guaranteed to yield globally shortest paths when |Δx| and |Δy| trade-offs exist but preserves deterministic lexicographic behavior.
    - When both policies are used in identical static settings, executed distances (total steps) for A* will be ≤ greedy and equal when the greedy’s lexicographic descent already follows a shortest path.
- **Conservation**: Goods conserved across locations, agents, marketplace, and carry-over queues (carry-over diagnostic only)

### Validation Scenarios ✅ COMPLETE

**All V1-V10 Scenarios Implemented and Passing ✅**:

| Scenario | Config | Status | Expected Outcome | Numeric Check |
|----------|--------|---------|------------------|---------------|
| **V1: Edgeworth 2×2** | `config/edgeworth.yaml` | ✅ PASS | Analytic equilibrium match | `‖p_computed - p_analytic‖ < 1e-8` |
| **V2: Spatial Null** | `config/zero_movement_cost.yaml` | ✅ PASS | Phase 2 = Phase 1 exactly | `efficiency_loss < 1e-10` |
| **V3: Market Access** | `config/small_market.yaml` | ✅ PASS | Efficiency loss vs. baseline | `efficiency_loss > 0.1 units of good 1` |
| **V4: Throughput Cap** | `config/rationed_market.yaml` | ✅ PASS | Queue formation, carry-over orders | `uncleared_orders > 0` |
| **V5: Spatial Dominance** | `config/infinite_movement_cost.yaml` | ✅ PASS | Phase 2 efficiency ≤ Phase 1 | `spatial_welfare ≤ walrasian_welfare` |
| **V6: Price Normalization** | `config/price_validation.yaml` | ✅ PASS | p₁ ≡ 1 and rest-goods convergence | `p[0] == 1.0 and ||Z_market(p)[1:]||_∞ < 1e-8` |
| **V7: Empty Marketplace** | `config/empty_market.yaml` | ✅ PASS | Skip price computation and clearing | `prices == None and trades == []` |
| **V8: Stop Conditions** | `config/termination.yaml` | ✅ PASS | Horizon, market clearing, or stale progress | Proper termination detection |
| **V9: Scale Invariance** | `config/scale_test.yaml` | ✅ PASS | Price scaling preserves allocation | Price scaling consistency |
| **V10: Spatial Null (Unit Test)** | `config/spatial_null_test.yaml` | ✅ PASS | κ=0, all agents at market initially | `phase2_allocation == phase1_allocation` |

**Production-Ready Validation Framework**:
- **217/217 tests passing** (205 unit tests + 12 validation scenarios)
- **Complete economic validation** covering all fundamental properties
- **Comprehensive edge case handling** for robust real-world deployment
- **Research-grade validation** suitable for publication-quality experiments


**EV Measurement**: All efficiency_loss values computed in money space at Phase-1 p* with p*₁=1: EV_i = e(p*, x_i^{Phase2}) - e(p*, x_i^{Phase1}) where Phase-2 consumption bundles computed using budget-constrained demand with w_i = max(0, p·ω_i^{total} - κ·d_i). Report Σᵢ EVᵢ in units of good 1.

#### Per-Agent Budget Feasibility Test:
For each round and agent, derive value of executed buys vs sells at equilibrium prices p from Parquet log and assert:
```python
def test_per_agent_budget_feasibility():
    for round_data in simulation_log:
        for agent_id in round_data.executed_trades:
            buy_value = sum(p[g] * executed_buys[agent_id][g] for g in goods)
            sell_value = sum(p[g] * executed_sells[agent_id][g] for g in goods)
            assert buy_value <= sell_value + tolerance  # value feasibility per round
```

#### Critical Edge Cases (Implementation Must Handle):
- **Empty marketplace**: `market_agents = []` → `prices = None, trades = []` (skip price computation and clearing; log "no-price" round)
- **Single participant**: `len(market_agents) < 2` → skip pricing (requires ≥2 agents for meaningful equilibrium)
- **Single good**: `n_goods < 2` → skip pricing (numéraire degeneracy - need ≥2 goods for relative prices)
- **Zero personal inventory**: Agent at marketplace but no goods to trade → valid participant for pricing, but generates empty orders
- **Order priority**: Pro-rata rationing with deterministic tie-breaking by agent ID
- **Order invalidation**: If personal stock changes before execution, invalidate stale orders

#### Settlement Timing:
**CRITICAL**: All order validation uses market-entry inventory snapshots. Personal inventory snapshots are taken at **market entry** time each round. Any mid-round inventory changes invalidate pending orders.

#### Running Validation:
```bash
pytest tests/validation/ --config config/edgeworth.yaml
python scripts/validate_scenario.py --all --output results/validation/
```

### Data Products & Reproducibility

### Structured Logging Schema
Per-round Parquet (and JSONL) records expose a stable, versioned schema. The current additive version is `schema_version: "1.1.0"` (1.0.0 + new spatial aggregate metrics). Minor version bumps reflect strictly additive, backward-compatible field introductions; major bumps denote breaking changes (renames / semantics shifts / removals).

Columns:
- **Schema**: `schema_version: "1.1.0"`
- **Identifiers**: `round, agent_id, timestamp`
- **Inventories**: `x_home[g], x_personal[g], x_total[g]` for each good g
- **Spatial (per-agent)**: 
    - `pos_x, pos_y, in_marketplace`
    - `distance_to_market`: minimum Manhattan/L1 distance to **any** marketplace cell (0 if inside). 4-neighborhood metric consistent with movement rules.
- **Spatial Aggregates (round-level, repeated per row for convenience)**: 
    - `max_distance_to_market`: \( \max_i \text{distance_to_market}_{i,t} \); measures the current spatial frontier (convergence lag of the furthest agent).
    - `avg_distance_to_market`: \( \frac{1}{N} \sum_{i=1}^N \text{distance_to_market}_{i,t} \); population-wide mean distance capturing overall convergence speed. Includes all agents (both in and out of marketplace) to preserve comparability across rounds with changing participation. Chosen over median for linear decomposability in welfare / friction attribution analyses.
- **Financing snapshots**: `personal_at_entry[g], wealth_at_pricing, in_market_bool` (guard against mid-round financing mutations)
- **Economics**: `p[g], z_market[g], executed_net[g], liquidity_gap[g], utility, move_cost, equivalent_variation`
- **LTE vs Execution**: `z_market[g]` (theoretical excess demand under total endowments) vs `executed_net[g]` (actual executed net trades) vs `liquidity_gap[g] = z_market[g] - executed_net[g]`
- **Sign conventions**: `z_market[g] = demand - endowment` (+ = excess demand), `executed_net[g] = buys - sells` (+ = net buyer), `liquidity_gap[g] > 0` = constrained by personal inventory
- **Termination tracking**: `termination_reason` ("horizon", "market_cleared", "stale_progress") logged in final round
- **Metadata**: `git_sha, config_hash, random_seed`

Definitions & Notes:
- Both aggregate distance metrics are deterministic given positions & marketplace geometry, enabling downstream replay validation.
- Aggregates repeat per record to simplify columnar analytics (no separate master table join required); they are numerically identical across all rows for a fixed `round`.
- `avg_distance_to_market` provides a smoother early-round convergence signal than `max_distance_to_market`, which can plateau if a single straggler stalls. Researchers can approximate a simple spatial convergence index via `avg_distance_to_market / max_distance_to_market_initial`.
- Addition of these aggregates from 1.0.0 → 1.1.0 required only a minor version bump (additive). Existing parsers that ignore unknown columns remain valid.

### Reproducibility Guarantees
- **Configuration**: Every experiment via `python scripts/run_simulation.py --config path.yaml --seed 42`
- **Deterministic**: Fixed random seeds with reproducible numpy/scipy versions, plus `random.seed(seed)` for Python RNG coverage
- **Cross-machine stability**: `OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1` and NumPy `PCG64` with fixed seeds to reduce solver jitter
- **Headless**: `--no-gui` mode for batch experiments and CI/CD
- **Analysis**: `make figures` regenerates all plots from Parquet logs
- **Version control**: Git SHA and dependency versions logged with each run

## Performance Optimization

### Computational Bottlenecks
- **Per-round equilibrium**: Profile equilibrium solver; **⚠️ WARNING: Price caching across rounds violates local-participants consistency; only use with --frozen-participation flag**
- **A* pathfinding**: Precompute distance field to marketplace, use greedy descent for ~O(N) speedup
- **Vectorized operations**: All agent properties as numpy arrays for 100+ agent performance

### Scalability Targets
- **Agent count**: Target 100+ agents with <30 seconds per 1000 rounds (reference hardware, G≤5)
- **Memory efficiency**: Pre-allocated arrays, minimal copying
- **Warm starts**: Use previous equilibrium as initial guess for fsolve

### Default Configuration Parameters
- **Goods count**: G ∈ {2,...,5} for most educational scenarios; single high-G stress config for performance testing. Goods are perfectly divisible (ℝ⁺), so proportional rationing introduces no rounding.
- **Grid size**: Default formula: `grid_side = max(15, ceil(2.5 * sqrt(n_agents)))` to decouple from agent count
- **Stop conditions**: Default horizon T=200 rounds, earlier if all agents reached marketplace and total unmet demand/supply below `RATIONING_EPS` for 5 consecutive rounds, or after `max_stale_rounds` without progress (default: 50). **Note**: Uses actual market clearing metrics, not carry-over state.
- **Price caching**: For profiling only - caveat that cached prices risk inconsistency when marketplace participation changes

## Implementation Interfaces & Invariants

### Core Protocols

#### UtilityProtocol:
```python
class UtilityProtocol:
    def u(self, x: np.ndarray) -> float:
        """Compute utility of consumption bundle x"""
        
    def demand(self, p: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Optimal demand given prices p and endowment w (optional closed form)"""
        
    def grad_u(self, x: np.ndarray) -> np.ndarray:
        """Gradient of utility function (optional for optimization)"""
```

#### MovementPolicy:
```python
class MovementPolicy:
    def step(self, agent_state: AgentState, world_state: WorldState) -> Position:
        """Choose next position given current state and world information"""
        
    def path_cost(self, start: Position, end: Position) -> float:
        """Compute movement cost between positions"""
```

#### EquilibriumSolver:
from typing import Tuple, Optional, List
import numpy as np
import scipy.optimize

```python
def solve_equilibrium(agents: List[Agent], 
                     normalization: str = 'good_1',
                     endowment_scope: str = 'total') -> Tuple[Optional[np.ndarray], float, float, str]:
    """Solve for market-clearing prices using specified endowment scope
    
    Args:
        agents: Marketplace participants only (for LTE computation)
        normalization: Price normalization method ('good_1' sets p[0] = 1.0)
        endowment_scope: 'total' (home+personal) or 'personal' (personal only)
    
    Returns:
        Tuple of (prices, z_rest_inf, walras_dot, status):
        - prices: Price vector with p[0] = 1.0, or None if no participants
        - z_rest_inf: ||Z_market(p)_{2:n}||_∞ (primary convergence metric)
        - walras_dot: |p·Z_market(p)| (Walras' Law sanity check)
        - status: 'converged', 'no_participants', 'failed', 'max_iterations'
    """
```

**Numerical Stability Guidelines:**
- **Primary convergence test**: ||Z(p)_{2:n}||_∞ < 1e-8 (the "rest-goods" system), not just |p·Z(p)| (Walras' Law can be near-zero away from equilibrium)
- **Parameterization**: Optimize in log-price space with p₁ ≡ 1 to enforce positivity without manual clipping; **primary positivity enforcement** through log-parameterization
- **Warm starts**: Start from p^{t-1} and rescale endowments to mean 1 per good for numerical conditioning
- **Initial prices**: Start with uniform prices (all goods equal value) for robust convergence. **For the very first round** when no prior prices exist, use `np.ones(n_goods-1)` as the initial guess for the rest-goods vector p_{2:n}, ensuring p₁ ≡ 1 normalization.
- **Bounds checking**: Enforce p ≥ 1e-10 as **last-ditch safety guard** to prevent numerical degeneracy when log-param fails
- **Jacobian conditioning**: Use regularization if condition number > 1e12
- **Algorithm recommendation**: Newton-Raphson for smooth Cobb-Douglas, gradient descent for general utilities

**Failure Modes & Remedies:**
- **fsolve fails**: Fall back to tâtonnement with adaptive step size
- **Negative price after log-param**: Project back to feasible region
- **NaN/Inf guard**: If Z_market(p) or p contains NaN/Inf, reset to uniform prices, halve the tâtonnement step size, and retry from the last feasible iterate
- **Queue growth without bound**: Raise throughput warning and consider capacity constraints

#### MarketClearing:
```python
def execute_constrained_clearing(agents: List[Agent], prices: np.ndarray, 
                               capacity: Optional[np.ndarray] = None) -> List[Trade]:
    """Clear market orders subject to personal inventory and optional throughput limits
    
    ALGORITHM CONTRACT: Constrained Clearing with Proportional Rationing
    1. Generate orders: For each marketplace agent, compute desired quantity per good
       (Cobb-Douglas: x_ij = alpha_ij * wealth / p_j minus current personal inventory)
    2. Separate buy/sell orders by sign
    3. For each good j:
       a. Execute sells up to min(total_sell_orders_j, personal_inventory_constraint_j)
       b. Execute buys up to min(total_buy_orders_j, executed_sells_j, capacity_j)
       c. If demand exceeds supply: ration buy orders proportionally by requested quantity
       d. If supply exceeds demand: ration sell orders proportionally by offered quantity
    4. Log unexecuted order quantities as carry-over for next round (repriced at future equilibrium). **Critical**: Carry-over is diagnostic only, repriced next round, never finances current buys.
    5. Return list of executed trades with (agent_id, good_id, quantity, price) tuples
    
    MATHEMATICAL CONTRACT (prevents money from thin air):
    For each good g:
      B_g = Σ_i b_ig                           # requested buys
      S_g = Σ_i s_ig                           # available sells (capped by personal inventory at market entry)
      Q_g = min(B_g, S_g)                      # executed volume
      
    Proportional fills:
      b_ig_exec = Q_g * b_ig / max(B_g, ε)     # buy-side proportional allocation
      s_ig_exec = Q_g * s_ig / max(S_g, ε)     # sell-side proportional allocation
      
    Inventory constraints:
      s_ig_exec ≤ s_ig^max                     # cannot sell more than personal stock at market entry
      
    Per-agent value feasibility (each round):
      Σ_g p_g * b_ig_exec ≤ Σ_g p_g * s_ig_exec  # buy value ≤ sell value
      
    Executed-layer conservation:
      Σ_i b_ig_exec = Σ_i s_ig_exec = Q_g      # goods balance at executed level
    
    INVARIANTS ENFORCED:
    - Personal inventory constraint: sells ≤ agent's personal stock at marketplace entry
    - Value feasibility: All value-feasibility checks use the **entry snapshot**, not mutable per-round inventory
    - Market balance: total buys = total sells for each good
    - Price consistency: all trades at equilibrium prices p from solve_equilibrium()
    - Conservation: no goods created or destroyed, only transferred between agents
    
    Args:
        agents: Marketplace participants with orders
        prices: Equilibrium price vector from solve_equilibrium
        capacity: Optional per-good throughput limits
        
    Returns:
        List of executed trades (may be rationed by personal inventory)
    """
```

### Economic Invariants
All implementations must satisfy:
- **Nonnegativity**: x ≥ 0 for all consumption bundles
- **Conservation**: ∑ᵢ xᵢ = ∑ᵢ ωᵢ before and after all operations
- **Budget feasibility**: p·x ≤ p·ω for all agents  
- **Per-round value feasibility**: For each agent in the market, value of same-round sells from personal stock at p ≥ value of buys at p
- **Personal inventory constraints**: Per-good sells ≤ personal inventory at market-entry snapshot
- **Zero-wealth exclusion**: Agents with wealth ≤ FEASIBILITY_TOL excluded from LTE pricing (prevents singular Jacobians, α-only demand with zero wealth)
- **Snapshot integrity**: `personal_at_entry` must equal actual personal inventory when agent enters marketplace (prevents mid-round financing)
- **Wealth consistency**: `wealth_at_pricing` must equal `p · total_endowment` at price computation time for all marketplace participants
- **Market status validation**: `in_market_bool` must match actual geometric position within marketplace boundaries
- **Home/personal conservation**: Transfers between ω^home and ω^personal when at home must conserve ω^total exactly (within FEASIBILITY_TOL)
- **Walras' Law**: ∑ᵢ pⱼ·zᵢⱼ(p) = 0 for any price vector
- **Normalization**: p₁ ≡ 1 (numéraire constraint)

### Performance Implementation Guidelines
- **Agent data structures**: Use numpy arrays for endowments, vectorized operations for excess demand computation
- **Spatial indexing**: Hash table for O(1) agent location lookup, spatial partitioning for movement pathfinding
- **Equilibrium solving**: Cache Jacobian computations for Cobb-Douglas utilities, reuse previous prices as warm start
- **Memory efficiency**: Preallocate arrays for agent states, reuse workspace arrays across rounds
- **Bottleneck profiling**: JIT-compile equilibrium solver and excess demand functions with numba for 10x+ speedup

## Technical Implementation

### Dependencies
- **Core**: numpy, scipy (optimization and numerical methods)
- **Visualization**: pygame (real-time agent movement display)
- **Performance**: numba (optional JIT compilation for bottlenecks)
- **Development**: pytest, black, mypy, flake8
- **Analysis**: pandas, matplotlib (data analysis and plotting)

### Key Design Principles
- **Theoretical foundation**: Start with rigorous Walrasian equilibrium, then add realistic frictions
- **Vectorized operations**: All agent properties and calculations use numpy arrays for 100+ agent performance
- **Spatial optimization**: A* pathfinding with caching for efficient movement
- **Economic validation**: Automated testing of all economic invariants and theoretical properties
- **Modular progression**: Clean separation between economic theory phases
- **Reproducibility**: All experiments configurable via YAML with fixed random seeds

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run validation tests
pytest tests/validation/

# Run specific scenario  
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42

# Generate figures from logs
make figures
```

## Future Extensions

### Phase 3: Local Price Formation
- **Bilateral bargaining**: Nash bargaining solution for co-located agents outside marketplace
- **Market mechanisms**: Continuous double auction with order books in marketplace
- **Spatial price variation**: Prices differ across locations, arbitrage opportunities drive movement
- **Market microstructure**: Bid-ask spreads, market makers, liquidity provision

### Phase 4: Advanced Economics  
- **Production**: Firms, technology, factor markets with spatial location decisions
- **Money and credit**: Monetary economics, banking, financial markets
- **Institutions**: Contracts, property rights, governance structures
- **Behavioral economics**: Bounded rationality, learning, social preferences
- **Policy analysis**: Framework for testing economic hypotheses and interventions

### Testing That Teaches Economics

Economics-aware tests for contributors:
```python
def test_walras_law_residual():
    """Walras' Law: price vector times excess demand must equal zero"""
    marketplace_agents = [agent for agent in agents if agent.at_marketplace()]
    prices, z_rest_inf, walras_dot, status = solve_equilibrium(marketplace_agents)
    assert status == 'converged', f"Solver failed with status: {status}"
    assert z_rest_inf < 1e-8, f"Rest-goods norm too high: {z_rest_inf}"
    assert abs(walras_dot) < 1e-8, f"Walras' Law violated: {walras_dot}"

def test_goods_conservation():
    """Total goods conserved across all bilateral and market operations"""
    initial_total = np.sum([agent.home_endowment + agent.personal_endowment for agent in agents], axis=0)
    run_simulation_round()
    final_total = np.sum([agent.home_endowment + agent.personal_endowment for agent in agents], axis=0)
    assert np.allclose(initial_total, final_total, atol=1e-10)

def test_spatial_dominance():
    """With infinite movement costs, Phase 2 efficiency ≤ Phase 1 efficiency"""
    efficiency_frictionless = run_phase1_simulation()
    efficiency_spatial = run_phase2_simulation(movement_cost=float('inf'))
    assert efficiency_spatial <= efficiency_frictionless

def test_movement_regime_optimality():
    """A* optimality only holds under static_additive_no_congestion regime"""
    # Test should only assert optimality under specific regime conditions
    regime = "static_additive_no_congestion"
    if regime == "static_additive_no_congestion":
        # A* pathfinding should be optimal for this configuration
        optimal_path = compute_optimal_path(start, goal, costs)
        astar_path = astar_pathfinding(start, goal, costs)
        assert path_cost(astar_path) == path_cost(optimal_path)
    else:
        # Skip optimality assertion for other regimes
        pytest.skip(f"A* optimality not guaranteed under regime: {regime}")
```

### Research Applications
- **Optimal market design**: How should market size and location affect welfare?
- **Transportation economics**: Model shipping costs, infrastructure investment
- **Urban economics**: Spatial equilibrium with heterogeneous locations
- **Development economics**: Market access and rural-urban linkages
- **Industrial organization**: Firm location and competition with spatial differentiation

## Implementation Examples

The following code sketches are for illustration only. The Agent class has been implemented in `src/core/agent.py` with comprehensive testing. Remaining modules need implementation as indicated by the Implementation Priorities:

from typing import Tuple, Optional, List
import numpy as np
import scipy.optimize

```python
# Walrasian equilibrium solver with proper normalization
def solve_equilibrium(agents: List[Agent], 
                     normalization: str = 'good_1',
                     endowment_scope: str = 'total') -> Tuple[Optional[np.ndarray], float, float, str]:
    """Solve for market-clearing prices with specified normalization"""
    n_goods = agents[0].alpha.size  # Infer from agent data
    if normalization == 'good_1':
        def excess_demand_normalized(p_rest):
            prices = np.concatenate([[1.0], p_rest])
            return aggregate_excess_demand(prices, agents, endowment_scope)[1:]
        
        p_rest = scipy.optimize.fsolve(excess_demand_normalized, np.ones(n_goods-1))
        prices = np.concatenate([[1.0], p_rest])
        z_rest_inf = np.linalg.norm(excess_demand_normalized(p_rest), ord=np.inf)
        walras_dot = abs(np.dot(prices, aggregate_excess_demand(prices, agents, endowment_scope)))
        return (prices, z_rest_inf, walras_dot, 'converged')

# Clean spatial trading protocol
def run_simulation_round(agents: List[Agent], grid: Grid, regime="static_additive_no_congestion") -> SimulationState:
    """Execute one round of spatial Walrasian simulation
    
    Args:
        regime: Movement optimality regime. A* pathfinding is optimal ONLY under 
                "static_additive_no_congestion" - fails with capacity/dynamic participation
    """
    # 1. Move agents toward marketplace first (A* optimal under current regime assumptions)
    assert regime == "static_additive_no_congestion", f"A* optimality not guaranteed under regime: {regime}"
    for agent in agents:
        agent_state = AgentState(position=agent.position, endowment=agent.get_total_endowment())
        world_state = WorldState(grid=grid, prices=None)  # Movement doesn't use current prices
        new_pos = agent.movement_policy.step(agent_state, world_state)
        grid.move_agent(agent, new_pos)
    
    # 2. Price on POST-MOVE marketplace participants' TOTAL endowments
    market_agents = grid.get_agents_in_marketplace()
    
    # Filter out zero-wealth agents to prevent singular Jacobians in solver
    viable_agents = []
    for agent in market_agents:
        wealth_i = np.dot(prices_prev_or_uniform, agent.get_total_endowment()) if prices_prev_or_uniform else 1.0
        if wealth_i > FEASIBILITY_TOL:  # Exclude degenerate wealth agents
            viable_agents.append(agent)
    
    n_goods = viable_agents[0].alpha.size if viable_agents else 0
    
    # Edge gates: prevent creative "partial pricing" in Phase-2
    if viable_agents and len(viable_agents) >= 2 and n_goods >= 2:
        prices, z_rest_inf, walras_dot, status = solve_equilibrium(
            viable_agents,
            endowment_scope="total",        # {"personal","total"}
            normalization="good_1"          # p1 ≡ 1
        )
        
        # Validate convergence and log diagnostics
        assert status == 'converged', f"Solver failed: {status}"
        assert z_rest_inf < SOLVER_TOL, f"Poor convergence: ||Z_rest||_∞ = {z_rest_inf}"
        
        # Log convergence diagnostics for monitoring
        logging.info(f"Round convergence: ||Z_rest||_∞={z_rest_inf:.2e}, Walras={walras_dot:.2e}")
    else:
        prices, z_rest_inf, walras_dot, status = None, 0.0, 0.0, 'no_viable_participants'
    
    # 3. Clear marketplace orders only (constrained by personal inventory)
    if prices is not None:
        execute_constrained_clearing(market_agents, prices)
    else:
        # Empty marketplace round: no prices computed, trades = []
        pass
    
    # 4. Update state for next round
    return SimulationState(agents=agents, grid=grid, prices=prices)

# Market clearing with optional throughput constraints
def execute_constrained_clearing(agents: List[Agent], prices: np.ndarray, 
                               capacity: Optional[np.ndarray] = None) -> List[Trade]:
    """Clear market orders subject to personal inventory and optional throughput limits"""
    orders = [agent.generate_market_orders(prices) for agent in agents]
    
    if capacity is not None:
        return execute_rationed_clearing(orders, prices, capacity)
    else:
        # Constrained clearing (limited by personal inventory)
        return execute_constrained_match(orders, prices)  # <- new primitive
```

#### Constrained Matching Math (execute_constrained_match):
```
Algorithm Contract (prevents money from thin air):

For each good g:
  B_g = Σ_i b_ig                           # requested buys
  S_g = Σ_i s_ig                           # available sells (capped by personal inventory at market entry)
  Q_g = min(B_g, S_g)                      # executed volume
  
Proportional fills:
  b_ig_exec = Q_g * b_ig / max(B_g, ε)     # buy-side proportional allocation
  s_ig_exec = Q_g * s_ig / max(S_g, ε)     # sell-side proportional allocation
  
Inventory constraints:
  s_ig_exec ≤ s_ig^max                     # cannot sell more than personal stock at market entry
  
Per-agent value feasibility (each round):
  Σ_g p_g * b_ig_exec ≤ Σ_g p_g * s_ig_exec  # buy value ≤ sell value
  
Executed-layer conservation:
  Σ_i b_ig_exec = Σ_i s_ig_exec = Q_g      # goods balance at executed level
```

**Implementation Test (promote to first-class pytest):**
```python
def test_constrained_clearing_invariants(simulation_log):
    """Test that constrained clearing preserves economic invariants
    
    ENFORCES CLEARING CONTRACT:
    - Per-agent value feasibility: buy_value ≤ sell_value per round
    - Per-good executed conservation: total_buys = total_sells
    - Personal inventory constraint enforcement
    """
    for round_data in simulation_log:
        prices = round_data.prices
        for agent_id in round_data.executed_trades:
            # Value feasibility per agent per round (uses FEASIBILITY_TOL)
            buy_value = sum(prices[g] * round_data.executed_buys[agent_id][g] for g in range(len(prices)))
            sell_value = sum(prices[g] * round_data.executed_sells[agent_id][g] for g in range(len(prices)))
            assert buy_value <= sell_value + FEASIBILITY_TOL, f"Agent {agent_id} round {round_data.round}: buy_value={buy_value} > sell_value={sell_value}"
            
        # Conservation per good (uses FEASIBILITY_TOL)
        for g in range(len(prices)):
            total_buys = sum(round_data.executed_buys[agent_id][g] for agent_id in round_data.executed_trades)
            total_sells = sum(round_data.executed_sells[agent_id][g] for agent_id in round_data.executed_trades)
            assert abs(total_buys - total_sells) < FEASIBILITY_TOL, f"Good {g} round {round_data.round}: buys={total_buys} ≠ sells={total_sells}"
```

## Implementation Readiness Checklist

With the Agent framework complete, before implementing the economic engine:
- [x] ✅ Agent class with Cobb-Douglas utilities (COMPLETED)
- [x] ✅ Comprehensive testing suite with 15 unit tests (COMPLETED)
- [ ] Understand numéraire normalization (p₁ ≡ 1)
- [ ] Know primary convergence test (rest-goods norm ||Z(p)_{2:n}||_∞ < 1e-8)
- [ ] Grasp local-participants pricing principle (POST-MOVE marketplace agents only)
- [ ] Review constrained clearing algorithm contract in API section
- [ ] Set up validation scenario V1 (Edgeworth 2×2) for testing
- [ ] Understand settlement timing (market-entry inventory snapshots)
- [ ] Know common pitfalls (pricing before movement, wrong convergence test)
- [ ] Review edge cases (empty marketplace, single participant, single good)

## Repository Scaffolding

**Recommended folder structure for first PRs:**
- `src/core/` (agents, state, utils)
- `src/econ/` (solver, clearing)
- `src/spatial/` (grid, movement) 
- `scripts/run_simulation.py`, `scripts/validate_scenario.py`
- `tests/validation/` (V1–V8), `tests/unit/` (solver, clearing)
- `config/edgeworth.yaml`, `config/zero_movement_cost.yaml`

**Essential data types for logging:**
```python
@dataclass
class Trade:
    agent_id: int
    good_id: int
    quantity: float  # + = buy, - = sell
    price: float
    round: int

@dataclass  
class SimulationState:
    round: int
    prices: Optional[np.ndarray]  # None if no pricing this round
    trades: List[Trade]
    agent_positions: Dict[int, Tuple[int, int]]
    marketplace_participants: Set[int]
```

Final statistics on initial vs ending inventory, initial vs ending utility, and items sold/purchased for each agent are printed to the console, with real-time visualization via pygame.