# Teaching Handout: Walrasian Foundations (Spatial Twist) & Rationing Thought Experiment

Version: 2025-09-21  
Audience: Intermediate Micro / Computational / Early Graduate GE  
Prereqs: Cobb-Douglas demand, Walrasian equilibrium definition, basic Python/CLI familiarity

---
## Table of Contents
1. Overview & Pedagogical Arc  
2. Workflow 1a: Walrasian Foundations with a Spatial Twist  
3. Workflow 1e: Rationing / Throughput Thought Experiment  
4. Comparative Synthesis  
5. Assessment & Extensions  
6. Quick Reference (Commands & Metrics)  
7. Common Pitfalls & Instructor Tips

---
## 1. Overview & Pedagogical Arc
This handout provides two structured instructional workflows:
- **1a (Foundations + Spatial Neutrality)**: Show that embedding agents in space *does not* change Walrasian outcomes when movement is costless and access is universal (κ = 0). This cements core GE logic while building intuition for how spatial frictions will matter once introduced.
- **1e (Rationing / Capacity Wedge)**: Introduce execution throughput limits that prevent full clearing despite correct equilibrium prices—highlighting a second wedge (implementation constraint) distinct from spatial access or travel costs.

Recommended sequencing: Foundations → Spatial Neutrality → Access Frictions (small marketplace) → Rationing Capacity → Synthesis. 

Learning trajectory: Theory (equilibrium) → Space (participation timing) → Cost (budget wedge) → Capacity (execution wedge) → Welfare decomposition.

---
## 2. Workflow 1a: Walrasian Foundations with a Spatial Twist
### Objectives
1. Compute/verify Walrasian equilibrium (p with p₁ ≡ 1) and rest-goods convergence.  
2. Distinguish theoretical excess demand (based on total endowments of participants) from executed trades (constrained by physical presence).  
3. Demonstrate spatial neutrality under κ=0 and prompt, universal market access.  
4. Measure spatial deadweight loss when κ>0 or access is staggered.  
5. Connect welfare changes to money-metric equivalent variation (EV) in numéraire units.

### Config Progression
| Step | Config | Purpose |
|------|--------|---------|
| 1 | `config/edgeworth.yaml` | Analytic 2×2 benchmark (sanity & confidence) |
| 2 | `config/zero_movement_cost.yaml` | Spatial embedding, κ=0 → neutral outcome |
| 3 | `config/small_market.yaml` | Restricted access; delayed participation |
| 4 (opt) | `config/infinite_movement_cost.yaml` | Spatial dominance bound / autarky tendency |

### 70–80 Minute Session Plan
| Time | Activity | Key Emphasis |
|------|----------|--------------|
| 0–10 | Derive Cobb-Douglas demand & z(p) | α-weighted wealth allocation |
| 10–20 | Run Edgeworth scenario | Verify prices & residual norms |
| 20–25 | Introduce spatial layer conceptually | Movement order: move → price → clear |
| 25–35 | Run κ=0 spatial config | Show identical allocation & welfare |
| 35–50 | Run access-friction (small_market) | Participation lag & liquidity gap concept |
| 50–60 | Compute welfare loss vs baseline | EV interpretation (units of good 1) |
| 60–70 | Limiting cases κ→0 / κ→∞ | Continuity & dominance arguments |
| 70–80 | Q&A & recap invariants | p₁ ≡ 1, conservation, value feasibility |

### Key Metrics
- Prices and convergence: p, ||Z_rest||∞, |p·Z| (sanity)  
- Participation per round; distance-to-market distributions  
- Theoretical excess demand vs executed net (liquidity narrative)  
- EV relative to frictionless baseline  
- Rounds to “near-full access” (all agents in marketplace)

### Discussion Prompts
- Why does κ=0 neutralize space?  
- What economic role does normalization p₁ ≡ 1 play?  
- How does excluding non-participants from pricing preserve locality?  
- Would early credit (TOTAL_WEALTH financing) alter initial liquidity gaps?

### Quick Exercise (In-Class)
Given equilibrium prices p and an agent’s α, ω, compute x* and z. Confirm Σ z = 0 (rest goods). 

### Extensions
- Parameter sweep over κ; plot EV loss curve.  
- Introduce heterogeneity in initial distances; analyze first-round welfare inequality.  
- Compare speed-of-convergence metrics under different market sizes.

---
## 3. Workflow 1e: Rationing / Throughput Thought Experiment
### Objectives
1. Show how capacity constraints impose execution rationing even with correct equilibrium prices.  
2. Illustrate proportional rationing: Q_g = min(B_g, S_g, capacity_g); scale individual orders.  
3. Differentiate price-level scarcity signals from implementation bottlenecks.  
4. Quantify welfare impact of delayed rebalancing.  
5. Analyze incidence: which agents systematically experience unmet demand.

### Core Config
`config/rationed_market.yaml` (introduces per-good capacity limits). Use baseline (`zero_movement_cost.yaml` or `small_market.yaml`) for contrast.

### 55–60 Minute Session Plan
| Time | Activity | Focus |
|------|----------|-------|
| 0–8 | Recap unconstrained clearing invariants | Conservation & feasibility |
| 8–18 | Introduce capacity model | Distinguish theoretical Z vs executable volume |
| 18–30 | Run rationed scenario | Observe persistent unmet orders |
| 30–42 | Manual proportional rationing walkthrough | Compute scaling factors |
| 42–52 | Welfare path & time-to-clear analysis | EV vs baseline trajectory |
| 52–60 | Policy levers & reflection | Capacity investment vs spatial policies |

### Key Quantities
- For each good g: B_g (desired buys), S_g (desired sells), capacity_g, executed volume Q_g.  
- Allocation ratios: b_i_exec / b_i_req, s_i_exec / s_i_req.  
- Carry-over (unexecuted portion) time series.  
- Rounds until unmet demand below ε.  
- EV delta under increased capacity counterfactual.

### Classroom Exercise
Provide a snapshot table:
```
Agent | Desired Buy g2 | Desired Sell g1 | Capacity Share? | Executed Buy g2
```
Students compute proportional fill and verify Σ buys = Σ sells and per-agent buy value ≤ sell value.

### Discussion Prompts
- When does price fail to accelerate clearing?  
- Fairness of proportional rationing vs potential priority schemes.  
- Could rationing ever *improve* welfare distribution? Under what equity lens?

### Extensions
- Heterogeneous per-good capacities → sectoral bottlenecks.  
- Variant priority rule (instructor modifies code) to provoke fairness debate.  
- Combine κ>0 + capacity to layer wedges; decompose welfare loss components.

---
## 4. Comparative Synthesis
| Dimension | Workflow 1a (Spatial Foundations) | Workflow 1e (Rationing) |
|----------|----------------------------------|--------------------------|
| Primary Wedge | Delayed participation / travel cost | Execution throughput limit |
| Prices | Walrasian LTE on present participants | Same; not altered by capacity |
| Execution Gap Driver | Absent agents / inventory locality | Binding Q_g < B_g, S_g |
| Welfare Mechanism | Reduced ability to trade early or at all | Delayed completion of desired rebalancing |
| Policy Lever | Reduce κ, enlarge marketplace | Increase capacity, scheduling reforms |
| Diagnostic Metric | Participation lag, distance distribution | Fill ratios, capacity utilization |

Pedagogical Bridge: Use liquidity gap narrative from 1a to motivate richer decomposition (requested vs executed vs unmet) in 1e.

---
## 5. Assessment & Extensions
### Suggested Assessments
- Short quiz: Explain why theoretical pricing uses total endowments (not just personal inventory).  
- Problem set: Compute proportional rationing outcomes and EV under two capacity levels.  
- Mini-project: κ–capacity grid sweep → surface plot of efficiency loss.  
- Reflection: Real-world analog (port congestion, exchange trading halts, electricity grid).  

### Capstone Ideas
1. **Policy Optimization**: Given budget allowing either κ reduction or capacity increase, allocate scarce investment to minimize EV loss.  
2. **Liquidity Mode Counterfactual**: Prototype TOTAL_WEALTH financing and compare early-round allocation speed.  
3. **Algorithmic Variant**: Replace greedy movement with A*; measure change in time-to-full-participation and residual welfare loss.

---
## 6. Quick Reference (Commands & Metrics)
### Core Commands (Instructor Demo)
```bash
# Baseline analytic check
python scripts/run_simulation.py --config config/edgeworth.yaml --seed 42 --no-gui

# Spatial neutral (κ=0)
python scripts/run_simulation.py --config config/zero_movement_cost.yaml --seed 42 --no-gui

# Access friction
python scripts/run_simulation.py --config config/small_market.yaml --seed 42 --no-gui

# Rationing / capacity scenario
python scripts/run_simulation.py --config config/rationed_market.yaml --seed 42 --no-gui
```

### Metrics to Collect (Sample Logging Fields)
- `round`, `agent_id`, `pos_x`, `pos_y`, `in_marketplace`  
- `p[g]`, `z_market[g]`, `executed_net[g]`  
- (Planned enrichment) `requested_buy[g]`, `requested_sell[g]`, `unmet[g]`  
- `utility`, `equivalent_variation`, `distance_to_market`  
- `liquidity_gap[g] = z_market[g] - executed_net[g]`  

### Interpretation Cheatsheet
| Signal | Interpretation |
|--------|----------------|
| Persistent positive liquidity_gap[g] | Demand constrained by access/inventory or capacity |
| Slow decline in carry-over | Binding capacity or recurring arrival lag |
| EV loss near zero, despite delays | Trades not highly redistributive across preferences |
| Sharp EV drop when κ increases slightly | High marginal cost of early trade—steep welfare gradient |

---
## 7. Common Pitfalls & Instructor Tips
### Pitfalls
- Using |p·Z| alone for convergence (must check rest-goods norm).  
- Interpreting zero execution as absence of demand (may be capacity binding).  
- Treating carry-over as guaranteed future fill (it is diagnostic only; orders recompute each round).  
- Forgetting normalization p₁ ≡ 1 when comparing runs.

### Tips
- Pre-run simulations to cache outputs for live discussion.  
- Start with a visual (simple grid sketch) to anchor marketplace geometry.  
- Emphasize invariants (conservation, feasibility) as “economic unit tests.”  
- Encourage students to hypothesize metric trends before revealing plots (active prediction).  
- Use EV in units of good 1 to keep welfare comparisons concrete.

---
## 8. Glossary (Optional Quick Reminders)
- **LTE (Local Theoretical Equilibrium)**: Equilibrium computed from currently present marketplace participants' total endowments.  
- **Liquidity Gap**: Difference between theoretical excess demand and executed net trade (constraint diagnostic).  
- **Capacity Q_g**: Maximum executable volume for good g in a round; induces proportional rationing.  
- **Equivalent Variation (EV)**: Money-metric welfare measure relative to frictionless baseline at p*.  

---
## 9. Roadmap Hooks (For Advanced Classes)
- Upcoming logging fields: requested vs executed vs unmet per agent/good.  
- Financing modes (PERSONAL vs TOTAL_WEALTH) for liquidity regime comparison.  
- Pathfinding sophistication (A*) to discuss algorithmic efficiency vs economic outcome timing.  

---
*End of Handout*
