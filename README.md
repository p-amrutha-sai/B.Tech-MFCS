# Multi-Agent Path Finding (MAPF) Reduced to Boolean Satisfiability (SAT)
### Mathematical Foundations of Computer Science (MFCS) Term Project Prototype

---

## 1. Project Overview

**Multi-Agent Path Finding (MAPF)** is the fundamental algorithmic problem of computing collision-free paths for a set of $k$ agents navigating a shared graph $G = (V, E)$ from their respective starting positions $S = \{s_0, s_1, \dots, s_{k-1}\}$ to goal positions $G = \{g_0, g_1, \dots, g_{k-1}\}$.

While finding optimal collision-free paths is **NP-hard** for both makespan minimization and sum-of-costs, modern **Boolean Satisfiability (SAT)** solvers powered by Conflict-Driven Clause Learning (CDCL) provide an exceptionally efficient mechanism for global constraint reasoning. By encoding MAPF as a **Bounded Model Checking (BMC)** query in **Conjunctive Normal Form (CNF)**, we can compute makespan-optimal schedules while completely eliminating the deadlocks, livelocks, and myopic collisions that plague decoupled search techniques.

---

## 2. Mathematical Formulation & SAT Reduction

### 2.1 Boolean Variable Definition
Let $R = \{0, 1, \dots, k-1\}$ denote the set of $k$ agents, $V$ the set of obstacle-free grid cells, and $T \in \mathbb{N}$ the bounded makespan horizon.

For each agent $i \in R$, vertex $v \in V$, and discrete time step $t \in \{0, 1, \dots, T\}$, define the Boolean decision variable:
$$x_{i, v, t} \in \{\text{True}, \text{False}\}$$
$$x_{i, v, t} = \text{True} \iff \text{Agent } i \text{ occupies cell } v \text{ at time } t$$

**Variable Cardinality:**
$$|\text{Vars}| = k \cdot |V| \cdot (T + 1)$$
Each variable is bijectively mapped to a 1-based positive integer $\phi(i, v, t) \in \{1, \dots, |\text{Vars}|\}$ for standard DIMACS CNF representation.

---

### 2.2 CNF Constraint Families

#### Constraint 1: Initial and Goal Configurations (Unit Clauses)
At time $t = 0$, agent $i$ must be at its designated start cell $s_i$ and cannot be anywhere else:
$$(x_{i, s_i, 0}) \quad \land \bigwedge_{v \in V \setminus \{s_i\}} (\neg x_{i, v, 0}) \quad \forall i \in R$$

At time $t = T$, agent $i$ must arrive at its designated goal cell $g_i$ and cannot be anywhere else:
$$(x_{i, g_i, T}) \quad \land \bigwedge_{v \in V \setminus \{g_i\}} (\neg x_{i, v, T}) \quad \forall i \in R$$

#### Constraint 2: Uniqueness of Position (ALO & AMO)
At every discrete time step $t \in \{0, \dots, T\}$, each agent $i$ must occupy **exactly one** cell:
- **At-Least-One (ALO):**
  $$\bigvee_{v \in V} x_{i, v, t} \quad \forall i \in R, \forall t \in \{0, \dots, T\}$$
- **At-Most-One (AMO):**
  $$\neg x_{i, u, t} \lor \neg x_{i, v, t} \quad \forall i \in R, \forall t \in \{0, \dots, T\}, \forall u < v \in V$$

#### Constraint 3: Valid Graph Transitions (Motion & Wait Constraints)
At time $t+1$, an agent currently at cell $u$ must either wait at $u$ or move to an adjacent obstacle-free neighbor $v \in N(u)$:
$$x_{i, u, t} \implies \left( x_{i, u, t+1} \lor \bigvee_{v \in N(u)} x_{i, v, t+1} \right)$$
In CNF:
$$\neg x_{i, u, t} \lor x_{i, u, t+1} \lor \bigvee_{v \in N(u)} x_{i, v, t+1} \quad \forall i \in R, \forall u \in V, \forall t \in \{0, \dots, T-1\}$$

#### Constraint 4: Vertex Collision Avoidance
No two distinct agents $i < j$ can occupy the same cell $v \in V$ at the same time step $t$:
$$\neg (x_{i, v, t} \land x_{j, v, t}) \iff \neg x_{i, v, t} \lor \neg x_{j, v, t} \quad \forall 0 \le i < j < k, \forall v \in V, \forall t \in \{0, \dots, T\}$$

#### Constraint 5: Edge Collision Avoidance (Swapping)
No two distinct agents $i < j$ can traverse the same edge in opposite directions between $t$ and $t+1$:
$$\neg (x_{i, u, t} \land x_{i, v, t+1} \land x_{j, v, t} \land x_{j, u, t+1}) \iff \neg x_{i, u, t} \lor \neg x_{i, v, t+1} \lor \neg x_{j, v, t} \lor \neg x_{j, u, t+1}$$
$$\forall 0 \le i < j < k, \forall \{u, v\} \in E, \forall t \in \{0, \dots, T-1\}$$

---

### 2.3 Makespan Optimization via Bounded Model Checking (BMC)

The makespan $T^*$ is the minimal horizon such that the CNF formula is satisfiable.
Because MAPF exhibits **monotonic satisfiability** beyond $T^*$ (once an agent reaches its goal, it can remain there provided no other agent needs to traverse the cell):
1. **Lower Bound Initialization:**
   $$T_{\text{start}} = \max_{i \in R} \left( \text{dist}_{\text{BFS}}(s_i, g_i) \right)$$
2. **Iterative Horizon Escalation:**
   $$\text{For } T = T_{\text{start}}, T_{\text{start}} + 1, T_{\text{start}} + 2, \dots:$$
   - Construct CNF formula $\Phi(T)$.
   - Invoke CDCL SAT Solver (e.g., Glucose-4 via PySAT).
   - If $\Phi(T)$ is `UNSAT`, increment $T \leftarrow T + 1$.
   - If $\Phi(T)$ is `SAT`, decode the truth assignment into paths $\{P_0, \dots, P_{k-1}\}$ and terminate.

---

## 3. Project Architecture

```
mapf_sat_project/
├── README.md                      # Complete theoretical and practical guide
├── requirements.txt               # Dependencies (python-sat, matplotlib, numpy)
├── run_tests.py                   # Automated test suite (6 tests covering correctness)
├── scenarios.py                   # Pre-configured benchmark test scenarios
├── demo.py                        # Out-of-the-box demonstration runner
├── src/
│   ├── __init__.py
│   ├── grid.py                    # 2D Grid map, obstacles, neighbors, BFS distances
│   ├── sat_encoder.py             # Pure CNF SAT reduction and DIMACS exporter
│   ├── sat_solver.py              # BMC optimization loop and independent validator
│   ├── baseline_astar.py          # Decoupled Prioritized Space-Time A* baseline
│   └── visualizer.py              # Matplotlib GIF animator & Interactive HTML5 dashboard
└── output/                        # Generated visual artifacts (.gif, .html)
```

---

## 4. Benchmark Scenarios & Empirical Results

### Scenario 1: Warehouse Bottleneck (4 Robots Crossing $6 \times 6$ Grid)
- **Setup:** 4 storage shelf clusters (8 obstacle cells). Robots 0 & 1 swap opposite ends of the central horizontal aisle, while Robots 2 & 3 cross the vertical aisle.
- **Lower Bound:** BFS distance = 5.
- **BMC Trace:**
  - $T = 5$: `UNSAT` (4.32 ms, 672 vars, 12,928 clauses)
  - $T = 6$: `UNSAT` (5.59 ms, 784 vars, 15,132 clauses)
  - $T = 7$: `SAT` (5.84 ms, 896 vars, 17,336 clauses)
- **Result:** Optimal makespan $T^* = 7$ discovered in 0.046s. Formally verified zero collisions.

### Scenario 2: Narrow Corridor Alcove Swap (Deadlock Benchmark)
- **Setup:** $5 \times 3$ grid with a width-1 corridor along row 1 and a single alcove pocket at $(2, 2)$. Robots 0 & 1 start at $(0, 1)$ and $(4, 1)$ with swapped goals.
- **Decoupled A\* Baseline:** **DEADLOCKS!** Robot 0 claims the direct corridor and parks at $(4, 1)$, blocking Robot 1 permanently.
- **SAT Reduction:** **SUCCEEDS!** Discovers that Robot 1 pulls into alcove $(2, 2)$ at $t=3$, allowing Robot 0 to pass along $(2, 1)$, then steps out at $t=4$ to reach goal at $T^* = 6$.

### Scenario 3: Cyclic Ring Rotation (4 Robots)
- **Setup:** $4 \times 4$ room with a central $2 \times 2$ obstacle pillar. 4 robots on the ring perimeter rotate simultaneously clockwise.
- **SAT Reduction:** Finds simultaneous synchronized rotational plan at $T^* = 6$ in 1.41 ms.

---

## 5. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Automated Unit Tests
```bash
python3 run_tests.py
```

### Run Interactive Out-of-the-Box Demo
```bash
python3 demo.py
```
This runs all scenarios, prints ASCII playback, generates animated GIFs in `output/*.gif`, and produces interactive HTML5 visualizers in `output/*.html`.

---

## 6. Real-World Applications

1. **Automated Fulfillment Warehouses (Kiva / Amazon Robotics):**
   High-density fulfillment centers coordinate hundreds of autonomous drive units moving movable pod shelving. Narrow picking aisles create massive edge-collision and dead-end bottleneck risks. SAT/SMT-based bounded scheduling ensures provably collision-free pick-and-pack routing with minimal queue delays.

2. **Automated Container Terminals (Port Logistics):**
   Heavy Automated Guided Vehicles (AGVs) transport 40-foot shipping containers between ship-to-shore (STS) gantry cranes and automated stacking yards. Due to limited buffer tracks and massive turning radiuses, gridlock at intersections costs millions in port demurrage. SAT-based routing coordinates synchronized right-of-way passes without deadlocks.

3. **Airport Surface & Taxiway Routing:**
   Coordinating ground movement of commercial aircraft between arrival gates, de-icing pads, and active departure runways. Vertex collisions represent catastrophic runway incursions, while edge conflicts halt taxi queues. SAT formulations model discrete taxiway segment occupancy under strict separation minimums.

4. **Digital Microfluidic Biochips (DMFBs):**
   Lab-on-a-chip architectures manipulate discrete nanoliter chemical and biological droplets across 2D planar electrode arrays using electrowetting-on-dielectric (EWOD). Cross-contamination between incompatible biochemical droplets or unwanted premature mixing is modeled directly as space-time collision and proximity constraints in CNF.
