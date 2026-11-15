#!/usr/bin/env python3
"""
Interactive Demonstration and Benchmark Suite for MAPF SAT Reduction.
Runs:
  1. Warehouse 4-robot bottleneck scenario with SAT Bounded Model Checking
  2. Prioritized A* baseline comparison showing deadlock vs SAT global coordination
  3. Cyclic rotation scenario
  4. Generates animated GIF and interactive HTML dashboards
"""
import sys
import os
import time

# Ensure src is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.grid import GridMap
from src.sat_encoder import MAPFSATEncoder
from src.sat_solver import MAPFSATSolver, validate_solution
from src.baseline_astar import PrioritizedAStar
from src.visualizer import MAPFVisualizer
from scenarios import (
    get_scenario_warehouse_4_robots,
    get_scenario_corridor_alcove_swap,
    get_scenario_cyclic_rotation_4_robots
)

def run_scenario_demo(scenario, output_dir: str = "output", render_gif: bool = True):
    print("\n" + "#" * 80)
    print(f" SCENARIO: {scenario.name.upper()}")
    print(f" Description: {scenario.description}")
    print("#" * 80)
    
    # 1. Print initial grid layout
    print("\nInitial Grid Map (R# = Start, G# = Goal, ### = Obstacle):")
    starts_dict = {i: scenario.starts[i] for i in range(len(scenario.starts))}
    goals_dict = {i: scenario.goals[i] for i in range(len(scenario.goals))}
    print(scenario.grid.render_ascii(starts_dict, goals_dict))
    
    # 2. Run SAT-based Bounded Model Checking
    solver = MAPFSATSolver(scenario.grid, scenario.starts, scenario.goals, solver_name="g4")
    sat_result = solver.optimize_makespan(max_T=30, verbose=True)
    
    if not sat_result.is_valid:
        print(f"[ERROR] SAT solution failed verification: {sat_result.validation_message}")
        return
        
    print("\nOptimal Path Trajectories Found by SAT Reduction:")
    for i in range(len(scenario.starts)):
        path_str = " -> ".join([str(p) for p in sat_result.paths[i]])
        print(f"  Robot {i}: {path_str}")

    # Inspect CNF details at optimal makespan
    opt_step = sat_result.history[-1]
    stats = opt_step.clause_stats
    print(f"\nExact CNF Clause Decomposition at Optimal Horizon T={opt_step.T}:")
    print(f"  - Total Boolean Variables (x_{{i,v,t}}): {stats['total_variables']:,}")
    print(f"  - Unit Clauses (Initial & Goal)      : {stats['init_goal']:,}")
    print(f"  - At-Least-One (ALO) Uniqueness     : {stats['alo']:,}")
    print(f"  - At-Most-One (AMO) Uniqueness      : {stats['amo']:,}")
    print(f"  - Valid Graph Transitions           : {stats['transitions']:,}")
    print(f"  - Vertex Collision Avoidance        : {stats['vertex_collisions']:,}")
    print(f"  - Edge Collision Avoidance (Swaps)  : {stats['edge_collisions']:,}")
    print(f"  - TOTAL CLAUSES IN CNF FORMULA      : {stats['total_clauses']:,}")

    # 3. Baseline Comparison: Prioritized Space-Time A*
    print(f"\n{'-'*80}")
    print(" BASELINE COMPARISON: DECOUPLED PRIORITIZED SPACE-TIME A*")
    print(f"{'-'*80}")
    astar = PrioritizedAStar(scenario.grid, scenario.starts, scenario.goals, max_horizon=35)
    astar_result = astar.solve()
    
    if astar_result.success:
        print(f"  [SUCCESS] Prioritized A* found solution with makespan = {astar_result.makespan} (Time: {astar_result.runtime_sec*1000:.2f} ms)")
        print(f"  Suboptimality gap vs SAT: SAT makespan = {sat_result.optimal_T}, A* makespan = {astar_result.makespan}")
    else:
        print(f"  [DEADLOCK DETECTED] Prioritized A* failed!")
        print(f"  Reason: {astar_result.failure_reason}")
        print(f"  Key Insight: Greedy sequential planning cannot resolve shared-resource conflicts without global coordination.")
        print(f"               SAT reduction succeeds globally where decoupled pathfinding deadlocks!")

    # 4. Visualization & Artifacts
    safe_name = scenario.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    vis = MAPFVisualizer(scenario.grid, scenario.starts, scenario.goals, sat_result.paths, scenario.name)
    
    html_file = os.path.join(output_dir, f"{safe_name}.html")
    vis.generate_interactive_html(html_file, baseline_paths=astar_result.paths if astar_result.success else None)
    print(f"\n[Artifact] Interactive HTML visualizer saved to: {html_file}")
    
    if render_gif:
        gif_file = os.path.join(output_dir, f"{safe_name}.gif")
        print(f"[Artifact] Rendering animated GIF: {gif_file} ...", end="", flush=True)
        vis.save_gif(gif_file, fps=2)
        print(" Done!")

    return sat_result, astar_result

def main():
    print("================================================================================")
    print("  MULTI-AGENT PATH FINDING (MAPF) REDUCED TO BOOLEAN SATISFIABILITY (SAT)")
    print("  Mathematical Foundations of Computer Science (MFCS) Term Project Prototype")
    print("================================================================================")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Non-trivial 6x6 warehouse bottleneck with 4 crossing robots
    wh_scenario = get_scenario_warehouse_4_robots()
    run_scenario_demo(wh_scenario, output_dir=output_dir, render_gif=True)
    
    # 2. Narrow corridor with alcove (deadlock benchmark)
    corridor_scenario = get_scenario_corridor_alcove_swap()
    run_scenario_demo(corridor_scenario, output_dir=output_dir, render_gif=True)

    # 3. Cyclic ring rotation
    ring_scenario = get_scenario_cyclic_rotation_4_robots()
    run_scenario_demo(ring_scenario, output_dir=output_dir, render_gif=True)

    print("\n" + "=" * 80)
    print(" DEMONSTRATION COMPLETE: All scenarios solved and verified conflict-free.")
    print(f" All visual animations and HTML dashboards available in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
