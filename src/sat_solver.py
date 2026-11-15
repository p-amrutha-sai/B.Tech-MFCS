"""
Solver wrapper for Bounded Model Checking (BMC) Makespan Optimization using PySAT.
"""
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from pysat.solvers import Solver

from .grid import GridMap, Coordinate
from .sat_encoder import MAPFSATEncoder

@dataclass
class SolveStepResult:
    T: int
    is_sat: bool
    solve_time_sec: float
    num_vars: int
    num_clauses: int
    clause_stats: Dict[str, int]
    paths: Optional[Dict[int, List[Coordinate]]] = None

@dataclass
class OptimalMakespanResult:
    optimal_T: Optional[int]
    total_time_sec: float
    history: List[SolveStepResult]
    paths: Optional[Dict[int, List[Coordinate]]]
    is_valid: bool
    validation_message: str

def validate_solution(grid: GridMap, starts: List[Coordinate], goals: List[Coordinate], 
                      paths: Dict[int, List[Coordinate]], T: int) -> Tuple[bool, str]:
    """
    Independent formal verifier for MAPF solution validity.
    Verifies:
      1. Start configuration match
      2. Goal configuration match
      3. Valid graph transitions (wait or move to 4-neighbor)
      4. No vertex collisions at any time step
      5. No edge collisions (swapping) across any transition
    """
    num_robots = len(starts)
    
    # 1. Start positions
    for i in range(num_robots):
        if paths[i][0] != starts[i]:
            return False, f"Robot {i} start mismatch: expected {starts[i]}, got {paths[i][0]}"
            
    # 2. Goal positions
    for i in range(num_robots):
        if paths[i][T] != goals[i]:
            return False, f"Robot {i} goal mismatch: expected {goals[i]}, got {paths[i][T]}"
            
    # 3. Transitions
    for i in range(num_robots):
        for t in range(T):
            curr = paths[i][t]
            nxt = paths[i][t + 1]
            if not grid.is_free(curr[0], curr[1]):
                return False, f"Robot {i} in obstacle or out of bounds at t={t}: {curr}"
            if not grid.is_free(nxt[0], nxt[1]):
                return False, f"Robot {i} in obstacle or out of bounds at t={t+1}: {nxt}"
            if curr != nxt and nxt not in grid.get_neighbors(curr):
                return False, f"Robot {i} illegal jump at t={t}->{t+1}: {curr} -> {nxt}"

    # 4. Vertex collisions
    for t in range(T + 1):
        occupied = {}
        for i in range(num_robots):
            pos = paths[i][t]
            if pos in occupied:
                other = occupied[pos]
                return False, f"Vertex collision at t={t} at cell {pos} between Robot {other} and Robot {i}"
            occupied[pos] = i

    # 5. Edge collisions (swapping)
    for t in range(T):
        for i in range(num_robots):
            u_i = paths[i][t]
            v_i = paths[i][t + 1]
            for j in range(i + 1, num_robots):
                u_j = paths[j][t]
                v_j = paths[j][t + 1]
                if u_i == v_j and v_i == u_j and u_i != v_i:
                    return False, f"Edge swap collision between t={t} and t={t+1} across {u_i} <-> {v_i} between Robot {i} and Robot {j}"

    return True, "Solution rigorously verified: all constraints satisfied with zero collisions."

class MAPFSATSolver:
    """
    Orchestrates BMC makespan optimization for MAPF.
    Iterates T from lower bound upwards until a SAT model is found.
    """
    def __init__(self, grid: GridMap, starts: List[Coordinate], goals: List[Coordinate], solver_name: str = "g4"):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.solver_name = solver_name

    def solve_fixed_makespan(self, T: int) -> SolveStepResult:
        """Solves the decision problem for a single fixed horizon T."""
        encoder = MAPFSATEncoder(self.grid, self.starts, self.goals, T)
        clauses = encoder.get_all_clauses()
        stats = encoder.get_clause_stats()
        
        t_start = time.perf_counter()
        with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
            is_sat = solver.solve()
            solve_time = time.perf_counter() - t_start
            
            paths = None
            if is_sat:
                model = solver.get_model()
                paths = encoder.decode_solution(model)

        return SolveStepResult(
            T=T,
            is_sat=is_sat,
            solve_time_sec=solve_time,
            num_vars=encoder.num_vars,
            num_clauses=len(clauses),
            clause_stats=stats,
            paths=paths
        )

    def optimize_makespan(self, max_T: int = 50, verbose: bool = True) -> OptimalMakespanResult:
        """
        Iterative BMC loop:
          T = max_i(distance(s_i, g_i))
          while True:
            solve(T)
            if SAT: return T
            T = T + 1
        """
        t_total_start = time.perf_counter()
        
        # Theoretical lower bound from individual shortest paths
        lb_manhattan = max(self.grid.manhattan_distance(s, g) for s, g in zip(self.starts, self.goals))
        lb_bfs = max(self.grid.bfs_distance(s, g) for s, g in zip(self.starts, self.goals))
        T_start = int(max(lb_manhattan, lb_bfs))
        
        if verbose:
            print(f"\n{'='*72}")
            print(f" SAT-BASED MAPF BOUNDED MODEL CHECKING (BMC) MAKESPAN OPTIMIZATION")
            print(f"{'='*72}")
            print(f" Grid dimensions  : {self.grid.width} x {self.grid.height} ({len(self.grid.free_cells)} free cells)")
            print(f" Number of robots : {len(self.starts)}")
            print(f" Distance bounds  : Manhattan LB = {lb_manhattan}, Graph-BFS LB = {T_start}")
            print(f" SAT Engine       : PySAT ({self.solver_name.upper()})")
            print(f"{'-'*72}")
            print(f" {'T':>3} | {'Outcome':^8} | {'Variables':>10} | {'Clauses':>10} | {'Solve Time':>12} | {'Notes'}")
            print(f"{'-'*72}")

        history: List[SolveStepResult] = []
        for T in range(T_start, max_T + 1):
            step_res = self.solve_fixed_makespan(T)
            history.append(step_res)
            
            outcome_str = "SAT [OK]" if step_res.is_sat else "UNSAT"
            time_str = f"{step_res.solve_time_sec * 1000:.2f} ms"
            notes = "Optimal makespan found!" if step_res.is_sat else "Incrementing T <- T+1"
            
            if verbose:
                print(f" {T:>3} | {outcome_str:^8} | {step_res.num_vars:>10,} | {step_res.num_clauses:>10,} | {time_str:>12} | {notes}")

            if step_res.is_sat:
                total_time = time.perf_counter() - t_total_start
                # Validate the extracted paths
                is_valid, val_msg = validate_solution(self.grid, self.starts, self.goals, step_res.paths, T)
                if verbose:
                    print(f"{'-'*72}")
                    print(f" SUCCESS: Optimal Makespan T* = {T} in {total_time:.3f}s total ({len(history)} BMC iterations)")
                    print(f" Verification: {val_msg}")
                    print(f"{'='*72}\n")
                return OptimalMakespanResult(
                    optimal_T=T,
                    total_time_sec=total_time,
                    history=history,
                    paths=step_res.paths,
                    is_valid=is_valid,
                    validation_message=val_msg
                )

        total_time = time.perf_counter() - t_total_start
        if verbose:
            print(f"{'-'*72}")
            print(f" EXHAUSTED: Reached max horizon T={max_T} without SAT solution in {total_time:.3f}s")
            print(f"{'='*72}\n")
        return OptimalMakespanResult(
            optimal_T=None,
            total_time_sec=total_time,
            history=history,
            paths=None,
            is_valid=False,
            validation_message=f"No solution found up to max makespan {max_T}"
        )
