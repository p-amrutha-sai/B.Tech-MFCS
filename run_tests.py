"""
Comprehensive Unit and Integration Test Suite for MAPF SAT Reduction Prototype.
"""
import unittest
from src.grid import GridMap
from src.sat_encoder import MAPFSATEncoder
from src.sat_solver import MAPFSATSolver, validate_solution
from src.baseline_astar import PrioritizedAStar
from scenarios import get_scenario_corridor_alcove_swap, get_scenario_warehouse_4_robots

class TestMAPFSATReduction(unittest.TestCase):
    def setUp(self):
        # 3x3 grid with no obstacles
        self.grid = GridMap(3, 3)
        self.starts = [(0, 0), (2, 2)]
        self.goals = [(2, 2), (0, 0)]
        self.T = 4

    def test_variable_bijection(self):
        """Test bijection between Boolean tuples (i, v, t) and integer literals."""
        encoder = MAPFSATEncoder(self.grid, self.starts, self.goals, self.T)
        num_robots = 2
        num_cells = 9
        time_steps = 5  # 0, 1, 2, 3, 4
        expected_vars = num_robots * num_cells * time_steps
        
        self.assertEqual(encoder.num_vars, expected_vars)
        self.assertEqual(len(encoder.var_to_id), expected_vars)
        self.assertEqual(len(encoder.id_to_var), expected_vars)
        
        # Verify 1-based indexing
        self.assertEqual(min(encoder.id_to_var.keys()), 1)
        self.assertEqual(max(encoder.id_to_var.keys()), expected_vars)
        
        # Verify reversibility
        for key, var_id in encoder.var_to_id.items():
            self.assertEqual(encoder.id_to_var[var_id], key)

    def test_uniqueness_clauses(self):
        """Test ALO (At-Least-One) and AMO (At-Most-One) counts and structure."""
        encoder = MAPFSATEncoder(self.grid, self.starts, self.goals, self.T)
        num_robots = 2
        num_cells = 9
        time_steps = self.T + 1
        
        # ALO: 1 clause per robot per time step
        self.assertEqual(len(encoder.clauses_alo), num_robots * time_steps)
        for clause in encoder.clauses_alo:
            self.assertEqual(len(clause), num_cells)
            
        # AMO: n*(n-1)/2 clauses per robot per time step
        expected_amo_per_step = (num_cells * (num_cells - 1)) // 2
        self.assertEqual(len(encoder.clauses_amo), num_robots * time_steps * expected_amo_per_step)
        for clause in encoder.clauses_amo:
            self.assertEqual(len(clause), 2)
            self.assertTrue(clause[0] < 0 and clause[1] < 0)

    def test_vertex_collision_clauses(self):
        """Test vertex collision avoidance pairwise clauses."""
        encoder = MAPFSATEncoder(self.grid, self.starts, self.goals, self.T)
        num_robots = 2
        num_cells = 9
        time_steps = self.T + 1
        
        # Pairs of robots: C(2, 2) = 1
        expected_v_col = 1 * num_cells * time_steps
        self.assertEqual(len(encoder.clauses_vertex_collisions), expected_v_col)
        for clause in encoder.clauses_vertex_collisions:
            self.assertEqual(len(clause), 2)
            self.assertTrue(clause[0] < 0 and clause[1] < 0)

    def test_independent_validator_detects_flaws(self):
        """Test formal validator catches vertex collisions, edge swaps, and illegal moves."""
        # 3x3 grid: (0,0) to (0,2) for robot 0, (1,1) to (0,0) for robot 1
        starts = [(0, 0), (1, 1)]
        goals = [(0, 2), (0, 0)]
        T = 2
        
        # Valid graph transitions, but both land on (0, 1) at t=1!
        bad_paths_vcol = {
            0: [(0, 0), (0, 1), (0, 2)],
            1: [(1, 1), (0, 1), (0, 0)],
        }
        valid, msg = validate_solution(self.grid, starts, goals, bad_paths_vcol, T)
        self.assertFalse(valid)
        self.assertIn("Vertex collision", msg)

        # Deliberate edge swap across adjacent cells (0,0) and (1,0) at t=0 -> 1
        starts_swap = [(0, 0), (1, 0)]
        goals_swap = [(1, 0), (0, 0)]
        bad_paths_eswap = {
            0: [(0, 0), (1, 0), (1, 0)],
            1: [(1, 0), (0, 0), (0, 0)],  # Swapping across (0,0) <-> (1,0)
        }
        valid, msg = validate_solution(self.grid, starts_swap, goals_swap, bad_paths_eswap, T)
        self.assertFalse(valid)
        self.assertIn("Edge swap collision", msg)

        # Deliberate illegal teleport
        bad_paths_teleport = {
            0: [(0, 0), (2, 2), (0, 2)],  # Illegal jump from (0,0) to (2,2)
            1: [(1, 1), (0, 1), (0, 0)],
        }
        valid, msg = validate_solution(self.grid, starts, goals, bad_paths_teleport, T)
        self.assertFalse(valid)
        self.assertIn("illegal jump", msg)

    def test_bmc_makespan_optimality(self):
        """
        Verify makespan optimality:
        2 robots swapping in a 3x1 corridor with an alcove at (1, 1).
        Starts: R0 at (0, 0), R1 at (2, 0).
        Goals:  R0 at (2, 0), R1 at (0, 0).
        Shortest distance without collision is 2.
        With collision in 1-wide corridor, R0 must move to alcove (1, 1) and wait.
        Minimum makespan to swap is T=4.
        Therefore, T=2 and T=3 MUST be UNSAT, and T=4 MUST be SAT!
        """
        grid = GridMap(3, 2, obstacles={(0, 1), (2, 1)})  # Alcove only at (1, 1)
        starts = [(0, 0), (2, 0)]
        goals = [(2, 0), (0, 0)]
        
        solver = MAPFSATSolver(grid, starts, goals)
        
        # Test T=2: must be UNSAT
        res_t2 = solver.solve_fixed_makespan(2)
        self.assertFalse(res_t2.is_sat, "T=2 should be UNSAT due to corridor conflict")

        # Test T=3: must be UNSAT
        res_t3 = solver.solve_fixed_makespan(3)
        self.assertFalse(res_t3.is_sat, "T=3 should be UNSAT due to alcove maneuver timing")

        # Test T=4: must be SAT
        res_t4 = solver.solve_fixed_makespan(4)
        self.assertTrue(res_t4.is_sat, "T=4 must be SAT")
        
        # Validate extracted solution
        valid, msg = validate_solution(grid, starts, goals, res_t4.paths, 4)
        self.assertTrue(valid, msg)

    def test_corridor_deadlock_scenario_sat_vs_astar(self):
        """Verify SAT succeeds on alcove scenario where Prioritized A* deadlocks."""
        scenario = get_scenario_corridor_alcove_swap()
        
        # 1. SAT solver
        sat_solver = MAPFSATSolver(scenario.grid, scenario.starts, scenario.goals)
        sat_res = sat_solver.optimize_makespan(max_T=15, verbose=False)
        self.assertTrue(sat_res.is_valid)
        self.assertIsNotNone(sat_res.optimal_T)

        # 2. Prioritized A* baseline
        astar = PrioritizedAStar(scenario.grid, scenario.starts, scenario.goals, max_horizon=20)
        astar_res = astar.solve()
        self.assertFalse(astar_res.success, "Prioritized A* should deadlock in 1-wide corridor")
        self.assertIsNotNone(astar_res.failure_reason)

if __name__ == "__main__":
    unittest.main()
