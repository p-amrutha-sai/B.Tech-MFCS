"""
Boolean Satisfiability (SAT) CNF Encoder for Multi-Agent Path Finding (MAPF).
Encodes discrete 2D grid MAPF as a Bounded Model Checking (BMC) query in Conjunctive Normal Form.
"""
from typing import List, Tuple, Dict, Set, Optional
from .grid import GridMap, Coordinate

class MAPFSATEncoder:
    """
    Translates a MAPF instance on a GridMap with bounded makespan T into DIMACS CNF clauses.
    
    Boolean Variables:
      x(i, v, t): True if robot i is at cell v at time step t.
      i in {0, ..., num_robots-1}
      v in free_cells
      t in {0, ..., T}
    """
    def __init__(self, grid: GridMap, starts: List[Coordinate], goals: List[Coordinate], T: int):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.num_robots = len(starts)
        self.T = T
        
        if len(starts) != len(goals):
            raise ValueError("Number of start and goal positions must match")
        for i, (s, g) in enumerate(zip(starts, goals)):
            if not grid.is_free(s[0], s[1]):
                raise ValueError(f"Robot {i} start position {s} is not in free space")
            if not grid.is_free(g[0], g[1]):
                raise ValueError(f"Robot {i} goal position {g} is not in free space")
                
        # Bidirectional variable mapping: (robot_id, cell, t) <-> DIMACS int >= 1
        self.var_to_id: Dict[Tuple[int, Coordinate, int], int] = {}
        self.id_to_var: Dict[int, Tuple[int, Coordinate, int]] = {}
        self._build_variable_mappings()
        
        # Clause containers categorized by constraint type
        self.clauses_init_goal: List[List[int]] = []
        self.clauses_alo: List[List[int]] = []
        self.clauses_amo: List[List[int]] = []
        self.clauses_transitions: List[List[int]] = []
        self.clauses_vertex_collisions: List[List[int]] = []
        self.clauses_edge_collisions: List[List[int]] = []
        
        self._encode_all_constraints()

    def _build_variable_mappings(self):
        """Constructs 1-based variable IDs for (robot_id, cell, t)."""
        var_id = 1
        for i in range(self.num_robots):
            for cell in self.grid.free_cells:
                for t in range(self.T + 1):
                    key = (i, cell, t)
                    self.var_to_id[key] = var_id
                    self.id_to_var[var_id] = key
                    var_id += 1
        self.num_vars = var_id - 1

    def var(self, i: int, cell: Coordinate, t: int) -> int:
        """Returns the positive integer literal for x_{i, cell, t}."""
        return self.var_to_id[(i, cell, t)]

    def _encode_all_constraints(self):
        """Generates all 5 constraint families in Conjunctive Normal Form (CNF)."""
        self._encode_initial_and_goal()
        self._encode_uniqueness_alo_amo()
        self._encode_valid_transitions()
        self._encode_vertex_collision_avoidance()
        self._encode_edge_collision_avoidance()

    def _encode_initial_and_goal(self):
        """
        Constraint 1: Initial positions at t=0 and Goal positions at t=T (Unit Clauses).
        - Robot i is at s_i at t=0: (x_{i, s_i, 0})
        - Robot i is NOT at v != s_i at t=0: (~x_{i, v, 0})
        - Robot i is at g_i at t=T: (x_{i, g_i, T})
        - Robot i is NOT at v != g_i at t=T: (~x_{i, v, T})
        """
        for i in range(self.num_robots):
            s_i = self.starts[i]
            g_i = self.goals[i]
            
            # Start position at t=0
            self.clauses_init_goal.append([self.var(i, s_i, 0)])
            for v in self.grid.free_cells:
                if v != s_i:
                    self.clauses_init_goal.append([-self.var(i, v, 0)])
                    
            # Goal position at t=T
            self.clauses_init_goal.append([self.var(i, g_i, self.T)])
            for v in self.grid.free_cells:
                if v != g_i:
                    self.clauses_init_goal.append([-self.var(i, v, self.T)])

    def _encode_uniqueness_alo_amo(self):
        r"""
        Constraint 2: Uniqueness of position.
        Each robot occupies at least one and at most one cell per time step:
        - At-Least-One (ALO): \bigvee_{v in V} x_{i, v, t}
        - At-Most-One (AMO): ~x_{i, u, t} \lor ~x_{i, v, t} for all u < v in V
        """
        free_cells = self.grid.free_cells
        num_cells = len(free_cells)
        
        for i in range(self.num_robots):
            for t in range(self.T + 1):
                # ALO: At least one cell occupied
                alo_clause = [self.var(i, v, t) for v in free_cells]
                self.clauses_alo.append(alo_clause)
                
                # AMO: At most one cell occupied (pairwise encoding)
                for u_idx in range(num_cells):
                    u = free_cells[u_idx]
                    for v_idx in range(u_idx + 1, num_cells):
                        v = free_cells[v_idx]
                        self.clauses_amo.append([-self.var(i, u, t), -self.var(i, v, t)])

    def _encode_valid_transitions(self):
        r"""
        Constraint 3: Valid Transitions (Continuity / Motion Constraints).
        At t+1, each robot must stay in its current cell or move to an adjacent obstacle-free cell:
          x_{i, u, t} => (x_{i, u, t+1} \lor \bigvee_{v in N(u)} x_{i, v, t+1})
        In CNF:
          ~x_{i, u, t} \lor x_{i, u, t+1} \lor \bigvee_{v in N(u)} x_{i, v, t+1}
        """
        for i in range(self.num_robots):
            for t in range(self.T):
                for u in self.grid.free_cells:
                    neighbors = self.grid.get_neighbors(u)
                    clause = [-self.var(i, u, t), self.var(i, u, t + 1)]
                    for v in neighbors:
                        clause.append(self.var(i, v, t + 1))
                    self.clauses_transitions.append(clause)

    def _encode_vertex_collision_avoidance(self):
        r"""
        Constraint 4: Vertex Collision Avoidance.
        No two robots occupy the same cell at the same time t:
          ~(x_{i, v, t} \land x_{j, v, t}) <=> ~x_{i, v, t} \lor ~x_{j, v, t}
        for all 0 <= i < j < num_robots, v in V, t in {0, ..., T}.
        """
        for t in range(self.T + 1):
            for v in self.grid.free_cells:
                for i in range(self.num_robots):
                    for j in range(i + 1, self.num_robots):
                        self.clauses_vertex_collisions.append([
                            -self.var(i, v, t),
                            -self.var(j, v, t)
                        ])

    def _encode_edge_collision_avoidance(self):
        r"""
        Constraint 5: Edge Collision Avoidance (Swapping).
        No two robots swap positions across an edge between t and t+1:
          ~(x_{i, u, t} \land x_{i, v, t+1} \land x_{j, v, t} \land x_{j, u, t+1})
        In CNF:
          ~x_{i, u, t} \lor ~x_{i, v, t+1} \lor ~x_{j, v, t} \lor ~x_{j, u, t+1}
        for all distinct robot pairs i < j, adjacent cells u, v, and t in {0, ..., T-1}.
        """
        # Find all undirected edges {u, v}
        undirected_edges = set()
        for u in self.grid.free_cells:
            for v in self.grid.get_neighbors(u):
                if u < v:
                    undirected_edges.add((u, v))
                    
        for t in range(self.T):
            for (u, v) in undirected_edges:
                for i in range(self.num_robots):
                    for j in range(i + 1, self.num_robots):
                        # Case 1: Robot i moves u->v while Robot j moves v->u
                        self.clauses_edge_collisions.append([
                            -self.var(i, u, t),
                            -self.var(i, v, t + 1),
                            -self.var(j, v, t),
                            -self.var(j, u, t + 1)
                        ])
                        # Case 2: Robot i moves v->u while Robot j moves u->v
                        self.clauses_edge_collisions.append([
                            -self.var(i, v, t),
                            -self.var(i, u, t + 1),
                            -self.var(j, u, t),
                            -self.var(j, v, t + 1)
                        ])

    def get_all_clauses(self) -> List[List[int]]:
        """Returns the combined list of all CNF clauses."""
        return (
            self.clauses_init_goal +
            self.clauses_alo +
            self.clauses_amo +
            self.clauses_transitions +
            self.clauses_vertex_collisions +
            self.clauses_edge_collisions
        )

    def get_clause_stats(self) -> Dict[str, int]:
        """Returns a breakdown of clause counts per constraint family."""
        return {
            "init_goal": len(self.clauses_init_goal),
            "alo": len(self.clauses_alo),
            "amo": len(self.clauses_amo),
            "transitions": len(self.clauses_transitions),
            "vertex_collisions": len(self.clauses_vertex_collisions),
            "edge_collisions": len(self.clauses_edge_collisions),
            "total_clauses": len(self.get_all_clauses()),
            "total_variables": self.num_vars
        }

    def to_dimacs(self) -> str:
        """Exports the CNF instance to standard DIMACS CNF format string."""
        clauses = self.get_all_clauses()
        lines = [
            f"c ============================================================",
            f"c MAPF to SAT Reduction - Bounded Model Checking (T = {self.T})",
            f"c Grid: {self.grid.width}x{self.grid.height}, Free cells: {len(self.grid.free_cells)}",
            f"c Robots: {self.num_robots}, Makespan: {self.T}",
            f"c Clauses: Total={len(clauses)} (Init/Goal={len(self.clauses_init_goal)}, "
            f"ALO={len(self.clauses_alo)}, AMO={len(self.clauses_amo)}, "
            f"Trans={len(self.clauses_transitions)}, "
            f"V-Col={len(self.clauses_vertex_collisions)}, E-Col={len(self.clauses_edge_collisions)})",
            f"c ============================================================",
            f"p cnf {self.num_vars} {len(clauses)}"
        ]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)

    def decode_solution(self, model: List[int]) -> Dict[int, List[Coordinate]]:
        """
        Decodes a satisfying assignment (truth model from SAT solver)
        into robot paths: paths[robot_id] = [cell_at_t0, cell_at_t1, ..., cell_at_tT].
        """
        paths: Dict[int, List[Optional[Coordinate]]] = {
            i: [None] * (self.T + 1) for i in range(self.num_robots)
        }
        
        # In PySAT, model is a list of signed integer literals where positive means True
        positive_literals = {lit for lit in model if lit > 0}
        
        for lit in positive_literals:
            if lit in self.id_to_var:
                robot_id, cell, t = self.id_to_var[lit]
                paths[robot_id][t] = cell
                
        # Validate that every robot has a position at each time step
        for i in range(self.num_robots):
            for t in range(self.T + 1):
                if paths[i][t] is None:
                    raise ValueError(f"Decoded solution incomplete: Robot {i} has no position at t={t}")
                    
        return paths  # type: ignore
