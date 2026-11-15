"""
Decoupled Prioritized Space-Time A* Baseline for MAPF.
Demonstrates cases where greedy sequential pathfinding fails or deadlocks.
"""
import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

from .grid import GridMap, Coordinate

@dataclass
class AStarResult:
    success: bool
    paths: Optional[Dict[int, List[Coordinate]]]
    makespan: Optional[int]
    runtime_sec: float
    failed_agent: Optional[int]
    failure_reason: Optional[str]

class PrioritizedAStar:
    """
    Classic Prioritized Planning MAPF baseline (Silver, 2005).
    Agents plan sequentially in order 0, 1, ..., k-1 using Space-Time A*.
    Higher priority agents reserve space-time vertices (x, y, t) and edges (u, v, t).
    Once an agent reaches its goal at t_goal, it reserves its goal cell for all t in [t_goal, max_t].
    """
    def __init__(self, grid: GridMap, starts: List[Coordinate], goals: List[Coordinate], max_horizon: int = 40):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.num_robots = len(starts)
        self.max_horizon = max_horizon

    def solve(self, agent_order: Optional[List[int]] = None) -> AStarResult:
        t_start = time.perf_counter()
        order = agent_order if agent_order is not None else list(range(self.num_robots))
        
        # Reservation tables
        # reserved_cells[(cell, t)] = agent_id
        # reserved_edges[(u, v, t)] = agent_id (moving from u to v between t and t+1)
        reserved_cells: Dict[Tuple[Coordinate, int], int] = {}
        reserved_edges: Dict[Tuple[Coordinate, Coordinate, int], int] = {}
        
        paths: Dict[int, List[Coordinate]] = {}
        
        for agent_id in order:
            s_i = self.starts[agent_id]
            g_i = self.goals[agent_id]
            
            agent_path = self._space_time_astar(
                agent_id=agent_id,
                start=s_i,
                goal=g_i,
                reserved_cells=reserved_cells,
                reserved_edges=reserved_edges
            )
            
            if agent_path is None:
                runtime = time.perf_counter() - t_start
                return AStarResult(
                    success=False,
                    paths=None,
                    makespan=None,
                    runtime_sec=runtime,
                    failed_agent=agent_id,
                    failure_reason=(
                        f"Robot {agent_id} deadlocked / could not find collision-free path "
                        f"to goal {g_i} within horizon {self.max_horizon} due to higher-priority reservations."
                    )
                )
                
            paths[agent_id] = agent_path
            
            # Commit reservations for this agent
            path_len = len(agent_path)
            for t in range(path_len):
                reserved_cells[(agent_path[t], t)] = agent_id
                if t < path_len - 1:
                    u = agent_path[t]
                    v = agent_path[t + 1]
                    reserved_edges[(u, v, t)] = agent_id
                    
            # Once at goal, agent stays at goal cell until max_horizon
            final_pos = agent_path[-1]
            for t in range(path_len, self.max_horizon + 1):
                reserved_cells[(final_pos, t)] = agent_id

        # Pad all paths to uniform makespan
        max_len = max(len(p) for p in paths.values())
        makespan = max_len - 1
        for agent_id in paths:
            last_pos = paths[agent_id][-1]
            while len(paths[agent_id]) <= makespan:
                paths[agent_id].append(last_pos)

        runtime = time.perf_counter() - t_start
        return AStarResult(
            success=True,
            paths=paths,
            makespan=makespan,
            runtime_sec=runtime,
            failed_agent=None,
            failure_reason=None
        )

    def _space_time_astar(self, agent_id: int, start: Coordinate, goal: Coordinate,
                          reserved_cells: Dict[Tuple[Coordinate, int], int],
                          reserved_edges: Dict[Tuple[Coordinate, Coordinate, int], int]) -> Optional[List[Coordinate]]:
        """
        Space-Time A* for a single agent on a 3D state space (x, y, t).
        Heuristic: BFS shortest path to goal ignoring other agents.
        """
        # (f_score, g_score, (pos, t))
        open_set: List[Tuple[int, int, Tuple[Coordinate, int]]] = []
        g_scores: Dict[Tuple[Coordinate, int], int] = {}
        came_from: Dict[Tuple[Coordinate, int], Tuple[Coordinate, int]] = {}
        
        start_state = (start, 0)
        g_scores[start_state] = 0
        h_start = self.grid.bfs_distance(start, goal)
        if h_start == float('inf'):
            return None
        heapq.heappush(open_set, (int(h_start), 0, start_state))
        
        while open_set:
            f, g, (curr_pos, t) = heapq.heappop(open_set)
            
            if g > g_scores.get((curr_pos, t), float('inf')):
                continue
                
            # Goal condition: agent reaches goal AND is free to remain at goal without future conflict
            if curr_pos == goal:
                # Check if remaining at goal causes conflict with future reservations
                can_stay_at_goal = True
                for future_t in range(t + 1, self.max_horizon + 1):
                    if (goal, future_t) in reserved_cells:
                        can_stay_at_goal = False
                        break
                if can_stay_at_goal:
                    # Reconstruct path
                    path = []
                    curr = (curr_pos, t)
                    while curr in came_from:
                        path.append(curr[0])
                        curr = came_from[curr]
                    path.append(start)
                    path.reverse()
                    return path

            if t >= self.max_horizon:
                continue

            # Candidate moves: stay at curr_pos OR move to 4-neighbor
            candidates = [curr_pos] + self.grid.get_neighbors(curr_pos)
            next_t = t + 1
            
            for next_pos in candidates:
                # 1. Vertex collision check: cell must not be reserved at next_t
                if (next_pos, next_t) in reserved_cells:
                    continue
                    
                # 2. Edge collision check: no agent can move next_pos -> curr_pos at t
                if (next_pos, curr_pos, t) in reserved_edges:
                    continue
                    
                tentative_g = g + 1
                next_state = (next_pos, next_t)
                
                if tentative_g < g_scores.get(next_state, float('inf')):
                    g_scores[next_state] = tentative_g
                    came_from[next_state] = (curr_pos, t)
                    h = self.grid.bfs_distance(next_pos, goal)
                    if h != float('inf'):
                        heapq.heappush(open_set, (int(tentative_g + h), tentative_g, next_state))
                        
        return None
