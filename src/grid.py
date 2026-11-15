"""
Grid representation and graph utilities for 2D Multi-Agent Path Finding.
"""
from collections import deque
from typing import List, Tuple, Set, Dict, Optional

Coordinate = Tuple[int, int]

class GridMap:
    """
    Represents a discrete 2D grid with static obstacles.
    Coordinates are 0-indexed: x in [0, width-1], y in [0, height-1].
    (0,0) is top-left or bottom-left depending on visualization (bottom-left standard for math/Cartesian).
    """
    def __init__(self, width: int, height: int, obstacles: Optional[Set[Coordinate]] = None):
        self.width = width
        self.height = height
        self.obstacles: Set[Coordinate] = set(obstacles) if obstacles else set()
        
        # Validate obstacles
        for ox, oy in self.obstacles:
            if not (0 <= ox < width and 0 <= oy < height):
                raise ValueError(f"Obstacle ({ox}, {oy}) is out of grid bounds ({width}x{height})")
                
        # Cache free cells
        self.free_cells: List[Coordinate] = sorted([
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.obstacles
        ])
        self.free_cells_set: Set[Coordinate] = set(self.free_cells)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.obstacles

    def get_neighbors(self, u: Coordinate) -> List[Coordinate]:
        """Returns 4-connected orthogonal free neighbors."""
        x, y = u
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_free(c[0], c[1])]

    def manhattan_distance(self, p1: Coordinate, p2: Coordinate) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def bfs_distance(self, start: Coordinate, goal: Coordinate) -> int:
        """Computes shortest obstacle-avoiding path length between start and goal."""
        if start == goal:
            return 0
        if not self.is_free(start[0], start[1]) or not self.is_free(goal[0], goal[1]):
            return float('inf')
            
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            curr, dist = queue.popleft()
            if curr == goal:
                return dist
            for nxt in self.get_neighbors(curr):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return float('inf')

    def render_ascii(self, agent_positions: Optional[Dict[int, Coordinate]] = None, 
                     goals: Optional[Dict[int, Coordinate]] = None) -> str:
        """
        Renders ASCII grid.
        '.' = free cell, '#' = obstacle
        Numbers (0-9) or letters = agent positions
        'G0', 'G1' = goals if not occupied
        """
        agent_positions = agent_positions or {}
        goals = goals or {}
        pos_to_agent = {pos: agent_id for agent_id, pos in agent_positions.items()}
        goal_to_agent = {pos: agent_id for agent_id, pos in goals.items()}
        
        lines = []
        border = "+" + "---+" * self.width
        lines.append(border)
        for y in range(self.height - 1, -1, -1):  # print Cartesian y descending
            row_str = "|"
            for x in range(self.width):
                pos = (x, y)
                if pos in pos_to_agent:
                    aid = pos_to_agent[pos]
                    cell_repr = f" R{aid} " if aid < 10 else f"R{aid}"
                elif pos in goal_to_agent:
                    gid = goal_to_agent[pos]
                    cell_repr = f" G{gid} " if gid < 10 else f"G{gid}"
                elif pos in self.obstacles:
                    cell_repr = "###"
                else:
                    cell_repr = " . "
                row_str += cell_repr + "|"
            lines.append(row_str)
            lines.append(border)
        return "\n".join(lines)
