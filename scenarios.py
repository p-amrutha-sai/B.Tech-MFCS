"""
Pre-configured non-trivial test scenarios for MAPF evaluation.
Includes warehouse bottlenecks, narrow corridor swapping with alcoves, and cyclic rotations.
"""
from dataclasses import dataclass
from typing import List, Set, Tuple
from src.grid import GridMap, Coordinate

@dataclass
class Scenario:
    name: str
    description: str
    grid: GridMap
    starts: List[Coordinate]
    goals: List[Coordinate]

def get_scenario_warehouse_4_robots() -> Scenario:
    """
    6x6 Warehouse Floor with 4 Storage Racks and a Central Crossway.
    4 robots navigate across orthogonal and diagonal trajectories, intersecting
    at the central high-traffic bottleneck.
    """
    width, height = 6, 6
    # Obstacles: 4 storage shelf clusters (each 2x1)
    obstacles = {
        (1, 1), (1, 2),  # West-South Shelf
        (1, 4), (1, 5),  # West-North Shelf
        (4, 1), (4, 2),  # East-South Shelf
        (4, 4), (4, 5),  # East-North Shelf
    }
    grid = GridMap(width, height, obstacles)
    
    # 4 Robots crossing paths
    starts = [
        (0, 3),  # Robot 0: West aisle -> East aisle
        (5, 3),  # Robot 1: East aisle -> West aisle (opposite!)
        (2, 0),  # Robot 2: South center -> North center
        (3, 5),  # Robot 3: North center -> South center
    ]
    goals = [
        (5, 3),  # Robot 0 Goal
        (0, 3),  # Robot 1 Goal
        (2, 5),  # Robot 2 Goal
        (3, 0),  # Robot 3 Goal
    ]
    
    return Scenario(
        name="Warehouse Bottleneck (4 Robots)",
        description=(
            "6x6 warehouse grid with 4 storage racks. Robots 0 & 1 swap opposite ends of the "
            "central horizontal corridor while Robots 2 & 3 cross the vertical corridor simultaneously."
        ),
        grid=grid,
        starts=starts,
        goals=goals
    )

def get_scenario_corridor_alcove_swap() -> Scenario:
    """
    5x3 Narrow Corridor with an Alcove Pocket.
    Demonstrates Prioritized A* Deadlock vs SAT Global Coordination.
    Robots 0 & 1 must swap positions across a width-1 hallway. A single pocket at (2, 2)
    allows one robot to pull aside while the other passes.
    """
    width, height = 5, 3
    # Solid walls on top and bottom rows, except alcove pocket at (2, 2)
    obstacles = {
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # Bottom wall
        (0, 2), (1, 2),         (3, 2), (4, 2),  # Top wall with pocket at (2, 2)
    }
    grid = GridMap(width, height, obstacles)
    
    # Robots starting at opposite ends of the hallway
    starts = [
        (0, 1),  # Robot 0
        (4, 1),  # Robot 1
    ]
    goals = [
        (4, 1),  # Robot 0 Goal
        (0, 1),  # Robot 1 Goal
    ]
    
    return Scenario(
        name="Narrow Corridor Alcove Swap (Deadlock Benchmark)",
        description=(
            "5x3 grid with a width-1 corridor and an alcove at (2, 2). "
            "Prioritized A* fails/deadlocks because Robot 0 claims the corridor and parks on the goal, "
            "whereas SAT discovers the coordinated sidestep maneuver into the alcove."
        ),
        grid=grid,
        starts=starts,
        goals=goals
    )

def get_scenario_cyclic_rotation_4_robots() -> Scenario:
    """
    4x4 Room with a Central 2x2 Pillar.
    4 robots in a closed ring wanting to shift clockwise by 2 positions.
    Requires synchronized collective motion.
    """
    width, height = 4, 4
    obstacles = {
        (1, 1), (1, 2),
        (2, 1), (2, 2)
    }
    grid = GridMap(width, height, obstacles)
    
    starts = [
        (0, 1),  # Robot 0 (West)
        (1, 3),  # Robot 1 (North)
        (3, 2),  # Robot 2 (East)
        (2, 0),  # Robot 3 (South)
    ]
    goals = [
        (3, 2),  # Robot 0 -> East
        (2, 0),  # Robot 1 -> South
        (0, 1),  # Robot 2 -> West
        (1, 3),  # Robot 3 -> North
    ]
    
    return Scenario(
        name="Cyclic Ring Rotation (4 Robots)",
        description=(
            "4x4 ring around a central pillar. 4 robots rotate around the perimeter, "
            "requiring simultaneous coordinated movement."
        ),
        grid=grid,
        starts=starts,
        goals=goals
    )

def get_all_scenarios() -> List[Scenario]:
    return [
        get_scenario_warehouse_4_robots(),
        get_scenario_corridor_alcove_swap(),
        get_scenario_cyclic_rotation_4_robots(),
    ]
