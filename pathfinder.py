"""
This module implements A* and Dijkstra as seperate means to compute optimal paths using each ghost's personal map.
A* is used for single-target pathfinding for the RL model, wheras Dijkstra's multi-target nature is used by CBBA to score multi-candidate tasks.
"""

import heapq
import math

WALL    =  1
EMPTY   =  0
PELLET  =  2
POWER   =  3
UNKNOWN = -1

#cell costs for path planning - wall taken to be impassable
_COST = {EMPTY: 1.0, PELLET: 1.0, POWER: 0.5, UNKNOWN: 3.0, WALL: math.inf}   #unknown territory taken to be passable but 3x more costly than known cells
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _cost(grid: list, r: int, c: int) -> float:
    val = grid[r][c]
    return _COST.get(val, math.inf)

def _in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

def _manhattan(a: tuple, b: tuple) -> float:    #we use manhattan as the A* heuristic since ghosts travel in cardinal directions, so its better than euclidean
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

def _reconstruct(came_from: dict, node: tuple) -> list:
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def astar(grid: list, start: tuple, goal: tuple) -> list:
    if start == goal:
        return [start]
    rows = len(grid)
    cols = len(grid[0])
    if not (_in_bounds(start[0], start[1], rows, cols) and _in_bounds(goal[0], goal[1], rows, cols)) or _cost(grid, start[0], start[1]) == math.inf or _cost(grid, goal[0], goal[1]) == math.inf:
        return []
    g_score: dict = {start: 0.0}
    came_from: dict = {}
    open_heap = [(0.0 + _manhattan(start, goal), 0.0, start)]
    closed: set = set()
    while open_heap:
        f, g, node = heapq.heappop(open_heap)
        if node in closed:   #best path to node alr known, skip
            continue
        closed.add(node)
        if node == goal:
            return _reconstruct(came_from, node)  #found goal, reconstruct and return path
        r, c = node
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, rows, cols):
                continue
            neighbour = (nr, nc)
            if neighbour in closed:
                continue
            step_cost = _cost(grid, nr, nc)
            if step_cost == math.inf:
                continue
            tentative_g = g + step_cost
            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour]  = tentative_g
                came_from[neighbour] = node
                f_new = tentative_g + _manhattan(neighbour, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, neighbour))
    return []   #if goal is unreachable from start

def dijkstra_multi(grid: list, start: tuple, targets: list) -> dict:
    if not targets:
        return {}
    rows = len(grid)
    cols = len(grid[0])
    
    target_set = set()
    for t in targets:
        if _in_bounds(t[0], t[1], rows, cols):
            target_set.add(t)
    results: dict = {t: (math.inf, []) for t in target_set}
    remaining = set(target_set)

    if start in target_set:
        results[start] = (0.0, [start])
        remaining.discard(start)

    if not remaining:
        return results

    dist: dict = {start: 0.0}
    came_from: dict = {}
    open_heap = [(0.0, start)]
    closed: set = set()

    while open_heap and remaining:
        d, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)

        if node in remaining:       #found a target, reconstruct path and store result
            path = _reconstruct(came_from, node)
            results[node] = (d, path)
            remaining.discard(node)
            if not remaining:
                break

        r, c = node
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, rows, cols):
                continue
            neighbour = (nr, nc)
            if neighbour in closed:
                continue
            step_cost = _cost(grid, nr, nc)
            if step_cost == math.inf:
                continue
            tentative = d + step_cost
            if tentative < dist.get(neighbour, math.inf):
                dist[neighbour]      = tentative
                came_from[neighbour] = node
                heapq.heappush(open_heap, (tentative, neighbour))

    return results      #return inf filled results - if all goals are unreachable from start

def next_step(grid: list, start: tuple, goal: tuple) -> tuple | None:       #return immediate next step towards goal
    path = astar(grid, start, goal)
    if len(path) >= 2:
        return path[1]
    return None