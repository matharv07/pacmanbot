"""
Templates for the task allocator for CBBA implementation + RL Task Generation

Task types are as follows:
HUNT - chase tracked or predicted Pacman position
CONVERT - eat a POWER pellet to convert it into a normal pellet
EVADE_TRACK - run from powered pacman while keeping track of it
EVADE_FLEE - flee from powered pacman when too close
EXPLORE - map out unexplored regions
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
import numpy as np
from pathfinder import dijkstra_multi

WALL    = 1
EMPTY   = 0
PELLET  = 2
POWER   = 3
UNKNOWN = -1

#Setup decay constants for CBBA
HUNT_SCALE    = 14.0
CONVERT_SCALE = 8.0
SAFE_RADIUS   = 8       #min safe power pacman distance
SAFE_SCALE    = 8.0
RECENCY_SCALE = 20.0    #sets up quantity to prioritize revisiting older mapped locations
EXPLORE_SCALE = 6.0
UNKNOWN_BONUS = 40      #5x reward(?) of looking for new locations over updating old ones
EXPLORE_TOP_K = 3       #number of top explore candidates passed to CBBA

class TaskType(IntEnum):
    HUNT        = 0
    CONVERT     = 1
    EVADE_TRACK = 2
    EXPLORE     = 3
    DYNAMIC     = 4     #rl generated waypoints that dont fit the above

@dataclass
class Task:
    task_type:     TaskType
    target_pos:    tuple          
    score:         float
    assigned_to:   int = -1
    created_frame: int = 0
    owner:         int = -1

def _dist_score(d: float, scale: float) -> float:   #normalize the distances received from dijkstra
    return math.exp(-d/scale) if d != math.inf and d >= 0 else 0.0

def _score_hunt(ghost, dists: dict) -> list[Task]:
    if ghost.pacman_powered:
        return []
    target = ghost.known_pacman or ghost.last_lost_pacman
    if target is None:
        return []
    pr, pc = target
    info = dists.get(target)
    if info is None:
        return []
    dist, _ = info
    if dist == math.inf:
        return []    
    tasks = []
    score = _dist_score(dist, HUNT_SCALE)
    tasks.append(Task(task_type=TaskType.HUNT, target_pos=target,
                      score=score, owner=ghost.gid))
    rows, cols = ghost.personal_map.shape
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        cr, cc = pr + dr*4, pc + dc*4
        if 0 <= cr < rows and 0 <= cc < cols and ghost.personal_map[cr, cc] != WALL:
            cutoff_info = dists.get((cr, cc))
            if cutoff_info and cutoff_info[0] != math.inf:
                cutoff_score = _dist_score(cutoff_info[0], HUNT_SCALE) * 0.85
                tasks.append(Task(task_type=TaskType.DYNAMIC, target_pos=(cr, cc), score=cutoff_score, owner=ghost.gid))
    return tasks

def _score_convert(ghost, dists: dict) -> List[Task]:
    tasks: list[Task] = []
    power_cells = np.argwhere(ghost.personal_map == POWER)
    for pos_arr in power_cells:
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        info = dists.get(pos)
        if info is None:
            continue
        dist, _ = info
        if dist == math.inf:
            continue
        score = _dist_score(dist, CONVERT_SCALE)
        tasks.append(Task(task_type=TaskType.CONVERT, target_pos=pos, score=score))
    return tasks

def _find_flee_pos(ghost, pacman_pos: tuple) -> Optional[tuple]:
    pr, pc = pacman_pos
    p_map = np.array(ghost.personal_map)
    valid = (p_map != WALL) & (p_map != UNKNOWN)
    if not valid.any():
        return None
    rows, cols = p_map.shape
    r_idx, c_idx = np.indices((rows, cols))
    dists = np.abs(r_idx - pr) + np.abs(c_idx - pc)
    dists[~valid] = -1
    max_idx = np.argmax(dists)
    best_r = max_idx // cols
    best_c = max_idx % cols
    return (int(best_r), int(best_c))

def _score_evade_track(ghost, dists: dict, frame: int) -> Optional[Task]:
    if not ghost.pacman_powered:
        return None
    target = ghost.known_pacman
    if target is None:
        return None
    info = dists.get(target)
    if info is None or info[0] == math.inf:
        return None
    dist, _ = info
    if dist < SAFE_RADIUS:
        flee_pos = _find_flee_pos(ghost, target)
        if flee_pos is None:
            return None
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=flee_pos, score=2.0, created_frame=frame, owner=ghost.gid)
    else:
        score = 0.5 
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=target, score=score, created_frame=frame, owner=ghost.gid)

def _score_explore(ghost, frame: int) -> List[Task]:
    p = ghost.personal_map
    rows, cols = p.shape
    interior = p
    ls = ghost.last_seen
    wall_mask = (interior == WALL)
    unknown_mask = (interior == UNKNOWN)
    ages = np.zeros_like(interior, dtype=np.float64)
    ages[unknown_mask] = frame + UNKNOWN_BONUS
    known_mask = (~wall_mask) & (~unknown_mask)
    ages[known_mask] = np.where(ls[known_mask] >= 0, frame - ls[known_mask], frame+1).astype(np.float64)
    ages[wall_mask] = -1
    flat_ages = ages.ravel()
    n_valid = np.sum(flat_ages >= 0)
    if n_valid == 0:
        return []
    k = min(EXPLORE_TOP_K, int(n_valid))
    top_flat = np.argpartition(flat_ages, -k)[-k:]
    top_flat = top_flat[np.argsort(flat_ages[top_flat])[::-1]]
    tasks: list = []
    interior_cols = cols
    for idx in top_flat:
        age = flat_ages[idx]
        if age < 0:
            continue
        r = int(idx // interior_cols)
        c = int(idx % interior_cols)
        pos = (r, c)
        score = 1.0 - math.exp(-age / RECENCY_SCALE)
        score *= _dist_score(abs(pos[0] - ghost.row) + abs(pos[1] - ghost.col), EXPLORE_SCALE)
        if getattr(ghost, 'cbba_agent', None):
            for key in ghost.cbba_agent.bundle:
                if key[1] == pos:
                    score += 0.5
        tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score))
    return tasks

def generate_tasks(ghost, frame: int) -> tuple[List[Task], dict]:
    start = (ghost.row, ghost.col)
    rows, cols = ghost.personal_map.shape
    targets: set = set()
    pac_pos = ghost.known_pacman or ghost.last_lost_pacman
    if pac_pos is not None:
        targets.add(pac_pos)
        pr, pc = pac_pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            cr, cc = pr + dr*4, pc + dc*4
            if 0 <= cr < rows and 0 <= cc < cols and ghost.personal_map[cr, cc] != WALL:
                targets.add((cr, cc))
    power_cells = np.argwhere(ghost.personal_map == POWER)
    for r, c in power_cells:
        targets.add((int(r), int(c)))
    corners = [(1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]
    for cn in corners:
        targets.add(cn)
    explore_tasks = _score_explore(ghost, frame)
    for et in explore_tasks:
        targets.add(et.target_pos)
    dists = dijkstra_multi(ghost.grid, start, list(targets))
    tasks: list[Task] = []
    if getattr(ghost, 'pacman_powered', False):
        evade_track = _score_evade_track(ghost, dists, frame)
        if evade_track is not None:
            if evade_track.target_pos not in dists:
                extra_dist = dijkstra_multi(ghost.grid, start, [evade_track.target_pos])
                dists.update(extra_dist)
            tasks.append(evade_track)
        tasks.extend(explore_tasks)
    else:
        tasks.extend(_score_hunt(ghost, dists))
        tasks.extend(_score_convert(ghost, dists))
        tasks.extend(explore_tasks)
    for t in tasks:
        if t.created_frame == 0:
            t.created_frame = frame
    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks, dists

def best_task(tasks: List[Task]) -> Optional[Task]:
    return max(tasks, key=lambda t: t.score) if tasks else None