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
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        cr, cc = float(pr + dr*4), float(pc + dc*4)
        if ghost.world and ghost.world.is_passable(cc, cr, radius=0.4):
            cutoff_info = dists.get((cr, cc))
            if cutoff_info and cutoff_info[0] != math.inf:
                cutoff_score = _dist_score(cutoff_info[0], HUNT_SCALE) * 0.85
                tasks.append(Task(task_type=TaskType.DYNAMIC, target_pos=(cr, cc), score=cutoff_score, owner=ghost.gid))
    return tasks

def _score_convert(ghost, dists: dict) -> List[Task]:
    tasks: list[Task] = []
    if not hasattr(ghost, 'known_power_pellets'): return tasks
    for pos in ghost.known_power_pellets:
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
    if getattr(ghost, 'world', None) is None: return None
    corners = [(1.5, 1.5), (1.5, float(ghost.world.width - 2)), (float(ghost.world.height - 2), 1.5), (float(ghost.world.height - 2), float(ghost.world.width - 2))]
    best_corner = None
    best_dist = -1
    for cr, cc in corners:
        d = abs(cr - pr) + abs(cc - pc)
        if d > best_dist:
            best_dist = d
            best_corner = (cr, cc)
    return best_corner

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
    ls = ghost.prm_last_seen
    ages = {}
    for node, last_seen_frame in ls.items():
        if last_seen_frame < 0:
            ages[node] = frame + UNKNOWN_BONUS
        else:
            ages[node] = frame - last_seen_frame
            
    sorted_nodes = sorted(ages.items(), key=lambda item: item[1], reverse=True)
    
    tasks: list = []
    for pos, age in sorted_nodes[:EXPLORE_TOP_K]:
        score = 1.0 - math.exp(-age / RECENCY_SCALE)
        score *= _dist_score(abs(pos[0] - ghost.y) + abs(pos[1] - ghost.x), EXPLORE_SCALE)
        if getattr(ghost, 'cbba_agent', None):
            for key in ghost.cbba_agent.bundle:
                if key[1] == pos:
                    score += 0.5
        tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score))
    return tasks

def generate_tasks(ghost, frame: int) -> tuple[List[Task], dict]:
    start = (float(ghost.y), float(ghost.x))
    targets: set = set()
    pac_pos = ghost.known_pacman or ghost.last_lost_pacman
    if pac_pos is not None:
        pr, pc = float(pac_pos[0]), float(pac_pos[1])
        targets.add((pr, pc))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            cr, cc = pr + dr*4, pc + dc*4
            if ghost.world and ghost.world.is_passable(cc, cr, radius=0.4):
                targets.add((cr, cc))
    for p in ghost.world.power_pellets:
        targets.add(p)
    corners = [(1.5, 1.5), (1.5, float(ghost.world.width - 2)), (float(ghost.world.height - 2), 1.5), (float(ghost.world.height - 2), float(ghost.world.width - 2))]
    for cn in corners:
        targets.add(cn)
    explore_tasks = _score_explore(ghost, frame)
    for et in explore_tasks:
        targets.add((float(et.target_pos[0]), float(et.target_pos[1])))
    dists = dijkstra_multi(ghost.world, start, list(targets))
    tasks: list[Task] = []
    if getattr(ghost, 'pacman_powered', False):
        evade_track = _score_evade_track(ghost, dists, frame)
        if evade_track is not None:
            if evade_track.target_pos not in dists:
                extra_dist = dijkstra_multi(ghost.world, start, [evade_track.target_pos])
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