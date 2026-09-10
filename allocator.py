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
    target_speed:  float = 1.0

def _dist_score(d: float, scale: float) -> float:   #normalize the distances received from dijkstra
    return math.exp(-d/scale) if d != math.inf and d >= 0 else 0.0

def _get_cutoff_candidates(ghost, pr: float, pc: float) -> list[tuple[float, float]]:
    candidates = []
    p_dir = getattr(ghost, '_player_dir', (0, 0))
    p_speed = math.hypot(p_dir[0], p_dir[1])
    if p_speed > 0.01:
        dy, dx = p_dir[0] / p_speed, p_dir[1] / p_speed
        offsets = [(dy * 3.0, dx * 3.0), (dy * 5.0, dx * 5.0), (-dx * 3.0, dy * 3.0), (dx * 3.0, -dy * 3.0)]
    else:
        offsets = [(-3.0, 0.0), (3.0, 0.0), (0.0, -3.0), (0.0, 3.0)]
    for dr, dc in offsets:
        cr, cc = float(pr + dr), float(pc + dc)
        if ghost.world and ghost.world.is_passable(cc, cr, radius=0.4):
            candidates.append((cr, cc))
    return candidates

def _score_hunt(ghost, dists: dict, frame: int) -> list[Task]:
    if ghost.pacman_powered:
        return []
    target = ghost.known_pacman or ghost.last_lost_pacman
    if target is None:
        return []
    pr, pc = target
    dist_key = (round(float(pr), 2), round(float(pc), 2))
    info = dists.get(dist_key) or dists.get(target)
    if info is None:
        return []
    dist, _ = info
    if dist == math.inf:
        return []    
    tasks = []
    score = _dist_score(dist, HUNT_SCALE)
    tasks.append(Task(task_type=TaskType.HUNT, target_pos=(pr, pc), score=score, created_frame=frame, owner=ghost.gid, target_speed=1.0))
    for cr, cc in _get_cutoff_candidates(ghost, pr, pc):
        cutoff_key = (round(cr, 2), round(cc, 2))
        cutoff_info = dists.get(cutoff_key) or dists.get((cr, cc))
        if cutoff_info and cutoff_info[0] != math.inf:
            cutoff_score = _dist_score(cutoff_info[0], HUNT_SCALE) * 0.85
            tasks.append(Task(task_type=TaskType.HUNT, target_pos=(cr, cc), score=cutoff_score, created_frame=frame, owner=ghost.gid, target_speed=1.0))
    return tasks

def _score_convert(ghost, dists: dict, frame: int) -> List[Task]:
    tasks: list[Task] = []
    if not hasattr(ghost, 'known_power_pellets'): return tasks
    for pos in ghost.known_power_pellets:
        yx_pos = (pos[1], pos[0])
        dist_key = (round(yx_pos[0], 2), round(yx_pos[1], 2))
        info = dists.get(dist_key) or dists.get(yx_pos)
        if info is None:
            continue
        dist, _ = info
        if dist == math.inf:
            continue
        score = _dist_score(dist, CONVERT_SCALE) * 0.60
        tasks.append(Task(task_type=TaskType.CONVERT, target_pos=yx_pos, score=score, created_frame=frame, target_speed=1.0))
    return tasks

def _find_flee_pos(ghost, pacman_pos: tuple) -> Optional[tuple]:
    pr, pc = pacman_pos
    if getattr(ghost, 'world', None) is None: return None
    prm_nodes = getattr(ghost.world, 'prm_nodes', None)
    if prm_nodes:
        best_node = None
        best_safety = -math.inf
        for n in prm_nodes:
            pac_d = math.hypot(n[0] - pr, n[1] - pc)
            ghost_d = math.hypot(n[0] - ghost.y, n[1] - ghost.x)
            safety = pac_d - 0.5 * ghost_d
            if pac_d >= SAFE_RADIUS:
                safety += 10.0
            if safety > best_safety:
                best_safety = safety
                best_node = (float(n[0]), float(n[1]))
        if best_node is not None:
            return best_node
    corners = [(1.5, 1.5), (1.5, float(ghost.world.width - 2)), (float(ghost.world.height - 2), 1.5), (float(ghost.world.height - 2), float(ghost.world.width - 2))]
    passable_corners = [c for c in corners if ghost.world.is_passable(c[1], c[0], radius=0.4)] or corners
    best_corner = None
    best_dist = -1
    for cr, cc in passable_corners:
        d = math.hypot(cr - pr, cc - pc)
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
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=flee_pos, score=2.0, created_frame=frame, owner=ghost.gid, target_speed=1.0)
    else:
        score = 0.5 
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=target, score=score, created_frame=frame, owner=ghost.gid, target_speed=0.5)

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
        tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score, created_frame=frame, target_speed=0.8))
    return tasks

def generate_tasks(ghost, frame: int) -> tuple[List[Task], dict]:
    start = (float(ghost.y), float(ghost.x))
    targets: set = set()
    pac_pos = ghost.known_pacman or ghost.last_lost_pacman
    if pac_pos is not None:
        pr, pc = float(pac_pos[0]), float(pac_pos[1])
        targets.add((pr, pc))
        for cr, cc in _get_cutoff_candidates(ghost, pr, pc):
            targets.add((cr, cc))
    if getattr(ghost, 'pacman_powered', False) and ghost.known_pacman:
        flee_p = _find_flee_pos(ghost, ghost.known_pacman)
        if flee_p is not None:
            targets.add(flee_p)
    for p in getattr(ghost, 'known_power_pellets', []):
        targets.add((p[1], p[0]))
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
        tasks.extend(_score_hunt(ghost, dists, frame))
        tasks.extend(_score_convert(ghost, dists, frame))
        tasks.extend(explore_tasks)
    for t in tasks:
        if t.created_frame == 0:
            t.created_frame = frame
    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks, dists

def best_task(tasks: List[Task]) -> Optional[Task]:
    return max(tasks, key=lambda t: t.score) if tasks else None