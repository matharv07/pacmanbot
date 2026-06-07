"""
Templates for the task allocator for CBBA implementation

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

from pathfinder import dijkstra_multi

WALL    = 1
EMPTY   = 0
PELLET  = 2
POWER   = 3
UNKNOWN = -1

#Setup decay constants for CBBA
HUNT_SCALE    = 10.0
CONVERT_SCALE = 8.0
SAFE_RADIUS   = 8       #min safe power pacman distance
SAFE_SCALE    = 8.0
RECENCY_SCALE = 20.0    #sets up quantity to prioritize revisiting older mapped locations
EXPLORE_SCALE = 6.0
UNKNOWN_BONUS = 100     #5x reward(?) of looking for new locations over updating old ones
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

def _dist_score(d: float, scale: float) -> float:   #normalize the distances received from dijkstra
    return math.exp(-d / scale) if d != math.inf and d >= 0 else 0.0

def _score_hunt(ghost, dists: dict) -> Optional[Task]:
    if ghost.pacman_powered:
        return None
    target = ghost.known_pacman or ghost.last_lost_pacman  #fallback to last known location
    if target is None:
        return None
    info = dists.get(target)
    if info is None:
        return None
    dist, _ = info
    if dist == math.inf:
        return None
    score = _dist_score(dist, HUNT_SCALE)
    return Task(task_type=TaskType.HUNT, target_pos=target, score=score)


def _score_convert(ghost, dists: dict) -> List[Task]:
    tasks: list[Task] = []
    rows = len(ghost.personal_map)
    cols = len(ghost.personal_map[0])
    for r in range(rows):
        for c in range(cols):
            if ghost.personal_map[r][c] == POWER:
                pos = (r, c)
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
    best_pos  = None
    best_dist = -1
    rows = len(ghost.personal_map)
    cols = len(ghost.personal_map[0])
    for r in range(rows):
        for c in range(cols):
            if ghost.personal_map[r][c] in (WALL, UNKNOWN):
                continue
            d = abs(r - pr) + abs(c - pc)
            if d > best_dist:
                best_dist = d
                best_pos  = (r, c)
    return best_pos


def _score_evade_track(ghost, dists: dict, frame: int) -> Optional[Task]:
    if not ghost.pacman_powered:
        return None
    target = ghost.known_pacman
    if target is None:
        return None
    info = dists.get(target)
    if info is None:
        return None
    dist, _ = info
    if dist == math.inf:
        return None
    if dist < SAFE_RADIUS:
        #inside danger zone - fleeing to farthest known cell
        flee_pos = _find_flee_pos(ghost, target)
        if flee_pos is None:
            return None
        return Task(task_type=TaskType.EVADE_TRACK,target_pos=flee_pos, score=2.0, created_frame=frame)
    #outside danger zone - score by how far we are (farther => higher score)
    score = 1.0 - _dist_score(dist, SAFE_SCALE)
    return Task(task_type=TaskType.EVADE_TRACK, target_pos=_find_flee_pos(ghost, target) or target, score=score, created_frame=frame)

def _score_explore(ghost, frame: int) -> List[Task]:    #pick top-K locations with unknown or older info
    rows = len(ghost.personal_map)
    cols = len(ghost.personal_map[0])
    scored: list = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if ghost.personal_map[r][c] == WALL:
                continue
            if ghost.personal_map[r][c] == UNKNOWN:
                age = frame + UNKNOWN_BONUS
            else:
                last = ghost.last_seen[r][c]
                age = frame - last if last >= 0 else frame + 1
            scored.append((age, (r, c)))
    if not scored:
        return []
    top = heapq.nlargest(EXPLORE_TOP_K, scored, key=lambda x: x[0])
    tasks: list = []
    for age, pos in top:
        score = 1.0 - math.exp(-age / RECENCY_SCALE)
        score *= _dist_score(abs(pos[0] - ghost.row) + abs(pos[1] - ghost.col), EXPLORE_SCALE)
        tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score))
    return tasks

def generate_tasks(ghost, frame: int) -> List[Task]:
    start = (ghost.row, ghost.col)
    rows  = len(ghost.personal_map)
    cols  = len(ghost.personal_map[0])
    targets: set = set()
    pac_pos = ghost.known_pacman or ghost.last_lost_pacman
    if pac_pos is not None:
        targets.add(pac_pos)
    for r in range(rows):
        for c in range(cols):
            if ghost.personal_map[r][c] == POWER:
                targets.add((r, c))
    corners = [(1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]
    for cn in corners:
        targets.add(cn)

    explore_tasks = _score_explore(ghost, frame)
    for et in explore_tasks:
        targets.add(et.target_pos)
    dists = dijkstra_multi(ghost.grid, start, list(targets))
    tasks: list[Task] = []
    hunt = _score_hunt(ghost, dists)
    if hunt is not None:
        tasks.append(hunt)
    tasks.extend(_score_convert(ghost, dists))
    evade_track = _score_evade_track(ghost, dists, frame)
    if evade_track is not None:
        tasks.append(evade_track)
    tasks.extend(explore_tasks)

    for t in tasks:
        if t.created_frame == 0:
            t.created_frame = frame

    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks


def best_task(tasks: List[Task]) -> Optional[Task]:
    return max(tasks, key=lambda t: t.score) if tasks else None