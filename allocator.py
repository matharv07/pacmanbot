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

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from pathfinder import dijkstra_multi

WALL = 1
EMPTY = 0
PELLET = 2
POWER = 3
UNKNOWN = -1

#Setup decay constants for CBBA
HUNT_SCALE    = 10.0
CONVERT_SCALE = 8.0
RISK_SCALE    = 6.0     #ghost capture risk from pacman
SAFE_RADIUS   = 8       #min safe power pacman distance
SAFE_SCALE    = 8.0
RECENCY_SCALE = 20.0    #sets up quantity to prioritize revisiting older mapped locations
EXPLORE_SCALE = 6.0
UNKNOWN_BONUS = 100     #5x reward(?) of looking for new locations over updating old ones

class TaskType(IntEnum):
    HUNT        = 0
    CONVERT     = 1
    EVADE_TRACK = 2
    EVADE_FLEE  = 3
    EXPLORE     = 4

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
    target = ghost.known_pacman
    info = dists.get(target)
    dist, path = info
    if target is None:
        target = ghost.last_lost_pacman  #fallback to last known location - we set this up in the verifier method
    if target is None or ghost.pacman_powered or info is None or dist == math.inf:
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
                dist, path = info
                if info is None or dist == math.inf:
                    continue
                score = _dist_score(dist, CONVERT_SCALE)
                tasks.append(Task(task_type=TaskType.CONVERT, target_pos=pos, score=score))
    return tasks

def _score_evade_track(ghost, dists: dict, frame: int) -> Optional[Task]:
    target = ghost.known_pacman
    info = dists.get(target)
    dist, path = info
    if not ghost.pacman_powered or target is None or info is None or dist == math.inf or dist < SAFE_RADIUS:
        return None
    score = _dist_score(dist, SAFE_SCALE)
    return Task(task_type=TaskType.EVADE_TRACK, target_pos=target, score=score, created_frame=frame)

def _score_evade_flee(ghost, dists: dict) -> Optional[Task]:
    target = ghost.known_pacman
    info = dists.get(target)
    dist_to_pac, _ = info
    if not ghost.pacman_powered or target is None or info is None or dist_to_pac == math.inf or dist_to_pac >= SAFE_RADIUS:
        return None
    urgency_to_move = _dist_score(dist_to_pac, RISK_SCALE)
    #heuristic: we pick the map corner farthest from Pacman as flee destination - until next auction update
    rows = len(ghost.personal_map)
    cols = len(ghost.personal_map[0])
    corners = [(1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]
    pr, pc = target
    best_corner = max(corners, key=lambda c: abs(c[0] - pr) + abs(c[1] - pc))
    return Task(task_type=TaskType.EVADE_FLEE, target_pos=best_corner, score=urgency_to_move)

def _score_explore(ghost, frame: int) -> Optional[Task]:    #pick locations with unknown or older info
    rows = len(ghost.personal_map)
    cols = len(ghost.personal_map[0])
    best_pos = None
    best_age = -1
    for r in range(rows):
        for c in range(cols):
            if ghost.personal_map[r][c] == WALL:
                continue
            if ghost.personal_map[r][c] == UNKNOWN:
                age = frame + UNKNOWN_BONUS
            else:
                last = ghost.last_seen[r][c]
                age = frame - last if last >= 0 else frame + 1
            if age > best_age:
                best_age = age
                best_pos = (r, c)
    if best_pos is None:
        return None
    score = 1.0 - math.exp(-best_age / RECENCY_SCALE)
    #normalizing scoring using manhattan distance to make it similar terms to the other tasks
    score *= _dist_score(abs(best_pos[0] - ghost.row) + abs(best_pos[1] - ghost.col), EXPLORE_SCALE)
    return Task(task_type=TaskType.EXPLORE, target_pos=best_pos, score=score)

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

    explore_task = _score_explore(ghost, frame)
    if explore_task is not None:
        targets.add(explore_task.target_pos)
    dists = dijkstra_multi(ghost.personal_map, start, list(targets))
    tasks: list[Task] = []
    hunt = _score_hunt(ghost, dists)
    if hunt is not None:
        tasks.append(hunt)
    tasks.extend(_score_convert(ghost, dists))
    evade_track = _score_evade_track(ghost, dists, frame)
    if evade_track is not None:
        tasks.append(evade_track)
    evade_flee = _score_evade_flee(ghost, dists)
    if evade_flee is not None:
        tasks.append(evade_flee)
    if explore_task is not None:
        tasks.append(explore_task)

    for t in tasks:
        if t.created_frame == 0:
            t.created_frame = frame

    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks


def best_task(tasks: List[Task]) -> Optional[Task]:
    return max(tasks, key=lambda t: t.score) if tasks else None