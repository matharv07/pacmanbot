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
import os as _os
from pathfinder import ghost_dists

WALL    = 1
EMPTY   = 0
PELLET  = 2
POWER   = 3
UNKNOWN = -1

HUNT_SCALE    = 14.0
CONVERT_SCALE = 8.0
CONVERT_DENIAL_W = 7.0     #weight on racing Pacman to a power pellet it is closing on
SAFE_RADIUS   = 14         #min safe distance from a powered Pacman
SAFE_SCALE    = 8.0
RECENCY_SCALE = 20.0       #sets up quantity to prioritize revisiting older mapped locations
EXPLORE_SCALE = 6.0
UNKNOWN_BONUS = 40         #5x reward(?) of looking for new locations over updating old ones
EXPLORE_TOP_K = 3          #number of top explore candidates passed to CBBA
PELLET_THREAT_DIST = float(_os.environ.get("PELLET_THREAT_DIST", "4.5"))
PELLET_THREAT_HOLD = float(_os.environ.get("PELLET_THREAT_HOLD", "8.5"))
PELLET_THREAT_DAMP = float(_os.environ.get("PELLET_THREAT_DAMP", "0.3"))
CONVERT_RACE_RATIO = float(_os.environ.get("CONVERT_RACE_RATIO", "0.5"))
CONVERT_RACE_LOSS  = float(_os.environ.get("CONVERT_RACE_LOSS", "0.0"))

class TaskType(IntEnum):
    HUNT        = 0
    CONVERT     = 1
    EVADE_TRACK = 2
    EXPLORE     = 3
    DYNAMIC     = 4        #rl generated waypoints that dont fit the above
    FLANK       = 5        #multi-directional cutoff/corridor intercept

ORIGIN_HEURISTIC  = 0      #proposed by the allocator
ORIGIN_RL_ENDORSE = 1      #actor nominated a cell within 1.5 of an allocator proposal and amplified its score
ORIGIN_RL_NOVEL   = 2      #actor nominated a cell no allocator rule proposed

@dataclass
class Task:
    task_type:     TaskType
    target_pos:    tuple          
    score:         float
    assigned_to:   int = -1
    created_frame: int = 0
    owner:         int = -1
    target_speed:  float = 1.0
    origin:        int = ORIGIN_HEURISTIC

def _dist_score(d: float, scale: float) -> float:   #normalize the distances received from dijkstra
    return math.exp(-d/scale) if d != math.inf and d >= 0 else 0.0

def _get_cutoff_candidates(ghost, pr: float, pc: float) -> list[tuple[float, float]]:
    candidates = []
    p_dir = getattr(ghost, '_player_dir', (0, 0))
    p_speed = math.hypot(p_dir[0], p_dir[1])
    min_dim = min(ghost.world.height, ghost.world.width) if (getattr(ghost, 'world', None) and hasattr(ghost.world, 'height')) else 10.0
    #scale lead distances down on small grids (e.g. 7x9) to prevent projecting through outer walls
    d1 = min(3.0, max(1.5, min_dim * 0.25))
    d2 = min(5.0, max(2.5, min_dim * 0.40))
    if p_speed > 0.01:
        dy, dx = p_dir[0] / p_speed, p_dir[1] / p_speed
        offsets = [(dy * d1, dx * d1), (dy * d2, dx * d2), (-dx * d1, dy * d1), (dx * d1, -dy * d1)]
    else:
        offsets = [(-d1, 0.0), (d1, 0.0), (0.0, -d1), (0.0, d1)]
    for dr, dc in offsets:
        cr, cc = float(pr + dr), float(pc + dc)
        if ghost.world and ghost.world.is_passable(cc, cr, radius=0.4):
            candidates.append((cr, cc))
    #fallback to close cardinal offsets if directional projections are blocked by walls
    if len(candidates) < 2:
        for dr, dc in [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5)]:
            cr, cc = float(pr + dr), float(pc + dc)
            if ghost.world and ghost.world.is_passable(cc, cr, radius=0.4):
                cand = (cr, cc)
                if cand not in candidates:
                    candidates.append(cand)
    return candidates

def _lookup_dist(dists: dict, target: tuple) -> Optional[tuple]:
    if target in dists:
        return dists[target]
    tr, tc = float(target[0]), float(target[1])
    key_r2 = (round(tr, 2), round(tc, 2))
    if key_r2 in dists:
        return dists[key_r2]
    key_r1 = (round(tr, 1), round(tc, 1))
    if key_r1 in dists:
        return dists[key_r1]
    best_info = None
    min_d = 0.5
    for k, v in dists.items():
        d = abs(k[0] - tr) + abs(k[1] - tc)
        if d < min_d:
            min_d = d
            best_info = v
    return best_info

def pellet_threat(ghost, pac_pos) -> Optional[tuple]:
    if pac_pos is None or not getattr(ghost, 'known_power_pellets', None):
        return None
    p_dir = getattr(ghost, '_player_dir', (0.0, 0.0))
    p_speed = math.hypot(p_dir[0], p_dir[1])
    best, bd = None, PELLET_THREAT_DIST
    for px, py in ghost.known_power_pellets:
        dy, dx = py - pac_pos[0], px - pac_pos[1]
        d = math.hypot(dy, dx)
        if d >= bd:
            continue
        if p_speed > 0.05 and d > 1.0 and (dy * p_dir[0] + dx * p_dir[1]) / (d * p_speed) < -0.2:
            continue   #Pacman is heading away from this pellet
        bd, best = d, (py, px)
    return best

def _score_hunt(ghost, dists: dict, frame: int) -> list[Task]:
    if ghost.pacman_powered:
        return []
    hunt_targets = []  #list of (target_tuple, confidence_weight, is_primary)
    if ghost.known_pacman is not None:
        hunt_targets.append((ghost.known_pacman, 1.0, True))
    elif ghost.last_lost_pacman is not None and (frame - getattr(ghost, 'pacman_last_seen', 0) < 30):
        hunt_targets.append((ghost.last_lost_pacman, 0.85, True))
    elif hasattr(ghost, 'belief_map') and ghost.belief_map is not None:
        top = ghost.belief_map.top_cells(n=3)
        for idx, cell in enumerate(top):
            p = ghost.belief_map.probability_at(cell)
            conf = max(0.4, min(1.0, p * 5.0))
            hunt_targets.append(((float(cell[0]), float(cell[1])), conf, idx == 0))
    if not hunt_targets:
        return []
    tasks = []
    threat = pellet_threat(ghost, hunt_targets[0][0]) if hunt_targets[0][2] else None
    for target, conf, is_primary in hunt_targets:
        pr, pc = target
        pr_r, pc_r = round(float(pr), 1), round(float(pc), 1)
        info = _lookup_dist(dists, target)
        if info is None or info[0] == math.inf:
            continue
        dist = info[0]
        #gradient-based hunt score: linear base + steep acceleration as distance decreases
        base_score = _dist_score(dist, HUNT_SCALE)
        close_gradient = 2.5 * math.exp(-dist / 6.0)
        score = (1.2 + 2.0 * base_score + close_gradient) * conf
        if threat is not None and dist < PELLET_THREAT_HOLD:
            score *= PELLET_THREAT_DAMP
        tasks.append(Task(task_type=TaskType.HUNT, target_pos=(pr_r, pc_r), score=score, assigned_to=-1, created_frame=frame, owner=ghost.gid, target_speed=1.0))
    return tasks

def _score_convert(ghost, dists: dict, frame: int) -> List[Task]:
    tasks: list[Task] = []
    if not hasattr(ghost, 'known_power_pellets'): return tasks
    #pacman estimate comes only from LOS, relayed comms or this ghost's own belief map
    pac = ghost.known_pacman or ghost.last_lost_pacman
    if pac is None and getattr(ghost, 'belief_map', None) is not None:
        top = ghost.belief_map.top_cells(n=1)
        pac = top[0] if top else None
    powered = getattr(ghost, 'pacman_powered', False)
    for pos in ghost.known_power_pellets:
        yx_pos = (pos[1], pos[0])
        info = _lookup_dist(dists, yx_pos)
        if info is None:
            continue
        dist, _ = info
        if dist == math.inf:
            continue
        #distance-dependent conversion score that prioritizes power pellet denial when nearby
        score = 2.0 + 3.0 * _dist_score(dist, CONVERT_SCALE)
        #denial urgency: every pellet Pacman reaches costs the swarm a full powered phase,
        if pac is not None:
            d_pac = abs(pac[0] - yx_pos[0]) + abs(pac[1] - yx_pos[1])
            threat = math.exp(-d_pac / 7.0)
            race = 1.0 if dist <= d_pac * CONVERT_RACE_RATIO else CONVERT_RACE_LOSS
            score += CONVERT_DENIAL_W * threat * race
        if powered:
            score *= 0.4   #never walk into a powered Pacman for a pellet
        tasks.append(Task(task_type=TaskType.CONVERT, target_pos=yx_pos, score=score, created_frame=frame, owner=ghost.gid, target_speed=1.0))
    return tasks

def _find_flee_pos(ghost, pacman_pos: tuple) -> Optional[tuple]:
    pr, pc = pacman_pos
    if getattr(ghost, 'belief_map', None) is None: return None
    from pathfinder import find_topological_flee_target_belief
    top_target = find_topological_flee_target_belief(ghost.belief_map, (ghost.y, ghost.x), (pr, pc))
    if top_target is not None:
        return top_target
    prm_nodes = getattr(ghost.belief_map, '_open_cells', None)
    #direction vector from Pacman to Ghost (away from Pacman)
    d_pac_ghost = (ghost.y - pr, ghost.x - pc)
    cur_dist = math.hypot(d_pac_ghost[0], d_pac_ghost[1])
    dir_away = (d_pac_ghost[0] / cur_dist, d_pac_ghost[1] / cur_dist) if cur_dist > 0.01 else (0.0, 0.0)
    if prm_nodes:
        best_node = None
        best_safety = -math.inf
        for n in prm_nodes:
            pac_d = math.hypot(n[0] - pr, n[1] - pc)
            ghost_d = math.hypot(n[0] - ghost.y, n[1] - ghost.x)
            d_ghost_node = (n[0] - ghost.y, n[1] - ghost.x)
            node_dist = math.hypot(d_ghost_node[0], d_ghost_node[1])
            align = (d_ghost_node[0] * dir_away[0] + d_ghost_node[1] * dir_away[1]) / (node_dist + 1e-6) if node_dist > 0.01 else 0.0
            #penalize nodes that are closer to Pacman than the ghost currently is
            safety = pac_d - 0.4 * ghost_d + 6.0 * align
            if pac_d < cur_dist:
                safety -= 25.0
            if pac_d >= SAFE_RADIUS:
                safety += 10.0
            if safety > best_safety:
                best_safety = safety
                best_node = (float(n[0]), float(n[1]))
        if best_node is not None and best_safety > -10.0:
            return best_node
    corners = [(1.5, 1.5), (1.5, float(ghost.world.width - 2)), (float(ghost.world.height - 2), 1.5), (float(ghost.world.height - 2), float(ghost.world.width - 2))]
    passable_corners = [c for c in corners if ghost.world.is_passable(c[1], c[0], radius=0.4)] or corners
    best_corner = None
    best_dist = -math.inf
    for cr, cc in passable_corners:
        d = math.hypot(cr - pr, cc - pc)
        d_g = math.hypot(cr - ghost.y, cc - ghost.x)
        c_score = d - 0.3 * d_g
        if d < cur_dist:
            c_score -= 20.0
        if c_score > best_dist:
            best_dist = c_score
            best_corner = (cr, cc)
    return best_corner

def _score_evade_track(ghost, dists: dict, frame: int) -> Optional[Task]:
    if not ghost.pacman_powered:
        return None
    target = ghost.known_pacman or ghost.last_lost_pacman
    if target is None and hasattr(ghost, 'belief_map') and ghost.belief_map is not None:
        top = ghost.belief_map.top_cells(n=1)
        if top:
            target = top[0]
    if target is None:
        return None
    info = _lookup_dist(dists, target)
    dist = info[0] if (info is not None and info[0] != math.inf) else math.hypot(ghost.y - target[0], ghost.x - target[1])
    flee_pos = _find_flee_pos(ghost, target)
    if flee_pos is None:
        return None
    if dist < SAFE_RADIUS:              #high-priority emergency flee task
        score = 8.0 + 4.0 * math.exp(-dist / 4.0)
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=flee_pos, score=score, created_frame=frame, owner=ghost.gid, target_speed=1.0)
    else:                               #maintain distance and stay in safe quadrant
        score = 2.5
        return Task(task_type=TaskType.EVADE_TRACK, target_pos=flee_pos, score=score, created_frame=frame, owner=ghost.gid, target_speed=0.8)

def _score_explore(ghost, frame: int) -> List[Task]:
    ls = getattr(ghost, 'prm_last_seen', {})
    if not ls:
        return []
    bm = getattr(ghost, 'belief_map', None)
    if bm is not None and hasattr(bm, '_open_cells') and len(bm._open_cells) > 0 and len(getattr(bm, '_b_flat', [])) > 0:
        bm._ensure_initialised()
        open_cells = bm._open_cells
        open_arr = bm._open_arr
        b_flat = bm._b_flat
        n = len(open_cells)
        if len(open_arr) == n and len(b_flat) == n:
            ls_arr = np.array([ls.get(node, -1) for node in open_cells], dtype=np.int32)
            ages_np = np.where(ls_arr < 0, frame + UNKNOWN_BONUS, frame - ls_arr)
            recency = 1.0 - np.exp(-ages_np / RECENCY_SCALE)
            manhattan = np.abs(open_arr[:, 0] - ghost.y) + np.abs(open_arr[:, 1] - ghost.x)
            dist_factor = np.exp(-manhattan / EXPLORE_SCALE)
            belief_mult = 1.0 + 8.0 * b_flat
            scores = 0.5 * recency * dist_factor * belief_mult
            k = min(EXPLORE_TOP_K, n)
            if k <= 0:
                return []
            top_k = np.argpartition(-scores, k - 1)[:k]
            top_k = top_k[np.argsort(-scores[top_k])]
            bundle_nodes = set(key[1] for key in ghost.cbba_agent.bundle) if getattr(ghost, 'cbba_agent', None) else set()
            tasks: list = []
            for idx in top_k:
                pos = open_cells[idx]
                score = float(scores[idx])
                if pos in bundle_nodes:
                    score += 0.5
                tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score, created_frame=frame, owner=ghost.gid, target_speed=0.8))
            return tasks
    ages = {}
    for node, last_seen_frame in ls.items():
        if last_seen_frame < 0:
            ages[node] = frame + UNKNOWN_BONUS
        else:
            ages[node] = frame - last_seen_frame
    prob_map = {pos: bm.probability_at(pos) for pos in ages} if bm is not None else {}
    scored_nodes = []
    for pos, age in ages.items():
        recency = 1.0 - math.exp(-age / RECENCY_SCALE)
        dist_factor = _dist_score(abs(pos[0] - ghost.y) + abs(pos[1] - ghost.x), EXPLORE_SCALE)
        belief_mult = 1.0 + 8.0 * float(prob_map.get(pos, 0.0))
        scored_nodes.append((pos, 0.5 * recency * dist_factor * belief_mult))
    scored_nodes.sort(key=lambda item: item[1], reverse=True)
    bundle_nodes = set(key[1] for key in ghost.cbba_agent.bundle) if getattr(ghost, 'cbba_agent', None) else set()
    tasks: list = []
    for pos, score in scored_nodes[:EXPLORE_TOP_K]:
        if pos in bundle_nodes:
            score += 0.5
        tasks.append(Task(task_type=TaskType.EXPLORE, target_pos=pos, score=score, created_frame=frame, owner=ghost.gid, target_speed=0.8))
    return tasks

def generate_tasks(ghost, frame: int) -> tuple[List[Task], dict]:
    start = (float(ghost.y), float(ghost.x))
    targets: set = set()
    if getattr(ghost, 'pacman_powered', False):
        evade_target = ghost.known_pacman or ghost.last_lost_pacman
        if evade_target is None and hasattr(ghost, 'belief_map') and ghost.belief_map is not None:
            top = ghost.belief_map.top_cells(n=1)
            if top:
                evade_target = top[0]
        if evade_target is not None:
            targets.add((float(evade_target[0]), float(evade_target[1])))
            flee_p = _find_flee_pos(ghost, evade_target)
            if flee_p is not None:
                targets.add(flee_p)
    else:
        pac_pos = ghost.known_pacman or ghost.last_lost_pacman
        if pac_pos is not None:
            pr, pc = float(pac_pos[0]), float(pac_pos[1])
            targets.add((pr, pc))
        elif hasattr(ghost, 'belief_map') and ghost.belief_map is not None:
            top = ghost.belief_map.top_cells(n=3)
            for cell in top:
                targets.add((float(cell[0]), float(cell[1])))
    for p in getattr(ghost, 'known_power_pellets', []):
        targets.add((p[1], p[0]))
    explore_tasks = _score_explore(ghost, frame)
    for et in explore_tasks:
        targets.add((float(et.target_pos[0]), float(et.target_pos[1])))
    dists = ghost_dists(ghost, start, list(targets))
    tasks: list[Task] = []
    if getattr(ghost, 'pacman_powered', False):
        evade_track = _score_evade_track(ghost, dists, frame)
        if evade_track is not None:
            if evade_track.target_pos not in dists:
                extra_dist = ghost_dists(ghost, start, [evade_track.target_pos])
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