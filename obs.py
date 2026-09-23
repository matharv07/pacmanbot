"""
Observation construction and action-to-task bridging for the MAPPO pipeline.

Builds the 16-channel spatial tensor and ~100-dim vector tensor per ghost,
and converts the RL actor's sampled waypoints back into CBBA Task objects.
"""

import math
import random as _random
import numpy as np
from allocator import Task, TaskType, ORIGIN_RL_ENDORSE, ORIGIN_RL_NOVEL, _lookup_dist
import os as _os

WALL    = 1
PELLET  = 2
POWER   = 3
UNKNOWN = -1
MAX_H = 33
MAX_W = 41
MAX_GHOSTS   = 7
SPATIAL_CH   = 12        #number of spatial channels (see channel map below)
GLOBAL_SPATIAL_CH = 12   #number of channels in the omniscient global state
VEC_DIM      = 70
CRITIC_VEC_DIM = MAX_GHOSTS * VEC_DIM + MAX_GHOSTS
RL_SCORE_BASE   = float(_os.environ.get("RL_SCORE_BASE", "0.2"))
RL_SCORE_SPAN   = float(_os.environ.get("RL_SCORE_SPAN", "5.0"))
RL_ENDORSE_GAIN = float(_os.environ.get("RL_ENDORSE_GAIN", "2.0"))
MAX_CANDIDATES  = int(_os.environ.get("MAX_CANDIDATES", "24"))
CAND_FEAT_DIM   = 16

"""
Channel Map:
    0  is_wall           4  belief_map       8  peer_ghosts (with staleness decay)
    1  is_pellet         5  safety_map       9  staleness
    2  is_power          6  own_position     10 recent_nominations
    3  peer_intent       7  pacman_position  11 heuristic_candidates (score / max score)
"""

def _pacman_target(ghost):
    t = ghost.known_pacman
    if t is not None:
        return t
    t = ghost.last_lost_pacman
    if t is not None and (getattr(ghost, 'frame', 0) - getattr(ghost, 'pacman_last_seen', 0) <= 25):
        return t
    if hasattr(ghost.belief_map, 'top_cells'):
        top = ghost.belief_map.top_cells(n=1)
        if top:
            return top[0]
    return None

_INV_TWO_SIGMA_SQ = 1.0 / (2.0 * 0.6 * 0.6)

def _place_single_pixel(channel, fx, fy, rows: int, cols: int, obs_res: float):
    r, c = int(fy * obs_res), int(fx * obs_res)
    if 0 <= r < rows and 0 <= c < cols:
        channel[r, c] = 1.0

def _place_blob(channel, fy, fx, rows: int, cols: int, obs_res: float, scale: float = 1.0):
    y_scaled = fy * obs_res
    x_scaled = fx * obs_res
    cr, cc = int(y_scaled), int(x_scaled)
    for dr in (-1, 0, 1):
        nr = cr + dr
        if 0 <= nr < rows:
            dy = y_scaled - (nr + 0.5)
            dy2 = dy * dy
            for dc in (-1, 0, 1):
                nc = cc + dc
                if 0 <= nc < cols:
                    dx = x_scaled - (nc + 0.5)
                    d2 = dy2 + dx * dx
                    val = math.exp(-d2 * _INV_TWO_SIGMA_SQ) * scale
                    if val > channel[nr, nc]:
                        channel[nr, nc] = val

def build_spatial(ghost, recent_noms: np.ndarray, rows: int, cols: int, obs_resolution: float = 1.0) -> np.ndarray:
    """Returns (SPATIAL_CH, rows, cols) float32 tensor."""
    out = np.zeros((SPATIAL_CH, rows, cols), dtype=np.float32)
    for hy, hx in ghost.lidar_memory:
        r_idx = int(hy * obs_resolution)
        c_idx = int(hx * obs_resolution)
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            out[0, r_idx, c_idx] = 1.0
            if r_idx > 0: out[0, r_idx-1, c_idx] = 1.0
            if r_idx < rows-1: out[0, r_idx+1, c_idx] = 1.0
            if c_idx > 0: out[0, r_idx, c_idx-1] = 1.0
            if c_idx < cols-1: out[0, r_idx, c_idx+1] = 1.0
    for p in ghost.known_pellets:
        _place_single_pixel(out[1], p[0], p[1], rows, cols, obs_resolution)
    for p in ghost.known_power_pellets:
        _place_single_pixel(out[2], p[0], p[1], rows, cols, obs_resolution)
    if hasattr(ghost, 'cbba_agent'):
        for gid in range(MAX_GHOSTS):
            if gid == ghost.gid:
                continue
            t = ghost.cbba_agent.get_known_task_for(gid)
            if t is not None and t.target_pos is not None:
                ty, tx = float(t.target_pos[0]), float(t.target_pos[1])
                task_speed = float(getattr(t, 'target_speed', 1.0))
                is_flank = getattr(t, 'task_type', None) == TaskType.FLANK
                blob_scale = (1.4 if is_flank else 1.0) * task_speed
                _place_blob(out[3], ty, tx, rows, cols, obs_resolution, scale=blob_scale)
                peer_pos = ghost.known_agents.get(gid)
                py, px = None, None
                if peer_pos is not None and peer_pos != "UNKNOWN":
                    py, px = float(peer_pos[0]), float(peer_pos[1])
                elif hasattr(ghost, '_last_known_agent_pos') and gid in ghost._last_known_agent_pos:
                    lpos = ghost._last_known_agent_pos[gid]
                    if isinstance(lpos, (tuple, list)) and len(lpos) >= 2:
                        py, px = float(lpos[0]), float(lpos[1])
                if py is not None and px is not None:
                    dy = ty - py
                    dx = tx - px
                    dist = math.hypot(dy, dx)
                    if dist > 0.5:
                        steps = max(int(dist * obs_resolution * 2), 2)
                        for step in range(steps + 1):
                            iy = py + dy * (step / steps)
                            ix = px + dx * (step / steps)
                            r = int(iy * obs_resolution)
                            c = int(ix * obs_resolution)
                            if 0 <= r < rows and 0 <= c < cols:
                                line_val = (0.70 if is_flank else 0.35) * task_speed
                                if line_val > out[3, r, c]:
                                    out[3, r, c] = line_val
    bm = ghost.belief_map
    if hasattr(bm, '_open_arr') and len(bm._open_arr) > 0:
        cache = getattr(bm, '_spatial_rc_cache', None)
        cache_key = (obs_resolution, rows, cols)
        if cache is None or cache[0] != cache_key:
            r_arr = (bm._open_arr[:, 0] * obs_resolution).astype(np.int32)
            c_arr = (bm._open_arr[:, 1] * obs_resolution).astype(np.int32)
            valid = (r_arr >= 0) & (r_arr < rows) & (c_arr >= 0) & (c_arr < cols)
            valid_indices = np.where(valid)[0]
            bm._spatial_rc_cache = (cache_key, r_arr[valid], c_arr[valid], valid_indices)
        _, r_valid, c_valid, valid_indices = bm._spatial_rc_cache
        if hasattr(bm, '_b_flat') and bm._initialised:
            if len(valid_indices) <= len(bm._b_flat):
                np.maximum.at(out[4], (r_valid, c_valid), bm._b_flat[valid_indices])
                bm_peak = float(out[4].max())
                if bm_peak > 1e-6:
                    out[4] /= bm_peak
        if hasattr(bm, '_safety'):
            if len(valid_indices) <= len(bm._safety):
                np.maximum.at(out[5], (r_valid, c_valid), bm._safety[valid_indices])
    _place_blob(out[6], ghost.y, ghost.x, rows, cols, obs_resolution)
    target = _pacman_target(ghost)
    if target is not None:
        tr, tc = target
        _place_blob(out[7], float(tr), float(tc), rows, cols, obs_resolution)
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        pos = ghost.known_agents.get(gid)
        if pos is not None and pos != "UNKNOWN":
            _place_blob(out[8], float(pos[0]), float(pos[1]), rows, cols, obs_resolution)
        elif hasattr(ghost, '_last_known_agent_pos') and gid in ghost._last_known_agent_pos:
            last_pos = ghost._last_known_agent_pos[gid]
            if isinstance(last_pos, (tuple, list)) and len(last_pos) >= 2:
                last_frame = ghost.last_heartbeat.get(gid, ghost.frame) if hasattr(ghost, 'last_heartbeat') else ghost.frame
                delta_frames = max(0, ghost.frame - last_frame)
                decay = math.exp(-delta_frames / 30.0)
                if decay > 0.05:
                    _place_blob(out[8], float(last_pos[0]), float(last_pos[1]), rows, cols, obs_resolution, scale=decay)  
    stale_ch = np.ones((rows, cols), dtype=np.float32)
    if ghost.prm_last_seen:
        cur_frame = ghost.frame
        for (pr, pc), ls in ghost.prm_last_seen.items():
            if ls >= 0:
                ri = int(pr * obs_resolution)
                ci = int(pc * obs_resolution)
                if 0 <= ri < rows and 0 <= ci < cols:
                    stale_val = (cur_frame - ls) / 200.0
                    if stale_val < 0.0:
                        stale_val = 0.0
                    elif stale_val > 1.0:
                        stale_val = 1.0
                    stale_ch[ri, ci] = stale_val
    out[9] = stale_ch
    out[10] = recent_noms[:rows, :cols]
    #the heuristic's current proposals, so the actor arbitrates over them instead of guessing blind
    cands = getattr(ghost, '_rl_candidates', None)
    if cands:
        s_max = max(float(t.score) for t in cands) or 1.0
        for t in cands:
            r_t, c_t = int(t.target_pos[0] * obs_resolution), int(t.target_pos[1] * obs_resolution)
            if 0 <= r_t < rows and 0 <= c_t < cols:
                out[11, r_t, c_t] = max(out[11, r_t, c_t], float(t.score) / s_max)
    return out

def build_vector(ghost) -> np.ndarray:
    w_height = ghost.world.height
    w_width = ghost.world.width
    f = []
    f.extend([ghost.y / w_height, ghost.x / w_width])
    f.extend([ghost.y % 1.0, ghost.x % 1.0])
    timer = getattr(ghost, 'pacman_power_timer', 0)
    f.append(timer / 40.0 if getattr(ghost, 'pacman_powered', False) else 0.0)
    since = ghost.frame - ghost.pacman_last_seen if ghost.pacman_last_seen >= 0 else 200
    f.append(min(since, 200) / 200.0)
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        st = ghost.known_agents.get(gid)
        is_dead = 1.0 if (hasattr(ghost, 'is_agent_dead') and ghost.is_agent_dead(gid)) or (hasattr(ghost, 'dead_agents') and gid in ghost.dead_agents) else 0.0
        is_unknown = 1.0 if (st == "UNKNOWN" or st is None) and not is_dead else 0.0
        f.append(is_unknown)
        f.append(is_dead)
    n_dead = sum(1.0 for gid in range(MAX_GHOSTS) if gid != ghost.gid and ((hasattr(ghost, 'is_agent_dead') and ghost.is_agent_dead(gid)) or (hasattr(ghost, 'dead_agents') and gid in ghost.dead_agents)))
    f.append(n_dead / float(MAX_GHOSTS - 1))
    f.append(min(ghost.frame, 2000) / 2000.0)
    f.append(1.0 if getattr(ghost, 'in_fallback_mode', False) else 0.0)
    speed = math.hypot(ghost.vx, ghost.vy)
    max_speed = getattr(ghost, 'max_speed', 0.50)
    f.append(speed / max_speed if max_speed > 0 else 0.0)
    f.extend([ghost.vy / 5.0, ghost.vx / 5.0])
    target = _pacman_target(ghost)
    if target:
        f.extend([(target[0] - ghost.y) / w_height, (target[1] - ghost.x) / w_width])
    else:
        f.extend([0.0, 0.0])
    if hasattr(ghost.world, '_points_to_segments_dist_sq'):
        dist_sq, _ = ghost.world._points_to_segments_dist_sq(np.array([ghost.x]), np.array([ghost.y]))
        min_dist = math.sqrt(np.min(dist_sq)) if dist_sq.size > 0 else 10.0
        f.append(min(min_dist, 10.0) / 10.0)
    else:
        f.append(1.0)
    #Agent ID One-Hot across MAX_GHOSTS=7 (7 dims)
    agent_id_one_hot = [0.0] * MAX_GHOSTS
    if 0 <= ghost.gid < MAX_GHOSTS:
        agent_id_one_hot[ghost.gid] = 1.0
    f.extend(agent_id_one_hot)

    def _enc(t):
        if t is None:
            return [0.0] * 12
        v = [0.0] * 12
        tt = int(t.task_type)
        if 0 <= tt < 6:
            v[tt] = 1.0
        v[6] = t.target_pos[0] / w_height
        v[7] = t.target_pos[1] / w_width
        v[8] = min(max(t.score, -5.0), 5.0) / 5.0
        v[9] = min(ghost.frame - t.created_frame, 200) / 200.0
        v[10] = float(getattr(t, 'target_speed', 1.0))
        v[11] = 1.0
        return v
    own = []
    for key in ghost.cbba_agent.path[:3]:
        task = ghost.cbba_agent._task_map.get(key)
        if task:
            own.append(task)
    while len(own) < 3:
        own.append(None)
    for t in own:
        f.extend(_enc(t))

    return np.asarray(f, dtype=np.float32)

def build_valid_mask(ghost, rows: int, cols: int, obs_resolution: float = 1.0, spatial_walls: np.ndarray = None) -> np.ndarray:
    if spatial_walls is not None and spatial_walls.shape == (rows, cols):
        mask = (spatial_walls < 0.5)
    else:
        mask = np.ones((rows, cols), dtype=bool)
        for hy, hx in ghost.lidar_memory:
            r_idx = int(hy * obs_resolution)
            c_idx = int(hx * obs_resolution)
            if 0 <= r_idx < rows and 0 <= c_idx < cols:
                mask[r_idx, c_idx] = False
                if r_idx > 0: mask[r_idx-1, c_idx] = False
                if r_idx < rows-1: mask[r_idx+1, c_idx] = False
                if c_idx > 0: mask[r_idx, c_idx-1] = False
                if c_idx < cols-1: mask[r_idx, c_idx+1] = False
    if hasattr(ghost, 'cbba_agent') and hasattr(ghost.cbba_agent, '_unreachable_cache'):
        for pos, timeout_frame in ghost.cbba_agent._unreachable_cache.items():
            if timeout_frame > ghost.frame:
                r, c = int(pos[0] * obs_resolution), int(pos[1] * obs_resolution)
                if 0 <= r < rows and 0 <= c < cols:
                    mask[r, c] = False
    return mask

def select_candidates(tasks: list, seed: int = 0) -> list:
    if not tasks:
        return []
    ordered = sorted(tasks, key=lambda t: -float(t.score))[:MAX_CANDIDATES]
    _random.Random(seed).shuffle(ordered)
    return ordered

def flatten_cand_cells(cells, width: int):
    return cells[..., 0] * int(width) + cells[..., 1]

def build_candidates(ghost, rows: int, cols: int, obs_resolution: float = 1.0, dists: dict = None):
    """Per-candidate features for the actor's pointer head.

    Returns
    -------
    feats  : (MAX_CANDIDATES, CAND_FEAT_DIM) float32
    cells  : (MAX_CANDIDATES, 2) int64 — (row, col) of each target. Kept unflattened because observations are
             later zero-padded to the stage grid, which changes the row stride but not (row, col).
    mask   : (MAX_CANDIDATES,) bool   — True where a real candidate sits
    bc_tgt : (MAX_CANDIDATES,) float32 — the heuristic's own score, the BC target over this set
    """
    feats  = np.zeros((MAX_CANDIDATES, CAND_FEAT_DIM), dtype=np.float32)
    cells  = np.zeros((MAX_CANDIDATES, 2), dtype=np.int64)
    mask   = np.zeros((MAX_CANDIDATES,), dtype=bool)
    bc_tgt = np.zeros((MAX_CANDIDATES,), dtype=np.float32)
    cands = (getattr(ghost, '_rl_candidates', None) or [])[:MAX_CANDIDATES]
    if not cands:
        return feats, cells, mask, bc_tgt
    h = float(getattr(ghost.world, 'height', rows)) or float(rows)
    w = float(getattr(ghost.world, 'width', cols)) or float(cols)
    bm_top = []
    if getattr(ghost, 'belief_map', None) is not None and hasattr(ghost.belief_map, 'top_cells'):
        bm_top = ghost.belief_map.top_cells(n=5)
    peer_targets = []
    if hasattr(ghost, 'cbba_agent'):
        for gid in range(MAX_GHOSTS):
            if gid == ghost.gid:
                continue
            pt = ghost.cbba_agent.get_known_task_for(gid)
            if pt is not None and pt.target_pos is not None:
                peer_targets.append((float(pt.target_pos[0]), float(pt.target_pos[1])))
    own_path = set()
    if hasattr(ghost, 'cbba_agent'):
        try:
            own_path = set(ghost.cbba_agent.path)
        except Exception:
            own_path = set()
    n = len(cands)
    rank_of = {id(t): r for r, t in enumerate(sorted(cands, key=lambda z: -float(z.score)))}
    for i, t in enumerate(cands):
        ty, tx = float(t.target_pos[0]), float(t.target_pos[1])
        r_t = min(max(int(ty * obs_resolution), 0), rows - 1)
        c_t = min(max(int(tx * obs_resolution), 0), cols - 1)
        cells[i, 0], cells[i, 1] = r_t, c_t
        mask[i]  = True
        f = feats[i]
        tt = int(t.task_type)
        if 0 <= tt < 6:
            f[tt] = 1.0
        f[6]  = min(max(float(t.score), -5.0), 5.0) / 5.0
        f[7]  = min(max(ghost.frame - int(t.created_frame), 0), 200) / 200.0
        f[8]  = float(getattr(t, 'target_speed', 1.0))
        f[9]  = (ty - ghost.y) / h
        f[10] = (tx - ghost.x) / w
        d_path = None
        if dists:
            info = _lookup_dist(dists, (ty, tx))
            if info and info[0] != math.inf:
                d_path = float(info[0])
        if d_path is None:
            d_path = abs(ty - ghost.y) + abs(tx - ghost.x)
        f[11] = min(d_path, 40.0) / 40.0
        f[12] = 1.0 if _task_key_local(t) in own_path else 0.0
        if bm_top:
            d_b = min(abs(ty - b[0]) + abs(tx - b[1]) for b in bm_top)
            f[13] = math.exp(-d_b / 3.0)
        if peer_targets:
            d_p = min(abs(ty - p[0]) + abs(tx - p[1]) for p in peer_targets)
            f[14] = math.exp(-d_p / 3.0)
        f[15] = rank_of[id(t)] / float(max(1, n - 1))
        bc_tgt[i] = max(0.0, float(t.score))
    return feats, cells, mask, bc_tgt

def _task_key_local(task) -> tuple:
    #mirrors cbba._task_key without importing cbba (obs is imported by cbba's callers)
    return (int(task.task_type), (round(float(task.target_pos[0]), 1), round(float(task.target_pos[1]), 1)))

AUTH_SCORE = float(_os.environ.get("AUTH_SCORE", "100.0"))   #a confident pick must win its OWN auction outright

def authoritative_task(ghost, pick_idx: int, frame: int, target_speed: float = 1.0):
    """The actor's chosen candidate, made authoritative for THIS ghost."""
    cands = getattr(ghost, '_rl_candidates', None) or []
    if pick_idx is None or pick_idx < 0 or pick_idx >= min(len(cands), MAX_CANDIDATES):
        return None
    src = cands[int(pick_idx)]
    return Task(task_type=src.task_type, target_pos=src.target_pos, score=AUTH_SCORE, created_frame=frame,
                owner=ghost.gid, assigned_to=ghost.gid, target_speed=target_speed, origin=ORIGIN_RL_ENDORSE)

def build_cve(gids, ve, max_ghosts: int = MAX_GHOSTS, vec_dim: int = VEC_DIM):
    """Critic vector: every alive ghost's vector in gid order, plus a one-hot of which ghost this row is for.
    (N, MAX_GHOSTS*VEC_DIM + MAX_GHOSTS). Shared by the trainer and every evaluation path."""
    n = len(gids)
    joint = np.zeros((max_ghosts, vec_dim), dtype=np.float32)
    for i, gid in enumerate(gids):
        joint[gid] = ve[i]
    flat = joint.reshape(-1)
    out = np.zeros((n, max_ghosts * vec_dim + max_ghosts), dtype=np.float32)
    out[:, :max_ghosts * vec_dim] = flat
    for i, gid in enumerate(gids):
        out[i, max_ghosts * vec_dim + gid] = 1.0
    return out

def actions_to_tasks(ghost, cand_scores, cand_picks, frame: int, obs_resolution: float = 1.0, target_speed: float = 1.0, novel_scores=None, novel_indices=None) -> list:
    """Turn the actor's spatial decisions into CBBA nominations."""
    tasks = []
    scores_map = novel_scores if novel_scores is not None else cand_scores
    indices = novel_indices if novel_indices is not None else cand_picks
    if scores_map is None or indices is None:
        return tasks
    scores_arr = np.asarray(scores_map, dtype=np.float32)
    target = _pacman_target(ghost)
    bm_top = []
    if getattr(ghost, 'belief_map', None) is not None and hasattr(ghost.belief_map, 'top_cells'):
        bm_top = ghost.belief_map.top_cells(n=5)
    if scores_arr.ndim >= 2:
        rows, cols = scores_arr.shape[-2], scores_arr.shape[-1]
        seen = set()
        for item in indices:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                r, c = int(item[0]), int(item[1])
            else:
                idx = int(item)
                r, c = idx // cols, idx % cols
            if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in seen:
                continue
            seen.add((r, c))
            world_y = (float(r) + 0.5) / obs_resolution
            world_x = (float(c) + 0.5) / obs_resolution
            if hasattr(ghost, 'world') and not ghost.world.is_passable(world_x, world_y, radius=0.35):
                continue
            if getattr(ghost, 'pacman_powered', False) and target is not None:
                d_pac = math.hypot(world_y - float(target[0]), world_x - float(target[1]))
                if d_pac < 10.0:
                    continue
            conf = min(1.0, max(0.0, float(scores_arr[r, c])))
            score = RL_SCORE_BASE + RL_SCORE_SPAN * conf
            power_pellets = getattr(ghost, 'known_power_pellets', None) or []
            is_power = any(abs(world_y - p[1]) < 0.5 and abs(world_x - p[0]) < 0.5 for p in power_pellets)
            near_belief = any((abs(world_y - bc[0]) + abs(world_x - bc[1])) <= 3.0 for bc in bm_top)
            if is_power:
                tt = TaskType.CONVERT
            elif (target is not None and (abs(world_y - target[0]) + abs(world_x - target[1])) <= 3.0) or near_belief:
                tt = TaskType.HUNT
            else:
                tt = TaskType.DYNAMIC
            tasks.append(Task(task_type=tt, target_pos=(world_y, world_x), score=score, created_frame=frame,
                              owner=ghost.gid, assigned_to=ghost.gid, target_speed=target_speed, origin=ORIGIN_RL_NOVEL))
    elif scores_arr.ndim == 1:
        cands = getattr(ghost, '_rl_candidates', None) or []
        seen = set()
        for slot in indices:
            slot = int(slot)
            if slot < 0 or slot >= len(cands) or slot in seen:
                continue
            seen.add(slot)
            near = cands[slot]
            conf = float(scores_arr[slot]) if slot < len(scores_arr) else 0.0
            conf = min(1.0, max(0.0, conf))
            floor = RL_SCORE_BASE + RL_SCORE_SPAN * conf
            tasks.append(Task(task_type=near.task_type, target_pos=near.target_pos, score=max(float(near.score) * (1.0 + RL_ENDORSE_GAIN * conf), floor),
                              created_frame=frame, owner=ghost.gid, assigned_to=near.assigned_to, target_speed=target_speed, origin=ORIGIN_RL_ENDORSE))
    return tasks

def build_global_spatial(env, rows: int, cols: int, obs_resolution: float = 1.0) -> np.ndarray:
    out = np.zeros((GLOBAL_SPATIAL_CH, rows, cols), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    px = (grid_x.ravel() / obs_resolution) + (0.5 / obs_resolution)
    py = (grid_y.ravel() / obs_resolution) + (0.5 / obs_resolution)
    blocked = ~env.world.batch_is_passable(px, py, radius=0.35)
    out[0] = blocked.reshape(rows, cols).astype(np.float32)
    
    def _place_single_pixel(channel, fx, fy):
        r, c = int(fy * obs_resolution), int(fx * obs_resolution)
        if 0 <= r < rows and 0 <= c < cols:
            channel[r, c] = 1.0
    for p in env.world.pellets:
        _place_single_pixel(out[1], p[0], p[1])
    for p in env.world.power_pellets:
        _place_single_pixel(out[2], p[0], p[1])
    _BLOB_SIGMA = 0.6
    def _place_blob_global(channel, fy, fx):
        cr, cc = int(fy * obs_resolution), int(fx * obs_resolution)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    d2 = (fy * obs_resolution - (nr + 0.5))**2 + (fx * obs_resolution - (nc + 0.5))**2
                    channel[nr, nc] = max(channel[nr, nc], np.exp(-d2 / (2 * _BLOB_SIGMA**2)))      
    if not env.player.dead:
        _place_blob_global(out[3], env.player.y, env.player.x)
        if getattr(env.player, 'powered', False):
            timer = getattr(env.player, 'power_timer', 0)
            out[4] = min(timer / 40.0, 1.0)
    for g in env.ghosts.values():
        if not g.dead and 0 <= g.gid < MAX_GHOSTS:
            _place_blob_global(out[5 + g.gid], g.y, g.x)
    return out