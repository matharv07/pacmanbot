"""
Observation construction and action-to-task bridging for the MAPPO pipeline.

Builds the 16-channel spatial tensor and ~100-dim vector tensor per ghost,
and converts the RL actor's sampled waypoints back into CBBA Task objects.
"""

import math
import numpy as np
from allocator import Task, TaskType, ORIGIN_RL_ENDORSE, ORIGIN_RL_NOVEL

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
RL_SCORE_BASE   = 0.2
RL_SCORE_SPAN   = 10.0
RL_ENDORSE_GAIN = 2.0

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

def actions_to_tasks(ghost, scores_map: np.ndarray, indices: list, frame: int, obs_resolution: float = 1.0, target_speed: float = 1.0) -> list:
    if not isinstance(scores_map, np.ndarray):
        scores_map = np.array(scores_map, dtype=np.float32)
    if scores_map.ndim < 2:
        rows = int(getattr(ghost.world, 'height', 10) * obs_resolution)
        cols = int(getattr(ghost.world, 'width', 10) * obs_resolution)
        new_map = np.zeros((rows, cols), dtype=np.float32)
        for idx_i, (r, c) in enumerate(indices):
            if 0 <= r < rows and 0 <= c < cols:
                val = float(scores_map[idx_i]) if idx_i < len(scores_map) else 1.0
                new_map[r, c] = val
        scores_map = new_map
    rows, cols = scores_map.shape
    tasks = []
    target = _pacman_target(ghost)
    bm_top = []
    if hasattr(ghost, 'belief_map') and ghost.belief_map is not None and hasattr(ghost.belief_map, 'top_cells'):
        bm_top = ghost.belief_map.top_cells(n=5)
    cands = getattr(ghost, '_rl_candidates', None) or []
    for r, c in indices:
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        world_y = (float(r) + 0.5) / obs_resolution
        world_x = (float(c) + 0.5) / obs_resolution
        if not ghost.world.is_passable(world_x, world_y, radius=0.35):
            continue
        rel = min(1.0, max(0.0, float(scores_map[r, c])))
        conf = rel * rel
        score = RL_SCORE_BASE + RL_SCORE_SPAN * conf
        near = None
        for t in cands:
            if abs(t.target_pos[0] - world_y) + abs(t.target_pos[1] - world_x) <= 1.5:
                near = t
                break
        if near is not None:
            tasks.append(Task(task_type=near.task_type, target_pos=near.target_pos, score=max(float(near.score) * (1.0 + RL_ENDORSE_GAIN * conf), score), created_frame=frame, owner=ghost.gid, assigned_to=ghost.gid, target_speed=target_speed, origin=ORIGIN_RL_ENDORSE))
            continue
        is_power = any(abs(world_y - p[1]) < 0.5 and abs(world_x - p[0]) < 0.5 for p in ghost.known_power_pellets)
        near_belief = any((abs(world_y - bc[0]) + abs(world_x - bc[1])) <= 3.0 for bc in bm_top)
        if is_power:
            tt = TaskType.CONVERT
        elif (target is not None and (abs(world_y - target[0]) + abs(world_x - target[1])) <= 3.0) or near_belief:
            tt = TaskType.HUNT
        else:
            tt = TaskType.DYNAMIC
        tasks.append(Task(task_type=tt, target_pos=(world_y, world_x), score=score, created_frame=frame, owner=ghost.gid, assigned_to=ghost.gid, target_speed=target_speed, origin=ORIGIN_RL_NOVEL))
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