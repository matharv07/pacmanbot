"""
Observation construction and action-to-task bridging for the MAPPO pipeline.

Builds the 16-channel spatial tensor and ~100-dim vector tensor per ghost,
and converts the RL actor's sampled waypoints back into CBBA Task objects.
"""

import numpy as np
from allocator import Task, TaskType

WALL    = 1
PELLET  = 2
POWER   = 3
UNKNOWN = -1
MAX_H = 33
MAX_W = 41
MAX_GHOSTS   = 7
SPATIAL_CH   = 16        #number of spatial channels (see channel map below)
GLOBAL_SPATIAL_CH = 11   #number of channels in the omniscient global state
VEC_DIM      = 110
CRITIC_VEC_DIM = MAX_GHOSTS * VEC_DIM + MAX_GHOSTS

"""
Channel Map:
    0  is_wall           4  belief_map       8–13  other ghosts (6 ch)
    1  is_pellet         5  safety_map       14    staleness
    2  is_power          6  own_position     15    recent_nominations
    3  is_unknown        7  pacman_position
"""

def _pacman_target(ghost):
    t = ghost.known_pacman
    if t is not None:
        return t
    t = ghost.last_lost_pacman
    if t is not None:
        return t
    if hasattr(ghost.belief_map, 'top_cells'):
        top = ghost.belief_map.top_cells(n=1)
        if top:
            return top[0]
    return None


def build_spatial(ghost, recent_noms: np.ndarray, rows: int, cols: int, obs_resolution: float = 1.0) -> np.ndarray:
    """
    Returns (SPATIAL_CH, rows, cols) float32 tensor.
    """
    out = np.zeros((SPATIAL_CH, rows, cols), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    px = (grid_x.ravel() / obs_resolution) + (0.5 / obs_resolution)
    py = (grid_y.ravel() / obs_resolution) + (0.5 / obs_resolution)
    for hy, hx in ghost.lidar_memory:
        r_idx = int(hy * obs_resolution)
        c_idx = int(hx * obs_resolution)
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            out[0, r_idx, c_idx] = 1.0
            if r_idx > 0: out[0, r_idx-1, c_idx] = 1.0
            if r_idx < rows-1: out[0, r_idx+1, c_idx] = 1.0
            if c_idx > 0: out[0, r_idx, c_idx-1] = 1.0
            if c_idx < cols-1: out[0, r_idx, c_idx+1] = 1.0

    def _place_single_pixel(channel, fx, fy):
        r, c = int(fy * obs_resolution), int(fx * obs_resolution)
        if 0 <= r < rows and 0 <= c < cols:
            channel[r, c] = 1.0

    for p in ghost.known_pellets:
        _place_single_pixel(out[1], p[0], p[1])
    for p in ghost.known_power_pellets:
        _place_single_pixel(out[2], p[0], p[1])

    bm = ghost.belief_map
    if hasattr(bm, '_open_arr') and len(bm._open_arr) > 0:
        r_arr = (bm._open_arr[:, 0] * obs_resolution).astype(np.int32)
        c_arr = (bm._open_arr[:, 1] * obs_resolution).astype(np.int32)
        if hasattr(bm, '_b_flat') and bm._initialised:
            valid = (r_arr >= 0) & (r_arr < rows) & (c_arr >= 0) & (c_arr < cols) & (np.arange(len(r_arr)) < len(bm._b_flat))
            np.maximum.at(out[4], (r_arr[valid], c_arr[valid]), bm._b_flat[valid])
        if hasattr(bm, '_safety'):
            valid = (r_arr >= 0) & (r_arr < rows) & (c_arr >= 0) & (c_arr < cols) & (np.arange(len(r_arr)) < len(bm._safety))
            np.maximum.at(out[5], (r_arr[valid], c_arr[valid]), bm._safety[valid])

    _BLOB_SIGMA = 0.6

    def _place_blob(channel, fy, fx):
        cr, cc = int(fy * obs_resolution), int(fx * obs_resolution)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    d2 = (fy * obs_resolution - (nr + 0.5))**2 + (fx * obs_resolution - (nc + 0.5))**2
                    channel[nr, nc] = max(channel[nr, nc], np.exp(-d2 / (2 * _BLOB_SIGMA**2)))
                    
    _place_blob(out[6], ghost.y, ghost.x)
    target = _pacman_target(ghost)
    if target is not None:
        tr, tc = target
        _place_blob(out[7], float(tr), float(tc))
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        ch = 8 + (gid if gid < ghost.gid else gid - 1)
        pos = ghost.known_agents.get(gid)
        if pos is not None and pos != "UNKNOWN":
            _place_blob(out[ch], float(pos[0]), float(pos[1]))
            
    stale_ch = np.ones((rows, cols), dtype=np.float32)
    if ghost.prm_last_seen:
        nodes = np.array(list(ghost.prm_last_seen.keys()))
        last_seen = np.array(list(ghost.prm_last_seen.values()))
        ri = (nodes[:, 0] * obs_resolution).astype(np.int32)
        ci = (nodes[:, 1] * obs_resolution).astype(np.int32)
        valid = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
        v_ri, v_ci, v_ls = ri[valid], ci[valid], last_seen[valid]
        seen_mask = v_ls >= 0
        stale_vals = np.clip(ghost.frame - v_ls[seen_mask], 0, 200) / 200.0
        stale_ch[v_ri[seen_mask], v_ci[seen_mask]] = stale_vals
    out[14] = stale_ch
    out[15] = recent_noms[:rows, :cols]
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
        f.append(1.0 if st == "UNKNOWN" or st is None else 0.0)
    f.append(min(ghost.frame, 2000) / 2000.0)
    f.append(1.0 if getattr(ghost, 'in_fallback_mode', False) else 0.0)
    # NEW: Ghost's own continuous velocity
    import math
    speed = math.hypot(ghost.vx, ghost.vy)
    max_speed = getattr(ghost, 'max_speed', 0.5)
    f.append(speed / max_speed if max_speed > 0 else 0.0)
    f.extend([ghost.vy / 5.0, ghost.vx / 5.0])
    # NEW: Relative distance to Pacman (if known)
    target = _pacman_target(ghost)
    if target:
        f.extend([(target[0] - ghost.y) / w_height, (target[1] - ghost.x) / w_width])
    else:
        f.extend([0.0, 0.0])
    # NEW: Distance to nearest obstacle
    if hasattr(ghost.world, '_points_to_segments_dist_sq'):
        dist_sq, _ = ghost.world._points_to_segments_dist_sq(np.array([ghost.x]), np.array([ghost.y]))
        min_dist = math.sqrt(np.min(dist_sq)) if dist_sq.size > 0 else 10.0
        f.append(min(min_dist, 10.0) / 10.0)
    else:
        f.append(1.0)

    def _enc(t):
        if t is None:
            return [0.0] * 10
        v = [0.0] * 10
        v[int(t.task_type)] = 1.0
        v[5] = t.target_pos[0] / w_height
        v[6] = t.target_pos[1] / w_width
        v[7] = min(max(t.score, -5.0), 5.0) / 5.0
        v[8] = min(ghost.frame - t.created_frame, 200) / 200.0
        v[9] = 1.0
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
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        f.extend(_enc(ghost.cbba_agent.get_known_task_for(gid)))
    return np.asarray(f, dtype=np.float32)

def build_valid_mask(ghost, rows: int, cols: int, obs_resolution: float = 1.0) -> np.ndarray:
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

def actions_to_tasks(ghost, scores_map: np.ndarray, indices: list, frame: int, obs_resolution: float = 1.0) -> list:
    rows, cols = scores_map.shape
    tasks = []
    target = _pacman_target(ghost)
    for r, c in indices:
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        world_y = (float(r) + 0.5) / obs_resolution
        world_x = (float(c) + 0.5) / obs_resolution
        if not ghost.world.is_passable(world_x, world_y, radius=0.35):
            continue
        score = float(scores_map[r, c])
        is_power = any(abs(world_y - p[1]) < 0.5 and abs(world_x - p[0]) < 0.5 for p in ghost.known_power_pellets)
        if is_power:
            tt = TaskType.CONVERT
        elif target is not None and (abs(world_y - target[0]) + abs(world_x - target[1])) <= 3.0:
            tt = TaskType.HUNT
        else:
            tt = TaskType.DYNAMIC
        tasks.append(Task(task_type=tt, target_pos=(world_y, world_x), score=score, created_frame=frame, owner=ghost.gid))
    return tasks

def build_global_spatial(env, rows: int, cols: int, obs_resolution: float = 1.0) -> np.ndarray:
    out = np.zeros((11, rows, cols), dtype=np.float32)
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
    def _place_blob(channel, fy, fx):
        cr, cc = int(fy * obs_resolution), int(fx * obs_resolution)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    d2 = (fy * obs_resolution - (nr + 0.5))**2 + (fx * obs_resolution - (nc + 0.5))**2
                    channel[nr, nc] = max(channel[nr, nc], np.exp(-d2 / (2 * _BLOB_SIGMA**2)))
                    
    if not env.player.dead:
        _place_blob(out[3], env.player.y, env.player.x)
    for g in env.ghosts.values():
        if not g.dead:
            _place_blob(out[4 + g.gid], g.y, g.x)
    return out