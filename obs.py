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
SPATIAL_CH   = 16       #number of spatial channels (see channel map below)
GLOBAL_SPATIAL_CH = 11   #number of channels in the omniscient global state
VEC_DIM      = 104
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


def build_spatial(ghost, recent_noms: np.ndarray, rows: int = None, cols: int = None) -> np.ndarray:
    """
    Returns (SPATIAL_CH, rows, cols) float32 tensor.
    """
    if rows is None or cols is None:
        rows, cols = ghost.personal_map.shape
    out = np.zeros((SPATIAL_CH, rows, cols), dtype=np.float32)
    p = ghost.personal_map
    #channels 0–3: one-hot personal map
    out[0] = (p == WALL)
    out[1] = (p == PELLET)
    out[2] = (p == POWER)
    out[3] = (p == UNKNOWN)
    
    #channel 4: belief map — project flat probability onto grid via PRM nodes
    bm = ghost.belief_map
    if hasattr(bm, '_b_flat') and bm._initialised and bm._open_cells:
        for i, (r, c) in enumerate(bm._open_cells):
            ri, ci = int(r), int(c)
            if 0 <= ri < rows and 0 <= ci < cols and i < len(bm._b_flat):
                out[4, ri, ci] = max(out[4, ri, ci], bm._b_flat[i])
    
    #channel 5: safety map — project flat safety onto grid via PRM nodes
    if hasattr(bm, '_safety') and bm._open_cells:
        for i, (r, c) in enumerate(bm._open_cells):
            ri, ci = int(r), int(c)
            if 0 <= ri < rows and 0 <= ci < cols and i < len(bm._safety):
                out[5, ri, ci] = max(out[5, ri, ci], bm._safety[i])
    
    #gaussian blob position encoding — preserves sub-cell precision for continuous coordinates
    _BLOB_SIGMA = 0.6
    def _place_blob(channel, fy, fx):
        """Place a small Gaussian blob at continuous position (fy, fx) on the given channel."""
        cr, cc = int(fy), int(fx)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    d2 = (fy - (nr + 0.5))**2 + (fx - (nc + 0.5))**2
                    channel[nr, nc] = max(channel[nr, nc], np.exp(-d2 / (2 * _BLOB_SIGMA**2)))
    _place_blob(out[6], ghost.y, ghost.x)
    target = _pacman_target(ghost)
    if target is not None:
        tr, tc = target
        if 0 <= tr < rows and 0 <= tc < cols:
            _place_blob(out[7], float(tr) + 0.5, float(tc) + 0.5)
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        ch = 8 + (gid if gid < ghost.gid else gid - 1)
        pos = ghost.known_agents.get(gid)
        if pos is not None and pos != "UNKNOWN":
            r, c = pos
            if 0 <= r < rows and 0 <= c < cols:
                _place_blob(out[ch], float(r), float(c))
    
    #channel 14: staleness from prm_last_seen dict
    stale_ch = np.zeros((rows, cols), dtype=np.float32)
    for node, last_seen in ghost.prm_last_seen.items():
        ri, ci = int(node[0]), int(node[1])
        if 0 <= ri < rows and 0 <= ci < cols:
            if last_seen < 0:
                stale_ch[ri, ci] = 1.0  # never seen = max staleness
            else:
                stale_ch[ri, ci] = min(ghost.frame - last_seen, 200) / 200.0
    out[14] = stale_ch
    out[15] = recent_noms[:rows, :cols]
    return out

def build_vector(ghost) -> np.ndarray:
    rows = len(ghost.grid)
    cols = len(ghost.grid[0])
    f = []
    f.extend([ghost.y / rows, ghost.x / cols])
    #sub-cell fractional position — gives exact continuous position info beyond spatial blob
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

    def _enc(t):
        if t is None:
            return [0.0] * 10
        v = [0.0] * 10
        v[int(t.task_type)] = 1.0
        v[5] = t.target_pos[0] / rows
        v[6] = t.target_pos[1] / cols
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

def build_valid_mask(ghost, rows: int = None, cols: int = None) -> np.ndarray:
    if rows is None or cols is None:
        rows, cols = ghost.personal_map.shape
    p = ghost.personal_map
    mask = (p != WALL)
    return mask

def actions_to_tasks(ghost, scores_map: np.ndarray, indices: list, frame: int) -> list:
    rows, cols = ghost.personal_map.shape
    tasks = []
    target = _pacman_target(ghost)
    for r, c in indices:
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        cell = ghost.personal_map[int(r), int(c)]
        if cell == WALL:
            continue
        score = float(scores_map[r, c])
        if cell == POWER:
            tt = TaskType.CONVERT
        elif target is not None and (abs(r - target[0]) + abs(c - target[1])) <= 3:
            tt = TaskType.HUNT
        else:
            tt = TaskType.DYNAMIC
        tasks.append(Task(task_type=tt, target_pos=(r, c), score=score, created_frame=frame, owner=ghost.gid))
    return tasks

def build_global_spatial(env, rows, cols) -> np.ndarray:
    #build the omniscient global state for the critic
    _BLOB_SIGMA = 0.6
    def _place_blob(channel, fy, fx):
        cr, cc = int(fy), int(fx)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    d2 = (fy - (nr + 0.5))**2 + (fx - (nc + 0.5))**2
                    channel[nr, nc] = max(channel[nr, nc], np.exp(-d2 / (2 * _BLOB_SIGMA**2)))
    out = np.zeros((11, rows, cols), dtype=np.float32)
    out[0] = (env.grid == 1)
    out[1] = (env.grid == 2)
    out[2] = (env.grid == 3)
    if not env.player.dead:
        _place_blob(out[3], env.player.y, env.player.x)
    for g in env.ghosts.values():
        if not g.dead:
            _place_blob(out[4 + g.gid], g.y, g.x)
    return out