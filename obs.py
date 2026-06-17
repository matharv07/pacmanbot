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

# Pad all grids to this size so the CNN dimensions never change between
# curriculum stages.  48×48 comfortably fits the 41×33 max grid.
MAX_H = 33
MAX_W = 41
MAX_GHOSTS   = 7
SPATIAL_CH   = 16   # number of spatial channels (see channel map below)
GLOBAL_SPATIAL_CH = 5 # number of channels in the omniscient global state
VEC_DIM      = 102
CRITIC_VEC_DIM = MAX_GHOSTS * VEC_DIM + MAX_GHOSTS

# ── Channel map ──────────────────────────────────────────────────────
#  0  is_wall           4  belief_map       8–13  other ghosts (6 ch)
#  1  is_pellet         5  safety_map       14    staleness
#  2  is_power          6  own_position     15    recent_nominations
#  3  is_unknown        7  pacman_position
# ─────────────────────────────────────────────────────────────────────

def _pacman_target(ghost):
    """Best current estimate of Pacman's position."""
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

    Parameters
    ----------
    ghost : Ghost
        The ghost whose observation we are building.
    recent_noms : ndarray
        Decayed map of this ghost's recent waypoint nominations.
    rows : int, optional
    cols : int, optional
    """
    if rows is None or cols is None:
        rows, cols = ghost.personal_map.shape
    out = np.zeros((SPATIAL_CH, rows, cols), dtype=np.float32)

    p = ghost.personal_map

    # Channels 0–3: one-hot personal map
    out[0] = (p == WALL)
    out[1] = (p == PELLET)
    out[2] = (p == POWER)
    out[3] = (p == UNKNOWN)

    # 4: belief map
    if hasattr(ghost.belief_map, '_b'):
        out[4] = ghost.belief_map._b[:rows, :cols]

    # 5: safety map
    if hasattr(ghost.belief_map, '_safety'):
        out[5] = ghost.belief_map._safety[:rows, :cols]

    # 6: own position
    out[6, ghost.row, ghost.col] = 1.0

    # 7: best Pacman estimate
    target = _pacman_target(ghost)
    if target is not None:
        tr, tc = target
        if 0 <= tr < rows and 0 <= tc < cols:
            out[7, tr, tc] = 1.0

    # 8–13: other ghosts (one channel per slot, skip own gid)
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        ch = 8 + (gid if gid < ghost.gid else gid - 1)
        pos = ghost.known_agents.get(gid)
        if pos is not None and pos != "UNKNOWN":
            r, c = pos
            if 0 <= r < rows and 0 <= c < cols:
                out[ch, r, c] = 1.0

    # 14: staleness (frames since last observation, capped & normalised)
    ls = np.asarray(ghost.last_seen, dtype=np.float32)
    stale = np.clip(ghost.frame - ls, 0, 200) / 200.0
    out[14] = stale

    # 15: recent nominations
    out[15] = recent_noms[:rows, :cols]

    return out


def build_vector(ghost) -> np.ndarray:
    """
    Returns flat float32 vector (~100 features).

    Parameters
    ----------
    ghost : Ghost
    """
    rows = len(ghost.grid)
    cols = len(ghost.grid[0])
    f = []

    # ── scalar features ──────────────────────────────────────────────
    f.extend([ghost.row / rows, ghost.col / cols])                  # own pos
    timer = getattr(ghost, 'pacman_power_timer', 0)
    f.append(timer / 40.0 if getattr(ghost, 'pacman_powered', False) else 0.0)  # powered
    since = ghost.frame - ghost.pacman_last_seen if ghost.pacman_last_seen >= 0 else 200
    f.append(min(since, 200) / 200.0)                               # time since sighting

    for gid in range(MAX_GHOSTS):                                   # 6 dead flags
        if gid == ghost.gid:
            continue
        st = ghost.known_agents.get(gid)
        f.append(1.0 if st == "UNKNOWN" or st is None else 0.0)

    f.append(min(ghost.frame, 2000) / 2000.0)                       # normalised frame
    f.append(1.0 if getattr(ghost, 'in_fallback_mode', False) else 0.0) # fallback flag

    # ── own CBBA bundle (up to 3 tasks, 10 features each) ───────────
    def _enc(t):
        if t is None:
            return [0.0] * 10
        v = [0.0] * 10
        v[int(t.task_type)] = 1.0       # one-hot type (0-4)
        v[5] = t.target_pos[0] / rows
        v[6] = t.target_pos[1] / cols
        v[7] = min(max(t.score, -5.0), 5.0) / 5.0
        v[8] = min(ghost.frame - t.created_frame, 200) / 200.0
        v[9] = 1.0                       # valid flag
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

    # ── other ghosts' active tasks (up to 6 × 10 = 60) ─────────────
    for gid in range(MAX_GHOSTS):
        if gid == ghost.gid:
            continue
        f.extend(_enc(ghost.cbba_agent.get_known_task_for(gid)))

    return np.asarray(f, dtype=np.float32)


def build_valid_mask(ghost, rows: int = None, cols: int = None) -> np.ndarray:
    """Returns (rows, cols) bool mask — True where the actor may nominate."""
    if rows is None or cols is None:
        rows, cols = ghost.personal_map.shape
    p = ghost.personal_map
    mask = (p != WALL)
    return mask


def actions_to_tasks(ghost, scores_map: np.ndarray,
                     indices: list, frame: int) -> list:
    """
    Convert the RL actor's sampled cell indices + independent sigmoid scores
    into a list of CBBA-compatible Task objects.

    Parameters
    ----------
    ghost : Ghost
    scores_map : ndarray (MAX_H, MAX_W)  — sigmoid scores from the actor
    indices : list of (row, col) tuples   — K nominated cells
    frame : int
    """
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

        # infer task type from cell content & proximity to Pacman
        if cell == POWER:
            tt = TaskType.CONVERT
        elif target is not None and (abs(r - target[0]) + abs(c - target[1])) <= 3:
            tt = TaskType.HUNT
        else:
            tt = TaskType.DYNAMIC

        tasks.append(Task(task_type=tt, target_pos=(r, c),
                          score=score, created_frame=frame, owner=ghost.gid))
    return tasks

def build_global_spatial(env, rows, cols) -> np.ndarray:
    """Builds the omniscient global state for the Critic."""
    out = np.zeros((5, rows, cols), dtype=np.float32)
    # 0: Walls, 1: Pellets, 2: Power
    out[0] = (env.grid == 1)
    out[1] = (env.grid == 2)
    out[2] = (env.grid == 3)
    # 3: True Pacman
    if not env.player.dead:
        out[3, env.player.row, env.player.col] = 1.0
    # 4: True Ghosts
    for g in env.ghosts.values():
        if not g.dead:
            out[4, g.row, g.col] = 1.0
    return out
