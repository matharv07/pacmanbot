from __future__ import annotations
import math
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from net import MovementPredictor, PREDICTOR_IN_DIM, PREDICTOR_HIDDEN_DIM

WALL = 1

ALPHA_UNIFORM      = 0.15
ALPHA_MOMENTUM     = 0.15
MOMENTUM_DECAY     = 15
TAU_RECENCY        = 60
MIN_CONFIDENCE     = 0.02
LOS_CERTAINTY      = 1.0
LOST_SPREAD        = 0.60
COMPRESS_THRESHOLD = 0.0005

DANGER_SIGMA       = 6.0
STALENESS_DECAY    = 40.0
UNSEEN_GHOST_PRIOR = 0.30
PRIOR_UNIFORM_WT   = 1.0
MIN_SAFETY         = 1e-6
SAFETY_RECOMPUTE_EVERY = 8

HUNT_SIGMA         = 5.0
HUNT_CROWD_WEIGHT  = 0.4


def _compute_8ray_clearances(pacman_pos: tuple, known_walls: set = None, map_width: float = 28.0, 
                            map_height: float = 36.0, max_dist: float = 3.0, world = None) -> np.ndarray:
    """Computes normalized [0, 1] clearances along 8 cardinal/diagonal directions against discovered walls."""
    if hasattr(known_walls, 'width') or hasattr(known_walls, 'height'):
        world = known_walls
        known_walls = set()
    if world is not None:
        map_width = float(getattr(world, 'width', map_width))
        map_height = float(getattr(world, 'height', map_height))
    py, px = float(pacman_pos[0]), float(pacman_pos[1])
    angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], dtype=np.float32)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])  # (8, 2) [dx, dy]
    clearances = np.full(8, max_dist, dtype=np.float32)
    #vectorised boundary box clearance
    dx = dirs[:, 0]
    dy = dirs[:, 1]
    safe_dx = np.where(np.abs(dx) > 1e-5, dx, 1.0)
    safe_dy = np.where(np.abs(dy) > 1e-5, dy, 1.0)
    x_dist = np.where(dx > 1e-5, (map_width - px) / safe_dx, np.where(dx < -1e-5, -px / safe_dx, np.inf))
    y_dist = np.where(dy > 1e-5, (map_height - py) / safe_dy, np.where(dy < -1e-5, -py / safe_dy, np.inf))
    clearances = np.minimum(clearances, np.minimum(x_dist, y_dist))
    #vectorised wall intersection check across all discovered walls
    if known_walls is not None and len(known_walls) > 0:
        if isinstance(known_walls, np.ndarray):
            w_arr = known_walls
        else:
            w_arr = np.array(list(known_walls), dtype=np.float32)
        Vx = w_arr[:, 1] - px  # wx - px
        Vy = w_arr[:, 0] - py  # wy - py
        wall_radius = 0.5
        #broadcast across (8, N) rays and wall points
        proj = dirs[:, 0:1] * Vx[np.newaxis, :] + dirs[:, 1:2] * Vy[np.newaxis, :]
        perp = np.abs(-dirs[:, 1:2] * Vx[np.newaxis, :] + dirs[:, 0:1] * Vy[np.newaxis, :])
        hit_mask = (proj > 0) & (proj <= clearances[:, np.newaxis]) & (perp <= wall_radius)
        proj_hits = np.where(hit_mask, proj, np.inf)
        min_proj = np.min(proj_hits, axis=1)
        clearances = np.minimum(clearances, min_proj)
    return np.clip(clearances / max_dist, 0.0, 1.0).astype(np.float32)

def extract_movement_features(pacman_pos: tuple, current_vel: tuple, prev_vel: tuple, known_walls: set = None,
                              map_width: float = 28.0, map_height: float = 36.0, known_ghosts: list = None,
                              known_pellets = None, is_powered: bool = False, world = None) -> np.ndarray:
    """Constructs 19-dim feature vector for opponent intent and velocity prediction."""
    if hasattr(known_walls, 'width') or hasattr(known_walls, 'height'):
        world = known_walls
        known_walls = set()
    if world is not None:
        map_width = float(getattr(world, 'width', map_width))
        map_height = float(getattr(world, 'height', map_height))
    feats = np.zeros(PREDICTOR_IN_DIM, dtype=np.float32)
    py, px = float(pacman_pos[0]), float(pacman_pos[1])
    #kinematic history (4 dims)
    feats[0] = float(current_vel[0])
    feats[1] = float(current_vel[1])
    feats[2] = float(prev_vel[0])
    feats[3] = float(prev_vel[1])
    #8-ray geometric clearances (8 dims) computed against discovered walls only
    feats[4:12] = _compute_8ray_clearances(pacman_pos, known_walls=known_walls, map_width=map_width, map_height=map_height, max_dist=3.0)
    #threat context (3 dims: dy/d, dx/d, dist/10.0)
    if known_ghosts:
        g_arr = np.array(known_ghosts, dtype=np.float32)
        g_dy = g_arr[:, 0] - py
        g_dx = g_arr[:, 1] - px
        g_dists = np.hypot(g_dy, g_dx)
        min_idx = int(np.argmin(g_dists))
        d_g = float(g_dists[min_idx])
        if d_g > 1e-4:
            feats[12] = float(g_dy[min_idx]) / d_g
            feats[13] = float(g_dx[min_idx]) / d_g
        feats[14] = min(d_g, 10.0) / 10.0
    else:
        feats[14] = 1.0
    #objective context (3 dims: dy/d, dx/d, dist/10.0)
    if known_pellets:
        p_arr = np.array(list(known_pellets), dtype=np.float32)
        p_dy = p_arr[:, 1] - py
        p_dx = p_arr[:, 0] - px
        p_dists = np.hypot(p_dy, p_dx)
        min_p_idx = int(np.argmin(p_dists))
        d_p = float(p_dists[min_p_idx])
        if d_p > 1e-4:
            feats[15] = float(p_dy[min_p_idx]) / d_p
            feats[16] = float(p_dx[min_p_idx]) / d_p
        feats[17] = min(d_p, 10.0) / 10.0
    else:
        feats[17] = 1.0
    #power state (1 dim)
    feats[18] = 1.0 if is_powered else 0.0
    return feats

class BeliefMap:
    def __init__(self, gid: int, rows: int = 36, cols: int = 28, pacman_start: Optional[tuple] = None):
        self.gid = gid
        if hasattr(rows, 'height') and hasattr(rows, 'width'):
            self.rows = int(rows.height)
            self.cols = int(rows.width)
        else:
            self.rows = int(rows)
            self.cols = int(cols)
        self._initialised = False
        self.last_known_pos: Optional[tuple] = None
        self.last_known_dir: tuple = (0, 0)
        self.frames_since_sighting: int = 9999
        self._pacman_start: Optional[tuple] = pacman_start
        self._open_cells: list = []
        self._neighbours: dict = {}
        self._open_idx_map: dict = {}
        self.n_nodes = 0
        self._open_arr = np.empty((0, 2), dtype=np.float32)
        self._b_flat = np.zeros(0, dtype=np.float32)
        self._topology_dirty = True
        self._nbr_idx = np.empty((0, 0), dtype=np.int32)
        self._nbr_count = np.empty((0,), dtype=np.int32)
        self._safety = np.ones(0, dtype=np.float32)
        self._safety_grid = np.ones((self.rows, self.cols), dtype=np.float32)
        self._b_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        if pacman_start is not None: self._add_node(pacman_start)
        self._last_ghost_snapshot: dict = {}
        self._ghost_last_seen: dict[int, int] = {}
        self._disabled_wall_nodes: set = set()
        self._disabled_wall_idxs: list = []
        self._disabled_wall_arr: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._observed_wall_coords: set = set()
        self._grid_dirty: bool = True
        self._last_safety_frame: int = -999
        self._last_powered: bool = False
        self._payload_cache: dict | None = None
        self._payload_dirty: bool = True
        self.predictor = MovementPredictor(in_dim=PREDICTOR_IN_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM)
        self.predictor_hx = torch.zeros(1, PREDICTOR_HIDDEN_DIM)
        self.predicted_dir = (0.0, 0.0)
        self.predicted_vel = (0.0, 0.0)
        self._tree = None

    def _reset_topology(self):
        self._open_cells = []
        self._neighbours = {}
        self._open_idx_map = {}
        self.n_nodes = 0
        self._open_arr = np.empty((0, 2), dtype=np.float32)
        self._b_flat = np.zeros(0, dtype=np.float32)
        self._safety = np.ones(0, dtype=np.float32)
        self._disabled_wall_nodes = set()
        self._disabled_wall_idxs = []
        self._disabled_wall_arr = np.empty((0, 2), dtype=np.float32)
        self._observed_wall_coords = set()
        self._grid_dirty = True
        self._topology_dirty = True
        self._tree = None

    def _add_node(self, node: tuple) -> bool:
        if node in self._open_idx_map:
            return False
        idx = len(self._open_cells)
        self._open_cells.append(node)
        self._open_idx_map[node] = idx
        self._neighbours[node] = []
        self.n_nodes = len(self._open_cells)
        self._b_flat = np.append(self._b_flat, 0.0).astype(np.float32)
        self._safety = np.append(self._safety, 1.0).astype(np.float32)
        if len(self._open_cells) == 1:
            self._open_arr = np.array([node], dtype=np.float32)
        else:
            self._open_arr = np.vstack([self._open_arr, np.array([node], dtype=np.float32)])
        self._topology_dirty = True
        return True

    def _connect_nodes(self, n1: tuple, n2: tuple, dist: float):
        if n1 == n2:
            return
        if n1 not in self._neighbours:
            self._neighbours[n1] = []
        if n2 not in self._neighbours:
            self._neighbours[n2] = []
        if n2 not in self._neighbours[n1]:
            self._neighbours[n1].append(n2)
        if n1 not in self._neighbours[n2]:
            self._neighbours[n2].append(n1)
        self._topology_dirty = True

    def init_full_topology(self, world_prm_graph: dict = None):
        """Initializes dense UNKNOWN grid topology (0.8m resolution, blind hypothesis space)."""
        self._reset_topology()
        BELIEF_GRID_STEP = 0.8
        grid_nodes = []
        y = BELIEF_GRID_STEP / 2.0
        while y < self.rows:
            x = BELIEF_GRID_STEP / 2.0
            while x < self.cols:
                grid_nodes.append((round(float(y), 2), round(float(x), 2)))
                x += BELIEF_GRID_STEP
            y += BELIEF_GRID_STEP
        self._open_cells = grid_nodes
        self._open_idx_map = {node: i for i, node in enumerate(grid_nodes)}
        self._neighbours = {node: [] for node in grid_nodes}
        self.n_nodes = len(grid_nodes)
        self._open_arr = np.array(grid_nodes, dtype=np.float32)
        self._b_flat = np.zeros(self.n_nodes, dtype=np.float32)
        self._safety = np.ones(self.n_nodes, dtype=np.float32)

        CONNECT_RADIUS = BELIEF_GRID_STEP * 1.5
        if len(self._open_arr) > 0:
            self._tree = cKDTree(self._open_arr)
            pairs = self._tree.query_pairs(r=CONNECT_RADIUS)
            for i, j in pairs:
                ni, nj = self._open_cells[i], self._open_cells[j]
                self._neighbours[ni].append(nj)
                self._neighbours[nj].append(ni)
        self._walkable_mask = np.ones(self.n_nodes, dtype=bool)
        self._topology_dirty = True
        self._compute_topology()
        self._ensure_initialised()

    def _closest_node(self, pos: tuple):
        if not self._open_cells:
            return -1
        if self._tree is None:
            self._tree = cKDTree(self._open_arr)
        _, idx = self._tree.query([pos])
        return int(idx[0])

    def _closest_nodes_batch(self, pos_list: list):
        if not self._open_cells or not pos_list:
            return []
        if self._tree is None:
            self._tree = cKDTree(self._open_arr)
        pos_arr = np.array(pos_list, dtype=np.float32)
        _, indices = self._tree.query(pos_arr)
        return indices.tolist()

    def observe(self, pacman_pos: tuple, pacman_dir: tuple = (0, 0), current_vel: tuple = None,
                prev_vel: tuple = None, known_ghosts: list = None, known_pellets = None,
                known_walls: set = None, is_powered: bool = False):
        self._ensure_initialised()
        #sensor observations are trusted directly (no oracle passability check)
        total = float(self._b_flat.sum()) or 1.0
        self._b_flat *= (1.0 - LOS_CERTAINTY)
        idx = self._closest_node(pacman_pos)
        if idx >= 0:
            self._b_flat[idx] += total * LOS_CERTAINTY
        self.last_known_pos = pacman_pos
        self.last_known_dir = pacman_dir
        #neural movement predictor forward inference
        cur_v = current_vel if current_vel is not None else pacman_dir
        p_v = prev_vel if prev_vel is not None else (0.0, 0.0)
        walls_to_use = known_walls if known_walls is not None else (self._disabled_wall_arr if hasattr(self, '_disabled_wall_arr') and len(self._disabled_wall_arr) > 0 else self._disabled_wall_nodes)
        feats = extract_movement_features(pacman_pos=pacman_pos, current_vel=cur_v, prev_vel=p_v,
            known_walls=walls_to_use, map_width=float(self.cols), map_height=float(self.rows),
            known_ghosts=known_ghosts or [], known_pellets=known_pellets or [], is_powered=is_powered)
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0)
            base_v = torch.tensor([[cur_v[0], cur_v[1]]], dtype=torch.float32)
            pred_v, self.predictor_hx = self.predictor(x, self.predictor_hx, base_vel=base_v)
            pv = pred_v[0].detach().cpu().numpy()
            self.predicted_vel = (float(pv[0]), float(pv[1]))
            n = math.hypot(pv[0], pv[1])
            if n > 0.01: self.predicted_dir = (float(pv[0] / n), float(pv[1] / n))
            else: self.predicted_dir = pacman_dir
        self.frames_since_sighting = 0
        self._normalise()
        self._payload_dirty = True

    def observe_lost(self, last_pos: tuple):
        self._ensure_initialised()
        idx = self._closest_node(last_pos)
        if idx < 0:
            return
        outgoing = self._b_flat[idx] * LOST_SPREAD
        node = self._open_cells[idx]
        neighbours = self._neighbours.get(node, [])
        guide_dir = self.predicted_dir if self.predicted_dir != (0, 0) else self.last_known_dir
        if neighbours and guide_dir != (0, 0):
            r, c = node
            dr, dc = guide_dir
            weights = {}
            total_w = 0.0
            for nr, nc in neighbours:
                alignment = (nr - r) * dr + (nc - c) * dc
                w = max(0.0, alignment + 1.0)
                weights[(nr, nc)] = w
                total_w += w
            if total_w > 0:
                for nbr_node, w in weights.items():
                    nbr_idx = self._open_idx_map.get(nbr_node)
                    if nbr_idx is not None:
                        self._b_flat[nbr_idx] += outgoing * (w / total_w)
                self._b_flat[idx] -= outgoing
        self.last_known_pos = last_pos
        self.frames_since_sighting = 0
        self._normalise()
        self._payload_dirty = True

    def observe_walls_batch(self, wall_positions: list, ghost_positions: list = None, known_pellets = None, known_power = None, is_powered: bool = False):
        """Marks a batch of discovered walls as dead zones, severs edges incrementally,
        and redistributes trapped probability without cascading full topology rebuilds."""
        if not wall_positions:
            return
        if not hasattr(self, '_observed_wall_coords'):
            self._observed_wall_coords = set()
        unseen_walls = []
        for w in wall_positions:
            w_tup = (round(float(w[0]), 2), round(float(w[1]), 2))
            if w_tup not in self._observed_wall_coords:
                self._observed_wall_coords.add(w_tup)
                unseen_walls.append(w)
        if not unseen_walls:
            return
        self._ensure_initialised()
        if self.n_nodes == 0 or len(self._open_arr) == 0:
            return
        if self._tree is None:
            self._tree = cKDTree(self._open_arr)

        newly_disabled = []
        for wall_pos in unseen_walls:
            wy, wx = float(wall_pos[0]), float(wall_pos[1])
            close_idxs = self._tree.query_ball_point([wy, wx], r=0.6)
            if not close_idxs:
                dist, idx = self._tree.query([wy, wx])
                if dist <= 1.0:
                    close_idxs = [int(idx)]
            for idx in close_idxs:
                node = self._open_cells[idx]
                if node in self._disabled_wall_nodes:
                    continue
                self._disabled_wall_nodes.add(node)
                self._disabled_wall_idxs.append(idx)
                if hasattr(self, '_walkable_mask'):
                    self._walkable_mask[idx] = False
                newly_disabled.append((idx, node))
        if not newly_disabled:
            return
        new_coords = [self._open_cells[idx] for idx, _ in newly_disabled]
        if not hasattr(self, '_disabled_wall_arr') or self._disabled_wall_arr.size == 0:
            self._disabled_wall_arr = np.array(new_coords, dtype=np.float32)
        else:
            self._disabled_wall_arr = np.vstack([self._disabled_wall_arr, np.array(new_coords, dtype=np.float32)])
        dis_idxs = []
        for idx, node in newly_disabled:
            dis_idxs.append(idx)
            trapped_prob = float(self._b_flat[idx])
            self._b_flat[idx] = 0.0
            if trapped_prob > 1e-6:
                self._redistribute_wall_probability(wall_node=node, trapped_prob=trapped_prob, ghost_positions=ghost_positions, 
                                                    known_pellets=known_pellets, known_power=known_power, is_powered=is_powered)
            old_nbrs = list(self._neighbours.get(node, []))
            self._neighbours[node] = []
            for nbr in old_nbrs:
                if node in self._neighbours.get(nbr, []):
                    self._neighbours[nbr].remove(node)
        if hasattr(self, '_nbr_idx') and self._nbr_idx.size > 0:
            dis_arr = np.array(dis_idxs, dtype=np.int32)
            self._nbr_idx[dis_arr, :] = -1
            self._nbr_dist[dis_arr, :] = 0.0
            self._nbr_dr[dis_arr, :] = 0.0
            self._nbr_dc[dis_arr, :] = 0.0
            self._nbr_count[dis_arr] = 0
            sever_mask = np.isin(self._nbr_idx, dis_arr)
            if np.any(sever_mask):
                self._nbr_idx[sever_mask] = -1
                self._nbr_dist[sever_mask] = 0.0
                self._nbr_dr[sever_mask] = 0.0
                self._nbr_dc[sever_mask] = 0.0
                self._nbr_count = np.sum(self._nbr_idx >= 0, axis=1).astype(np.int32)
            self._W_dirty = True
            self._graph_version = getattr(self, '_graph_version', 0) + 1
        self._normalise()
        self._payload_dirty = True

    def observe_wall(self, wall_pos: tuple, ghost_positions: list = None, known_pellets = None, known_power = None, is_powered: bool = False):
        """Marks a discovered wall as a dead zone, severs edges, and redistributes trapped probability."""
        self.observe_walls_batch([wall_pos], ghost_positions=ghost_positions, known_pellets=known_pellets, known_power=known_power, is_powered=is_powered)

    def _redistribute_wall_probability(self, wall_node: tuple, trapped_prob: float, ghost_positions: list = None,
                                known_pellets = None, known_power = None, is_powered: bool = False, tau: float = 0.5):
        """Redistributes trapped probability across multiple detour paths (Path A > Path C > Path B)."""
        valid_nbrs = [nbr for nbr in self._neighbours.get(wall_node, []) if nbr not in self._disabled_wall_nodes]
        if not valid_nbrs:
            return
        if len(valid_nbrs) == 1:
            u_idx = self._open_idx_map.get(valid_nbrs[0])
            if u_idx is not None:
                self._b_flat[u_idx] += trapped_prob
            return
        scores = []
        p_dir = self.predicted_dir if self.predicted_dir != (0, 0) else self.last_known_dir
        v_norm = math.hypot(p_dir[0], p_dir[1])
        pellet_set = set(known_pellets) if known_pellets else set()
        power_set = set(known_power) if known_power else set()
        g_positions = ghost_positions or []
        for u in valid_nbrs:
            dy, dx = u[0] - wall_node[0], u[1] - wall_node[1]
            d = math.hypot(dy, dx) + 1e-5
            u_dir = (dy / d, dx / d)
            #momentum alignment
            s_mom = 0.0
            if v_norm > 0.01:
                s_mom = u_dir[0] * (p_dir[0] / v_norm) + u_dir[1] * (p_dir[1] / v_norm)
            #threat repulsion (or hunting if powered)
            if g_positions:
                min_g = min([math.hypot(u[0] - gy, u[1] - gx) for gy, gx in g_positions], default=10.0)
                norm_g = min(min_g, 10.0) / 10.0
                s_threat = norm_g if not is_powered else (1.0 - norm_g)
            else:
                s_threat = 0.5
            #regular pellet attraction
            s_pellet = 0.0
            if pellet_set:
                u_idx = self._open_idx_map.get(u)
                if u_idx is not None and hasattr(self, '_pellet_score') and len(self._pellet_score) > u_idx:
                    s_pellet = float(self._pellet_score[u_idx])
                elif (u[1], u[0]) in pellet_set or u in pellet_set:
                    s_pellet = 1.0
            #power pellet beacon
            s_power = 0.0
            if power_set:
                u_idx = self._open_idx_map.get(u)
                if u_idx is not None and hasattr(self, '_power_score') and len(self._power_score) > u_idx:
                    s_power = float(self._power_score[u_idx])
                else:
                    min_pow = min([math.hypot(u[0] - p[1], u[1] - p[0]) for p in power_set], default=20.0)
                    s_power = math.exp(-min_pow / 6.0)
                s_power = s_power * 0.2 if is_powered else s_power * 2.0
            q = (1.5 * s_mom) + (2.0 * s_threat) + (1.0 * s_pellet) + (2.5 * s_power)
            scores.append(q)
        scores = np.array(scores, dtype=np.float32)
        exp_scores = np.exp((scores - np.max(scores)) / max(tau, 0.05))
        weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        for u, w in zip(valid_nbrs, weights):
            u_idx = self._open_idx_map.get(u)
            if u_idx is not None:
                self._b_flat[u_idx] += trapped_prob * w

    def observe_clear(self, visible_idxs: set, impassable_nodes: list, pacman_pos=None):
        self._ensure_initialised()
        if self._open_arr.size == 0:
            return
        if not hasattr(self, '_explored_nodes'):
            self._explored_nodes = set()
        if visible_idxs:
            self._explored_nodes.update(visible_idxs)
            idxs = visible_idxs if isinstance(visible_idxs, np.ndarray) else np.fromiter(visible_idxs, dtype=np.int32, count=len(visible_idxs))
            if pacman_pos is not None:
                pac_idx = self._closest_node(pacman_pos)
                idxs = idxs[idxs != pac_idx]
            self._b_flat[idxs] = 0.0
        if impassable_nodes:
            self.observe_walls_batch(impassable_nodes)

    def _update_pellet_score(self, known_pellets, known_power, is_powered: bool = False):
        new_pellets = set(known_pellets) if known_pellets else set()
        new_power = set(known_power) if known_power else set()
        if hasattr(self, '_last_known_pellets') and hasattr(self, '_last_known_power'):
            if self._last_known_pellets == new_pellets and self._last_known_power == new_power:
                if getattr(self, '_last_powered_flag', False) == is_powered:
                    return
        self._last_known_pellets = new_pellets
        self._last_known_power = new_power
        self._last_powered_flag = is_powered
        if self.n_nodes == 0 or len(self._open_arr) == 0:
            return
        if new_pellets:
            p_arr = np.array([[float(p[1]), float(p[0])] for p in new_pellets], dtype=np.float32)
            p_tree = cKDTree(p_arr)
            dists, _ = p_tree.query(self._open_arr)
            dists[np.isinf(dists)] = 9999.0
            self._pellet_dists = dists.astype(np.float32)
            self._pellet_score = np.exp(-dists / 6.0).astype(np.float32)
        else:
            self._pellet_dists = np.full(self.n_nodes, 9999.0, dtype=np.float32)
            self._pellet_score = np.zeros(self.n_nodes, dtype=np.float32)
        if new_power:
            pow_arr = np.array([[float(p[1]), float(p[0])] for p in new_power], dtype=np.float32)
            pow_tree = cKDTree(pow_arr)
            pow_dists, _ = pow_tree.query(self._open_arr)
            pow_dists[np.isinf(pow_dists)] = 9999.0
            self._power_dists = pow_dists.astype(np.float32)
            self._power_score = np.exp(-pow_dists / 6.0).astype(np.float32)
        else:
            self._power_dists = np.full(self.n_nodes, 9999.0, dtype=np.float32)
            self._power_score = np.zeros(self.n_nodes, dtype=np.float32)

    def _compute_diffusion_weights(self):
        if getattr(self, '_W_dirty', True) or not hasattr(self, '_base_W') or self._base_W.shape != self._nbr_idx.shape:
            self._base_W = np.zeros_like(self._nbr_idx, dtype=np.float32)
            self._valid_mask = self._nbr_idx >= 0
            self._base_W[self._valid_mask] = 1.0 / np.maximum(self._nbr_dist[self._valid_mask], 0.1)
            self._W_dirty = False
        W = self._base_W.copy()
        valid_mask = self._valid_mask
        if hasattr(self, '_danger'):
            my_danger = self._danger[:, np.newaxis]
            nbr_danger = self._danger[self._nbr_idx]
            nbr_danger = np.where(valid_mask, nbr_danger, my_danger)
            delta_danger = np.clip(my_danger - nbr_danger, -3.0, 3.0)
            W[valid_mask] *= np.exp(delta_danger[valid_mask] * 0.8)
        p_dir = self.predicted_dir if self.predicted_dir != (0, 0) else self.last_known_dir
        if self.last_known_pos is not None and p_dir != (0, 0):
            dr, dc = p_dir
            alignment = self._nbr_dr * dr + self._nbr_dc * dc
            momentum_str = math.exp(-self.frames_since_sighting / MOMENTUM_DECAY)
            mom_factor = np.where(alignment > 0, 1.0 + momentum_str * (alignment * 3.0), np.maximum(0.05, 1.0 + momentum_str * (alignment * 0.95)))
            W[valid_mask] *= mom_factor[valid_mask]
        if hasattr(self, '_pellet_dists'):
            my_dists = self._pellet_dists[:, np.newaxis]
            nbr_dists = self._pellet_dists[self._nbr_idx]
            nbr_dists = np.where(valid_mask, nbr_dists, my_dists)
            delta_pellet = np.clip(my_dists - nbr_dists, -3.0, 3.0)
            W[valid_mask] *= np.exp(delta_pellet[valid_mask] * 0.4)
        if hasattr(self, '_power_dists') and getattr(self, '_last_known_power', None):
            is_pow = getattr(self, '_last_powered_flag', False)
            my_pow = self._power_dists[:, np.newaxis]
            nbr_pow = self._power_dists[self._nbr_idx]
            nbr_pow = np.where(valid_mask, nbr_pow, my_pow)
            delta_pow = np.clip(my_pow - nbr_pow, -3.0, 3.0)
            pow_mult = 0.8 if not is_pow else 0.1
            W[valid_mask] *= np.exp(delta_pow[valid_mask] * pow_mult)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.where(W_sum > 0, W_sum, 1.0)
        W /= W_sum
        return W

    def diffuse(self, ghost_pos: tuple, known_pellets: set = None, known_power: set = None,
                ghost_positions: list = None, is_powered: bool = False):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting + 1, 9999)
        if self.n_nodes > 0:
            self._update_pellet_score(known_pellets, known_power, is_powered=is_powered)
            roi_threshold = 1e-10
            self._b_flat[self._b_flat < roi_threshold] = 0.0
            W = self._compute_diffusion_weights()
            valid_mask = self._valid_mask
            receivers = self._nbr_idx[valid_mask]
            #3 speed-calibrated diffusion iterations matching continuous physical limits
            for _ in range(3):
                outflow = self._b_flat * (ALPHA_UNIFORM + ALPHA_MOMENTUM)
                shares = outflow[:, np.newaxis] * W
                self._b_flat -= outflow
                weights = shares[valid_mask]
                self._b_flat += np.bincount(receivers, weights=weights, minlength=self.n_nodes)
            self._b_flat[self._b_flat < roi_threshold] = 0.0
            self._b_flat = np.maximum(0.0, self._b_flat)
            if hasattr(self, '_walkable_mask'):
                self._b_flat[~self._walkable_mask] = 0.0
            self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells or self.n_nodes == 0:
            return
        delta_fss = self.frames_since_sighting - sender_fss
        if delta_fss > 0:
            confidence = min(0.35, 1.0 - math.exp(-delta_fss / 12.0))
        elif delta_fss == 0:
            confidence = 0.05
        else:
            confidence = 0.0
        if confidence <= 0.0:
            return
        n = self.n_nodes
        s_flat = np.zeros(n, dtype=np.float32)
        idxs, vals = [], []
        for pt, v in cells.items():
            idx = self._open_idx_map.get(pt)
            if idx is not None:
                idxs.append(idx)
                vals.append(v)
        if idxs:
            s_flat[idxs] = vals
        s_total = float(s_flat.sum())
        if s_total < 1e-9:
            return
        s_flat /= s_total
        self._b_flat = (1.0 - confidence) * self._b_flat + confidence * s_flat
        lkp = payload.get("lkp")
        if lkp is not None and sender_fss < self.frames_since_sighting:
            self.last_known_pos = tuple(lkp)
            self.last_known_dir = tuple(payload.get("lkd", (0, 0)))
            p_dir = payload.get("pred_dir", payload.get("p_dir", (0, 0)))
            if p_dir != (0, 0):
                self.predicted_dir = tuple(p_dir)
            hx = payload.get("hx")
            if hx is not None:
                self.predictor_hx = torch.tensor(hx, dtype=torch.float32).view(1, -1)
            self.frames_since_sighting = sender_fss
        self._normalise()

    def get_payload(self) -> dict:
        self._ensure_initialised()
        if not self._payload_dirty and self._payload_cache is not None:
            return self._payload_cache
        above = self._b_flat >= COMPRESS_THRESHOLD
        if not above.any():
            cells = {}
        else:
            idxs = np.nonzero(above)[0]
            vals = np.round(self._b_flat[idxs], 5)
            cells = {self._open_cells[int(i)]: float(v) for i, v in zip(idxs, vals)}
        hx_list = self.predictor_hx.squeeze().tolist()
        if not isinstance(hx_list, list):
            hx_list = [hx_list]
        self._payload_cache = {"cells": cells, "fss": self.frames_since_sighting, "lkp": self.last_known_pos, 
            "lkd": self.last_known_dir, "p_dir": self.predicted_dir, "pred_dir": self.predicted_dir, "hx": hx_list}
        self._payload_dirty = False
        return self._payload_cache

    def top_cells(self, n: int = 5) -> list[tuple]:
        self._ensure_initialised()
        if len(self._b_flat) == 0:
            return []
        if n == 1:
            best_idx = int(np.argmax(self._b_flat))
            return [self._open_cells[best_idx]]
        k = min(n, len(self._b_flat))
        top_idx = np.argpartition(self._b_flat, -k)[-k:]
        top_idx = top_idx[np.argsort(self._b_flat[top_idx])[::-1]]
        return [self._open_cells[i] for i in top_idx]

    def probability_at(self, pos: tuple) -> float:
        self._ensure_initialised()
        idx = self._open_idx_map.get(pos)
        if idx is not None:
            return float(self._b_flat[idx])
        idx = self._closest_node(pos)
        if idx >= 0:
            return float(self._b_flat[idx])
        r, c = int(round(pos[0])), int(round(pos[1]))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if getattr(self, '_grid_dirty', True):
                self._sync_flat_to_grid()
                self._grid_dirty = False
            return float(self._b_grid[r, c])
        return 0.0

    def as_flat_list(self) -> list[float]:
        self._ensure_initialised()
        if getattr(self, '_grid_dirty', True):
            self._sync_flat_to_grid()
            self._grid_dirty = False
        return self._b_grid.flatten().tolist()

    def update_safety_map(self, known_agents: dict, current_frame: int, powered: bool = False, hunt_mode: str = "blend", pacman_pos: Optional[tuple] = None):
        new_snapshot = {gid: (int(pos[0]), int(pos[1])) for gid, pos in known_agents.items() if pos != "UNKNOWN"}
        positions_changed = (new_snapshot != self._last_ghost_snapshot)
        mode_changed = (powered != self._last_powered)
        due = (current_frame - self._last_safety_frame >= SAFETY_RECOMPUTE_EVERY)
        if not (positions_changed or mode_changed or due):
            return
        self._last_ghost_snapshot = new_snapshot
        self._last_powered = powered
        self._last_safety_frame = current_frame
        for gid, pos in known_agents.items():
            if pos != "UNKNOWN":
                self._ghost_last_seen[gid] = current_frame
        n_open = self.n_nodes
        if n_open == 0 or len(self._open_arr) == 0:
            return
        scores = np.zeros(self.n_nodes, dtype=np.float32)
        if powered:             #when Pacman is powered, danger emanates from Pacman
            p_pos = pacman_pos or self.last_known_pos
            if p_pos is None and len(self._b_flat) > 0:
                top = self.top_cells(n=1)
                if top:
                    p_pos = top[0]
            if p_pos is not None:
                pr, pc = float(p_pos[0]), float(p_pos[1])
                sigma = DANGER_SIGMA
                cutoff = float(3.0 * sigma)
                dists = np.hypot(self._open_arr[:, 0] - pr, self._open_arr[:, 1] - pc)
                mask = dists <= cutoff
                scores[mask] += 3.0 * np.exp(-dists[mask] / sigma)
        else:
            known_positions = []
            for gid, pos in known_agents.items():
                if pos == "UNKNOWN":
                    continue
                gr, gc = pos
                age = current_frame - self._ghost_last_seen.get(gid, current_frame)
                weight = math.exp(-age / STALENESS_DECAY)
                known_positions.append((float(gr), float(gc), weight))
            sigma = DANGER_SIGMA
            cutoff_steps = float(3.0 * sigma)
            for gr, gc, weight in known_positions:
                dists = np.hypot(self._open_arr[:, 0] - gr, self._open_arr[:, 1] - gc)
                mask = dists <= cutoff_steps
                scores[mask] += weight * np.exp(-dists[mask] / sigma)
        self._danger = scores
        self._safety = np.exp(-scores / 3.0)
        self._W_dirty = True

    def _compute_topology(self):
        n = len(self._open_cells)
        if n == 0:
            self._nbr_idx = np.full((0, 0), -1, dtype=np.int32)
            self._nbr_count = np.zeros(0, dtype=np.int32)
            return
        nbr_idx_list = []
        for cell in self._open_cells:
            nbrs = self._neighbours.get(cell, [])
            n_idx = [self._open_idx_map.get(nb) for nb in nbrs]
            nbr_idx_list.append([i for i in n_idx if i is not None])
        nbr_count_list = [len(idxs) for idxs in nbr_idx_list]
        max_nbrs = max(nbr_count_list) if nbr_count_list else 0
        self._nbr_idx = np.full((n, max(max_nbrs, 1)), -1, dtype=np.int32)
        for i, idxs in enumerate(nbr_idx_list):
            if idxs:
                self._nbr_idx[i, :len(idxs)] = idxs
        self._nbr_count = np.array(nbr_count_list, dtype=np.int32)
        self._nbr_dist = np.zeros_like(self._nbr_idx, dtype=np.float32)
        self._nbr_dr = np.zeros_like(self._nbr_idx, dtype=np.float32)
        self._nbr_dc = np.zeros_like(self._nbr_idx, dtype=np.float32)
        r_arr = self._open_arr[:, 0]
        c_arr = self._open_arr[:, 1]
        for k in range(self._nbr_idx.shape[1]):
            nbrs = self._nbr_idx[:, k]
            valid = nbrs >= 0
            if not np.any(valid):
                continue
            nr = r_arr[nbrs[valid]]
            nc = c_arr[nbrs[valid]]
            d_r = nr - r_arr[valid]
            d_c = nc - c_arr[valid]
            dist = np.hypot(d_r, d_c)
            safe_dist = np.where(dist > 0, dist, 1.0)
            self._nbr_dr[valid, k] = np.where(dist > 0, d_r / safe_dist, 0.0)
            self._nbr_dc[valid, k] = np.where(dist > 0, d_c / safe_dist, 0.0)
            self._nbr_dist[valid, k] = dist

        self._graph_version = getattr(self, '_graph_version', 0) + 1
        self._W_dirty = True

    def _ensure_initialised(self):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
        if self._initialised:
            return
        if self._pacman_start is not None:
            idx = self._open_idx_map.get(self._pacman_start)
            if idx is None:
                idx = self._closest_node(self._pacman_start)
            if idx >= 0:
                self._b_flat.fill(0.0)
                self._b_flat[idx] = 1.0
        else:
            n = self.n_nodes
            if n:
                self._b_flat[:] = 1.0 / n
        self._initialised = True

    def _sync_flat_to_grid(self):
        if self.n_nodes > 0:
            self._b_grid.fill(0.0)
            rs = np.clip(np.round(self._open_arr[:, 0]).astype(np.int32), 0, self.rows - 1)
            cs = np.clip(np.round(self._open_arr[:, 1]).astype(np.int32), 0, self.cols - 1)
            np.add.at(self._b_grid, (rs, cs), self._b_flat[:self.n_nodes])

    def _normalise(self):
        if self.n_nodes == 0:
            return
        if hasattr(self, '_walkable_mask'):
            self._b_flat[~self._walkable_mask] = 0.0
        elif hasattr(self, '_disabled_wall_idxs') and self._disabled_wall_idxs:
            self._b_flat[self._disabled_wall_idxs] = 0.0
        if np.isnan(self._b_flat).any():
            self._b_flat = np.nan_to_num(self._b_flat, nan=0.0)
        total = float(self._b_flat.sum())
        if total < 1e-12:
            self._b_flat[:] = 0.0
            if hasattr(self, '_walkable_mask'):
                valid_count = self._walkable_mask.sum()
                if valid_count > 0:
                    self._b_flat[self._walkable_mask] = 1.0 / valid_count
            else:
                self._b_flat[:] = 1.0 / self.n_nodes
        else:
            self._b_flat /= total
        self._grid_dirty = True
        self._payload_dirty = True