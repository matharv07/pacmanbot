from __future__ import annotations
import math
from typing import Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.spatial import cKDTree

WALL = 1

ALPHA_UNIFORM      = 0.20
ALPHA_MOMENTUM     = 0.25
MOMENTUM_DECAY     = 50
TAU_RECENCY        = 60
MIN_CONFIDENCE     = 0.02
LOS_CERTAINTY      = 0.99
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

class BeliefMap:
    def __init__(self, gid: int, world, pacman_start: Optional[tuple] = None):
        self.gid = gid
        self.world = world
        self.rows = world.height
        self.cols = world.width
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
        self._graph = sp.csr_matrix((0, 0), dtype=np.float32)

        self._safety = np.ones(0, dtype=np.float32)
        self._safety_grid = np.ones((self.rows, self.cols), dtype=np.float32)
        self._b_grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        if pacman_start is not None:
            self._add_node(pacman_start)
            
        self._last_ghost_snapshot: dict = {}
        self._ghost_last_seen: dict[int, int] = {}
        self._disabled_wall_nodes: set = set()
        self._last_safety_frame: int = -999
        self._last_powered: bool = False
        self._payload_cache: dict | None = None
        self._payload_dirty: bool = True

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

    def init_full_topology(self, world_prm_graph: dict):
        BELIEF_GRID_STEP = 0.8
        grid_nodes = []
        y = BELIEF_GRID_STEP / 2.0
        while y < self.rows:
            x = BELIEF_GRID_STEP / 2.0
            while x < self.cols:
                node = (float(y), float(x))
                if node not in self._open_idx_map:
                    self._open_idx_map[node] = len(self._open_cells)
                    self._open_cells.append(node)
                    self.n_nodes += 1
                    grid_nodes.append(node)
                x += BELIEF_GRID_STEP
            y += BELIEF_GRID_STEP

        if grid_nodes:
            new_nodes_arr = np.array(grid_nodes, dtype=np.float32)
            if self._b_flat is None or len(self._b_flat) == 0:
                self._b_flat = np.zeros(len(grid_nodes), dtype=np.float32)
                self._b_flat[0] = 1.0
                self._safety = np.ones(len(grid_nodes), dtype=np.float32)
                self._open_arr = new_nodes_arr
            else:
                self._b_flat = np.concatenate([self._b_flat, np.zeros(len(grid_nodes), dtype=np.float32)])
                self._safety = np.concatenate([self._safety, np.ones(len(grid_nodes), dtype=np.float32)])
                self._open_arr = np.vstack([self._open_arr, new_nodes_arr])

        CONNECT_RADIUS = BELIEF_GRID_STEP * 1.5
        if len(self._open_arr) > 0:
            tree = cKDTree(self._open_arr)
            pairs = tree.query_pairs(r=CONNECT_RADIUS)
            for node in self._open_cells:
                self._neighbours[node] = []
            for i, j in pairs:
                ni, nj = self._open_cells[i], self._open_cells[j]
                self._neighbours[ni].append(nj)
                self._neighbours[nj].append(ni)

        self._topology_dirty = True

    def _closest_node(self, pos: tuple):
        if not self._open_cells: return -1
        dist = np.sum((self._open_arr - np.array(pos, dtype=np.float32))**2, axis=1)
        return int(np.argmin(dist))

    def observe(self, pacman_pos: tuple, pacman_dir: tuple = (0, 0)):
        self._ensure_initialised()
        if not self.world.is_passable(pacman_pos[1], pacman_pos[0], radius=0.4):
            return
        total = float(self._b_flat.sum()) or 1.0
        self._b_flat *= (1.0 - LOS_CERTAINTY)
        idx = self._closest_node(pacman_pos)
        if idx >= 0:
            self._b_flat[idx] += total * LOS_CERTAINTY
        self.last_known_pos = pacman_pos
        self.last_known_dir = pacman_dir
        self.frames_since_sighting = 0
        self._normalise()
        self._payload_dirty = True

    def observe_lost(self, last_pos: tuple):
        self._ensure_initialised()
        idx = self._closest_node(last_pos)
        if idx < 0: return
        
        outgoing = self._b_flat[idx] * LOST_SPREAD
        node = self._open_cells[idx]
        neighbours = self._neighbours.get(node, [])
        if neighbours and self.last_known_dir != (0, 0):
            r, c = node
            dr, dc = self.last_known_dir
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

    def observe_clear(self, visible_idxs: set, impassable_nodes: list, pacman_pos=None):
        self._ensure_initialised()
        if self._open_arr.size == 0:
            return
            
        if visible_idxs:
            idxs = np.array(list(visible_idxs), dtype=np.int32)
            if pacman_pos is not None:
                pac_idx = self._closest_node(pacman_pos)
                idxs = idxs[idxs != pac_idx]
            self._b_flat[idxs] = 0.0
            
        disabled_any = False
        for node in impassable_nodes:
            if node not in self._disabled_wall_nodes:
                self._disabled_wall_nodes.add(node)
                disabled_any = True
                old_nbrs = self._neighbours.get(node, [])
                self._neighbours[node] = []
                for nbr in old_nbrs:
                    if node in self._neighbours[nbr]:
                        self._neighbours[nbr].remove(node)
                        
        if disabled_any:
            self._topology_dirty = True
            
        self._normalise()

    def diffuse(self, ghost_pos: tuple, known_pellets: set = None, known_power: set = None):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting + 1, 9999)
        if self.n_nodes > 0:
            self._predictive_diffuse(known_pellets, known_power)
            self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells or self.n_nodes == 0:
            return
        confidence = max(MIN_CONFIDENCE, math.exp(-sender_fss / TAU_RECENCY))
        n = self.n_nodes
        s_flat = np.full(n, COMPRESS_THRESHOLD / 2.0, dtype=np.float32)
        
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
        log_prior  = np.log(np.maximum(self._b_flat, 1e-12))
        log_sender = np.log(np.maximum(s_flat, 1e-12))
        self._b_flat = np.exp((1.0 - confidence) * log_prior + confidence * log_sender)
        lkp = payload.get("lkp")
        if lkp is not None and sender_fss < self.frames_since_sighting:
            self.last_known_pos        = tuple(lkp)
            self.last_known_dir        = tuple(payload.get("lkd", (0, 0)))
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
            cells = { self._open_cells[int(i)]: float(v) for i, v in zip(idxs, vals) }
        self._payload_cache = {"cells": cells, "fss": self.frames_since_sighting, "lkp": self.last_known_pos, "lkd": self.last_known_dir}
        self._payload_dirty = False
        return self._payload_cache

    def top_cells(self, n: int = 5) -> list[tuple]:
        self._ensure_initialised()
        if len(self._b_flat) == 0:
            return []
        k = min(n, len(self._b_flat))
        top_idx = np.argpartition(self._b_flat, -k)[-k:]
        top_idx = top_idx[np.argsort(self._b_flat[top_idx])[::-1]]
        return [self._open_cells[i] for i in top_idx]

    def probability_at(self, pos: tuple) -> float:
        self._ensure_initialised()
        r, c = int(round(pos[0])), int(round(pos[1]))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return float(self._b_grid[r, c])
        return 0.0

    def as_flat_list(self) -> list[float]:
        self._ensure_initialised()
        return self._b_grid.flatten().tolist()

    def update_safety_map(self, known_agents: dict, current_frame: int, powered: bool = False, hunt_mode: str = "blend"):
        new_snapshot = {gid: pos for gid, pos in known_agents.items() if pos != "UNKNOWN"}
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
        if n_open == 0:
            return
        known_positions = []
        n_unknown = 0
        for gid, pos in known_agents.items():
            if pos == "UNKNOWN":
                n_unknown += 1
                continue
            gr, gc = pos
            age = current_frame - self._ghost_last_seen.get(gid, current_frame)
            weight = math.exp(-age / STALENESS_DECAY)
            known_positions.append((gr, gc, weight))
            
        sigma = HUNT_SIGMA if powered else DANGER_SIGMA
        cutoff_steps = int(3.0 * sigma)
        scores = np.zeros(self.n_nodes, dtype=np.float32)
        proximal = np.zeros(self.n_nodes, dtype=np.float32)
        starts, weights = [], []
        for gr, gc, weight in known_positions:
            idx = self._closest_node((gr, gc))
            if idx >= 0:
                starts.append(idx)
                weights.append(weight)
        
        if starts and self._graph.shape[0] > 0:
            dist_matrix = csgraph.dijkstra(self._graph, directed=False, indices=starts, unweighted=False, limit=cutoff_steps)
            if dist_matrix.ndim == 1:
                dist_matrix = dist_matrix[np.newaxis, :]
            for i, weight in enumerate(weights):
                dist_flat = dist_matrix[i]
                mask_flat = dist_flat <= cutoff_steps
                contrib_vals = weight * np.exp(-(dist_flat[mask_flat] ** 2) / (2.0 * sigma ** 2))
                scores[mask_flat] += contrib_vals
                if powered:
                    proximal[mask_flat] = np.maximum(proximal[mask_flat], contrib_vals)

        if not powered:
            prior = PRIOR_UNIFORM_WT / max(n_open, 1)
            flat_unknown = n_unknown * (UNSEEN_GHOST_PRIOR / max(n_open, 1))
            scores += prior + flat_unknown
            max_score = np.max(scores)
            if max_score < MIN_SAFETY:
                max_score = MIN_SAFETY
            self._safety[:n_open] = 1.0 - (scores / max_score)
        else:
            max_crowd = np.max(scores)
            if max_crowd < MIN_SAFETY: max_crowd = MIN_SAFETY
            max_prox = np.max(proximal)
            if max_prox < MIN_SAFETY: max_prox = MIN_SAFETY
            c_norm = scores / max_crowd
            p_norm = proximal / max_prox
            if hunt_mode == "proximal":
                self._safety[:n_open] = p_norm
            elif hunt_mode == "crowd":
                self._safety[:n_open] = c_norm
            else:
                self._safety[:n_open] = ((1.0 - HUNT_CROWD_WEIGHT) * p_norm + HUNT_CROWD_WEIGHT * c_norm)
                
        self._safety_grid.fill(0.0)
        rs = np.clip(np.round(self._open_arr[:, 0]).astype(np.int32), 0, self.rows - 1)
        cs = np.clip(np.round(self._open_arr[:, 1]).astype(np.int32), 0, self.cols - 1)
        self._safety_grid[rs, cs] = self._safety[:n_open]
        
        for r, c in self._disabled_wall_nodes:
            ri, ci = int(round(r)), int(round(c))
            if 0 <= ri < self.rows and 0 <= ci < self.cols:
                self._safety_grid[ri, ci] = 0.0

    def safety_at(self, pos: tuple) -> float:
        r, c = int(round(pos[0])), int(round(pos[1]))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return float(self._safety_grid[r, c])
        return 0.0

    def safest_cells(self, n: int = 5) -> list[tuple]:
        if self.n_nodes == 0:
            return []
        k = min(n, self.n_nodes)
        top_idx = np.argpartition(self._safety[:self.n_nodes], -k)[-k:]
        top_idx = top_idx[np.argsort(self._safety[top_idx])[::-1]]
        return [self._open_cells[i] for i in top_idx]

    def safest_neighbour(self, pacman_pos: tuple) -> Optional[tuple]:
        candidates = self._neighbours.get(pacman_pos, [])
        if not candidates:
            return None
        best = None
        best_score = -1.0
        for node in candidates:
            idx = self._open_idx_map.get(node)
            if idx is not None and idx < len(self._safety):
                s = self._safety[idx]
                if s > best_score:
                    best_score = s
                    best = node
        return best

    def safety_as_flat_list(self) -> list[float]:
        return self._safety_grid.flatten().tolist()

    def safety_payload(self) -> dict:
        cells = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = float(self._safety_grid[r, c])
                if s < 0.95:
                    cells[(r, c)] = round(s, 4)
        return cells

    def _compute_topology(self):
        n = len(self._open_cells)
        if n == 0:
            self._nbr_idx = np.full((0, 0), -1, dtype=np.int32)
            self._nbr_count = np.zeros(0, dtype=np.int32)
            self._graph = sp.csr_matrix((0, 0), dtype=np.float32)
            return
        nbr_idx_list = []
        nbr_count_list = []
        for cell in self._open_cells:
            nbrs = self._neighbours.get(cell, [])
            n_idx = [self._open_idx_map.get(nb) for nb in nbrs]
            n_idx = [i for i in n_idx if i is not None]
            nbr_count_list.append(len(n_idx))
            nbr_idx_list.append(n_idx)
        max_nbrs = max(nbr_count_list, default=0)
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
        
        row_idx, col_idx, data_list = [], [], []
        for i, idxs in enumerate(nbr_idx_list):
            for j in idxs:
                dist = math.hypot(r_arr[i] - r_arr[j], c_arr[i] - c_arr[j])
                row_idx.append(i)
                col_idx.append(j)
                data_list.append(dist)
                
        for k in range(self._nbr_idx.shape[1]):
            nbrs = self._nbr_idx[:, k]
            valid = nbrs >= 0
            if not np.any(valid): continue
            
            nr = np.zeros(self.n_nodes, dtype=np.float32)
            nc = np.zeros(self.n_nodes, dtype=np.float32)
            nr[valid] = r_arr[nbrs[valid]]
            nc[valid] = c_arr[nbrs[valid]]
            
            d_r = nr - r_arr
            d_c = nc - c_arr
            dist = np.hypot(d_r, d_c)
            
            self._nbr_dr[valid, k] = np.where(dist[valid] > 0, d_r[valid] / dist[valid], 0.0)
            self._nbr_dc[valid, k] = np.where(dist[valid] > 0, d_c[valid] / dist[valid], 0.0)
            self._nbr_dist[valid, k] = dist[valid]

        if data_list:
            data = np.array(data_list, dtype=np.float32)
            self._graph = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            self._graph = sp.csr_matrix((n, n), dtype=np.float32)

    def _ensure_initialised(self):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
            
        if self._initialised:
            return
        if self._pacman_start is not None and self._pacman_start in self._open_idx_map:
            self._b_flat.fill(0.0)
            idx = self._open_idx_map.get(self._pacman_start)
            if idx is not None:
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
        total = float(self._b_flat.sum())
        if total < 1e-12:
            n = self.n_nodes
            self._b_flat[:] = 1.0 / n
        else:
            self._b_flat /= total
        self._sync_flat_to_grid()
        self._payload_dirty = True

    def _pellets_changed(self, known_pellets, known_power):
        if not hasattr(self, '_last_known_pellets'): return True
        p_set = set(known_pellets) if known_pellets else set()
        pow_set = set(known_power) if known_power else set()
        return self._last_known_pellets != p_set or self._last_known_power != pow_set

    def _update_pellet_score(self, known_pellets, known_power):
        self._last_known_pellets = set(known_pellets) if known_pellets else set()
        self._last_known_power = set(known_power) if known_power else set()
        self._pellet_score = np.zeros(self.n_nodes, dtype=np.float32)
        if self.n_nodes == 0: return
        all_p = list(self._last_known_pellets) + list(self._last_known_power)
        if not all_p: return
        p_arr = np.array(all_p, dtype=np.float32)
        tree = cKDTree(p_arr)
        dists, _ = tree.query(self._open_arr)
        self._pellet_score = np.exp(-dists / 5.0)

    def _predictive_diffuse(self, known_pellets, known_power):
        if self.n_nodes == 0 or self._nbr_idx.shape[1] == 0:
            return
            
        outflow = self._b_flat * (ALPHA_UNIFORM + ALPHA_MOMENTUM)
        W = np.zeros_like(self._nbr_idx, dtype=np.float32)
        valid_mask = self._nbr_idx >= 0
        
        W[valid_mask] = 1.0 / np.maximum(self._nbr_dist[valid_mask], 0.1)
        
        nbrs_safe = np.zeros_like(W)
        nbrs_safe[valid_mask] = self._safety[self._nbr_idx[valid_mask]]
        W[valid_mask] *= (nbrs_safe[valid_mask] ** 2)
        
        if self.last_known_pos is not None and self.last_known_dir != (0, 0):
            dr, dc = self.last_known_dir
            alignment = self._nbr_dr * dr + self._nbr_dc * dc
            momentum_str = math.exp(-self.frames_since_sighting / MOMENTUM_DECAY)
            mom_factor = 1.0 + momentum_str * (np.maximum(0.1, alignment + 1.0) - 1.0)
            W[valid_mask] *= mom_factor[valid_mask]
            
        if not hasattr(self, '_pellet_score') or self._pellets_changed(known_pellets, known_power):
            self._update_pellet_score(known_pellets, known_power)
            
        if hasattr(self, '_pellet_score'):
            p_score = self._pellet_score[self._nbr_idx[valid_mask]]
            W[valid_mask] *= (1.0 + 0.5 * p_score)
            
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.where(W_sum > 0, W_sum, 1.0)
        W /= W_sum
        
        shares = outflow[:, np.newaxis] * W
        self._b_flat -= outflow
        
        for k in range(self._nbr_idx.shape[1]):
            valid = self._nbr_idx[:, k] >= 0
            receivers = self._nbr_idx[valid, k]
            np.add.at(self._b_flat, receivers, shares[valid, k])
            
        self._b_flat = np.maximum(0.0, self._b_flat)