from __future__ import annotations
import math
from typing import Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.spatial import cKDTree
import torch
import torch.nn as nn

class LSTMMovementPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        with torch.no_grad():
            self.fc[-1].weight.fill_(0)
            self.fc[-1].bias.fill_(0)

    def forward(self, x, hx, cx, base_dir):
        hx, cx = self.lstm(x, (hx, cx))
        hx_norm = self.ln(hx)
        out = self.fc(hx_norm) + base_dir
        return out, hx, cx

WALL = 1

ALPHA_UNIFORM      = 0.15
ALPHA_MOMENTUM     = 0.15
MOMENTUM_DECAY     = 15
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

        self.lstm_predictor = LSTMMovementPredictor()
        self.lstm_hx = torch.zeros(1, 32)
        self.lstm_cx = torch.zeros(1, 32)
        self.predicted_dir = (0, 0)
        self._prm_mapping = None

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
            self._tree = cKDTree(self._open_arr)
            pairs = self._tree.query_pairs(r=CONNECT_RADIUS)
            for node in self._open_cells:
                self._neighbours[node] = []
            for i, j in pairs:
                ni, nj = self._open_cells[i], self._open_cells[j]
                self._neighbours[ni].append(nj)
                self._neighbours[nj].append(ni)

        self._topology_dirty = True

    def _closest_node(self, pos: tuple):
        if not self._open_cells: return -1
        if not hasattr(self, '_tree'):
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._open_arr)
        _, idx = self._tree.query([pos])
        return int(idx[0])

    def _closest_nodes_batch(self, pos_list: list):
        if not self._open_cells or not pos_list: return []
        if not hasattr(self, '_tree'):
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._open_arr)
        pos_arr = np.array(pos_list, dtype=np.float32)
        _, indices = self._tree.query(pos_arr)
        return indices.tolist()

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
        
        if pacman_dir != (0, 0):
            with torch.no_grad():
                idx = self._closest_node(pacman_pos)
                local_safe = float(self._safety[idx]) if idx >= 0 else 1.0
                local_pellet = float(self._pellet_score[idx]) if (idx >= 0 and hasattr(self, '_pellet_score')) else 0.0
                is_powered = 1.0 if getattr(self, '_last_powered', False) else 0.0
                
                x = torch.tensor([[
                    pacman_dir[0], pacman_dir[1],
                    is_powered,
                    local_safe,
                    local_pellet
                ]], dtype=torch.float32)
                
                base_dir = torch.tensor([[pacman_dir[0], pacman_dir[1]]], dtype=torch.float32)
                
                pred, self.lstm_hx, self.lstm_cx = self.lstm_predictor(x, self.lstm_hx, self.lstm_cx, base_dir)
                p = pred[0].numpy()
                n = np.linalg.norm(p)
                if n > 0:
                    self.predicted_dir = (float(p[0]/n), float(p[1]/n))
                else:
                    self.predicted_dir = pacman_dir

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
            
        if not hasattr(self, '_explored_nodes'):
            self._explored_nodes = set()
            
        if visible_idxs:
            self._explored_nodes.update(visible_idxs)
            idxs = np.array(list(visible_idxs), dtype=np.int32)
            if pacman_pos is not None:
                pac_idx = self._closest_node(pacman_pos)
                idxs = idxs[idxs != pac_idx]
            self._b_flat[idxs] = 0.0
            
        def _disable_edge(u, v):
            u_idx = self._open_idx_map.get(u)
            v_idx = self._open_idx_map.get(v)
            if u_idx is not None and v_idx is not None:
                if hasattr(self, '_nbr_idx'):
                    self._nbr_idx[u_idx, self._nbr_idx[u_idx] == v_idx] = -1
                    self._nbr_idx[v_idx, self._nbr_idx[v_idx] == u_idx] = -1
                    self._W_dirty = True
                if hasattr(self, '_graph') and self._graph.shape[0] > 0:
                    for idx in (u_idx, v_idx):
                        other_idx = v_idx if idx == u_idx else u_idx
                        start = self._graph.indptr[idx]
                        end = self._graph.indptr[idx+1]
                        for p in range(start, end):
                            if self._graph.indices[p] == other_idx:
                                self._graph.data[p] = 9999.0
                                
        disabled_any = False
        for node in impassable_nodes:
            if node not in self._disabled_wall_nodes:
                self._disabled_wall_nodes.add(node)
                old_nbrs = self._neighbours.get(node, [])
                self._neighbours[node] = []
                for nbr in old_nbrs:
                    if node in self._neighbours[nbr]:
                        self._neighbours[nbr].remove(node)
                    _disable_edge(node, nbr)
                        
        if hasattr(self, 'world') and hasattr(self.world, 'batch_line_of_sight') and visible_idxs:
            if not hasattr(self, '_checked_edges'):
                self._checked_edges = set()
            edges_to_check = []
            for i in visible_idxs:
                u = self._open_cells[i]
                if u in self._disabled_wall_nodes: continue
                for v in list(self._neighbours.get(u, [])):
                    edge = (u, v) if u < v else (v, u)
                    if edge not in self._checked_edges:
                        edges_to_check.append(edge)
                        self._checked_edges.add(edge)
                        
            if edges_to_check:
                p1s = np.array([(u[1], u[0]) for u, _ in edges_to_check], dtype=np.float32)
                p2s = np.array([(v[1], v[0]) for _, v in edges_to_check], dtype=np.float32)
                if hasattr(self.world, 'batch_line_of_sight_pairs'):
                    los = self.world.batch_line_of_sight_pairs(p1s, p2s, radius=0.4, step_size=0.4)
                    for (u, v), is_los in zip(edges_to_check, los):
                        if not is_los:
                            if v in self._neighbours.get(u, []):
                                self._neighbours[u].remove(v)
                            if u in self._neighbours.get(v, []):
                                self._neighbours[v].remove(u)
                            _disable_edge(u, v)
                else:
                    u_nodes = list(set([e[0] for e in edges_to_check]))
                    for u in u_nodes:
                        v_nodes = [e[1] for e in edges_to_check if e[0] == u]
                        if not v_nodes: continue
                        p1_xy = (u[1], u[0])
                        targets_xy = np.array([(v[1], v[0]) for v in v_nodes], dtype=np.float32)
                        los = self.world.batch_line_of_sight(p1_xy, targets_xy, radius=0.4, step_size=0.4)
                        for v, is_los in zip(v_nodes, los):
                            if not is_los:
                                if v in self._neighbours[u]:
                                    self._neighbours[u].remove(v)
                                if u in self._neighbours.get(v, []):
                                    self._neighbours[v].remove(u)
                                _disable_edge(u, v)
                            
        self._normalise()

    def diffuse(self, ghost_pos: tuple, known_pellets: set = None, known_power: set = None):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting + 1, 9999)
        if self.n_nodes > 0:
            if not hasattr(self, '_pellet_dists') or self._pellets_changed(known_pellets, known_power):
                self._update_pellet_score(known_pellets, known_power)
                
            W = self._compute_diffusion_weights()
            valid_mask = self._valid_mask
            receivers = self._nbr_idx[valid_mask]
            
            for _ in range(4):
                outflow = self._b_flat * (ALPHA_UNIFORM + ALPHA_MOMENTUM)
                shares = outflow[:, np.newaxis] * W
                self._b_flat -= outflow
                weights = shares[valid_mask]
                self._b_flat += np.bincount(receivers, weights=weights, minlength=self.n_nodes)
            self._b_flat = np.maximum(0.0, self._b_flat)
            self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells or self.n_nodes == 0:
            return
            
        delta_fss = self.frames_since_sighting - sender_fss
        if delta_fss > 0:
            confidence = 1.0 - math.exp(-delta_fss / 5.0)
        elif delta_fss == 0:
            confidence = 0.1
        else:
            confidence = 0.0
            
        if confidence == 0.0:
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
            self.last_known_pos        = tuple(lkp)
            self.last_known_dir        = tuple(payload.get("lkd", (0, 0)))
            
            p_dir = payload.get("p_dir", (0, 0))
            if p_dir != (0, 0):
                self.predicted_dir = tuple(p_dir)
                
            hx = payload.get("hx")
            cx = payload.get("cx")
            if hx is not None and cx is not None:
                import torch
                self.lstm_hx = torch.tensor(hx, dtype=torch.float32)
                self.lstm_cx = torch.tensor(cx, dtype=torch.float32)
                
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
        self._payload_cache = {
            "cells": cells, 
            "fss": self.frames_since_sighting, 
            "lkp": self.last_known_pos, 
            "lkd": self.last_known_dir, 
            "p_dir": self.predicted_dir,
            "hx": self.lstm_hx.tolist(),
            "cx": self.lstm_cx.tolist()
        }
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

        self._danger = scores.copy()
        
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
        for cell in self._open_cells:
            nbrs = self._neighbours.get(cell, [])
            n_idx = [self._open_idx_map.get(nb) for nb in nbrs]
            nbr_idx_list.append([i for i in n_idx if i is not None])
            
        nbr_count_list = [len(idxs) for idxs in nbr_idx_list]
        max_nbrs = max(nbr_count_list) if nbr_count_list else 0
        
        self._nbr_idx = np.full((n, max(max_nbrs, 1)), -1, dtype=np.int32)
        row_idx, col_idx = [], []
        
        for i, idxs in enumerate(nbr_idx_list):
            if idxs:
                self._nbr_idx[i, :len(idxs)] = idxs
                row_idx.extend([i] * len(idxs))
                col_idx.extend(idxs)
                
        self._nbr_count = np.array(nbr_count_list, dtype=np.int32)
        
        self._nbr_dist = np.zeros_like(self._nbr_idx, dtype=np.float32)
        self._nbr_dr = np.zeros_like(self._nbr_idx, dtype=np.float32)
        self._nbr_dc = np.zeros_like(self._nbr_idx, dtype=np.float32)
        
        r_arr = self._open_arr[:, 0]
        c_arr = self._open_arr[:, 1]
        
        for k in range(self._nbr_idx.shape[1]):
            nbrs = self._nbr_idx[:, k]
            valid = nbrs >= 0
            if not np.any(valid): continue
            
            nr = r_arr[nbrs[valid]]
            nc = c_arr[nbrs[valid]]
            
            d_r = nr - r_arr[valid]
            d_c = nc - c_arr[valid]
            dist = np.hypot(d_r, d_c)
            
            safe_dist = np.where(dist > 0, dist, 1.0)
            self._nbr_dr[valid, k] = np.where(dist > 0, d_r / safe_dist, 0.0)
            self._nbr_dc[valid, k] = np.where(dist > 0, d_c / safe_dist, 0.0)
            self._nbr_dist[valid, k] = dist
            
        if len(row_idx) > 0:
            row_idx = np.array(row_idx, dtype=np.int32)
            col_idx = np.array(col_idx, dtype=np.int32)
            d_r = r_arr[row_idx] - r_arr[col_idx]
            d_c = c_arr[row_idx] - c_arr[col_idx]
            data = np.hypot(d_r, d_c)
            self._graph = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            self._graph = sp.csr_matrix((n, n), dtype=np.float32)
            
        self._graph_version = getattr(self, '_graph_version', 0) + 1
        self._W_dirty = True

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
            
        if np.isnan(self._b_flat).any():
            self._b_flat = np.nan_to_num(self._b_flat, nan=0.0)
            
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
        new_pellets = set(known_pellets) if known_pellets else set()
        new_power = set(known_power) if known_power else set()
        
        current_graph_version = getattr(self, '_graph_version', 0)
        current_explored_len = len(getattr(self, '_explored_nodes', set()))
        if hasattr(self, '_last_known_pellets') and hasattr(self, '_pellet_dists'):
            if self._last_known_pellets == new_pellets and self._last_known_power == new_power:
                if getattr(self, '_last_pellet_graph_version', -1) == current_graph_version:
                    if getattr(self, '_last_explored_len', -1) == current_explored_len:
                        return # Cached!

        self._last_known_pellets = new_pellets
        self._last_known_power = new_power
        self._last_pellet_graph_version = current_graph_version
        self._last_explored_len = current_explored_len
        
        self._pellet_score = np.zeros(self.n_nodes, dtype=np.float32)
        self._pellet_dists = np.full(self.n_nodes, 9999.0, dtype=np.float32)
        if self.n_nodes == 0: return
        all_p = list(self._last_known_pellets) + list(self._last_known_power)
        if not all_p: return
        
        starts = self._closest_nodes_batch(all_p)
        starts = [idx for idx in starts if idx >= 0]
                
        if hasattr(self, '_explored_nodes') and self.n_nodes > 0:
            unexplored = set(range(self.n_nodes)) - self._explored_nodes
            starts.extend(list(unexplored))
                
        if starts and getattr(self, '_graph', None) is not None and self._graph.shape[0] > 0:
            starts = list(set(starts))
            n = self._graph.shape[0]
            
            indptr = self._graph.indptr
            indices = self._graph.indices
            data = self._graph.data
            
            new_indptr = np.append(indptr, indptr[-1] + len(starts))
            new_indices = np.append(indices, starts)
            new_data = np.append(data, np.zeros(len(starts), dtype=np.float32))
            
            aug_graph = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(n+1, n+1))
            
            min_dists = csgraph.dijkstra(aug_graph, directed=True, indices=n)[:n]
            min_dists[np.isinf(min_dists)] = 9999.0
            self._pellet_dists = min_dists
            self._pellet_score = np.exp(-min_dists / 10.0)
        else:
            p_arr = np.array(all_p, dtype=np.float32)
            tree = cKDTree(p_arr)
            dists, _ = tree.query(self._open_arr)
            dists[np.isinf(dists)] = 9999.0
            self._pellet_dists = dists
            self._pellet_score = np.exp(-dists / 5.0)

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
            delta_danger = my_danger - nbr_danger
            W[valid_mask] *= np.exp(delta_danger[valid_mask] * 0.8)
        
        if self.last_known_pos is not None and self.predicted_dir != (0, 0):
            dr, dc = self.predicted_dir
            alignment = self._nbr_dr * dr + self._nbr_dc * dc
            momentum_str = math.exp(-self.frames_since_sighting / MOMENTUM_DECAY)
            mom_factor = np.where(alignment > 0, 
                                  1.0 + momentum_str * (alignment * 3.0), 
                                  1.0 + momentum_str * (alignment * 0.9))
            W[valid_mask] *= mom_factor[valid_mask]
            
        if hasattr(self, '_pellet_dists'):
            my_dists = self._pellet_dists[:, np.newaxis]
            nbr_dists = self._pellet_dists[self._nbr_idx]
            nbr_dists = np.where(valid_mask, nbr_dists, my_dists)
            delta_pellet = my_dists - nbr_dists
            delta_pellet = np.clip(delta_pellet, -50.0, 50.0)
            W[valid_mask] *= np.exp(delta_pellet[valid_mask] * 0.5)
            
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.where(W_sum > 0, W_sum, 1.0)
        W /= W_sum
        return W