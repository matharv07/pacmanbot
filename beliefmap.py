from __future__ import annotations
import math
from typing import Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

WALL = 1

ALPHA_UNIFORM      = 0.20   #fraction of mass diffused to each neighbour every frame
ALPHA_MOMENTUM     = 0.25   #direction-based mass sharing
MOMENTUM_DECAY     = 50     #lower value removes trust from older sightings
TAU_RECENCY        = 60     #lower value adds trust to older messages
MIN_CONFIDENCE     = 0.02   #minimum trust in any received message
LOS_CERTAINTY      = 0.99   #trust in a direct sighting
LOST_SPREAD        = 0.60   #how to spread out probability if we lose sight of pacman
COMPRESS_THRESHOLD = 0.0005 #cells below this are omitted from payload

DANGER_SIGMA       = 6.0    #gaussian variance cells for ghost danger falloff
STALENESS_DECAY    = 40.0   #frames half-life for un-refreshed ghost positions
UNSEEN_GHOST_PRIOR = 0.30   #PRIOR danger weight for a ghost whose position is unknown
PRIOR_UNIFORM_WT   = 1.0    #weight of the uniform PRIOR
MIN_SAFETY         = 1e-6   #minimum safety per cell to avoid divide-by-zero in normalisation and -infinity in logloss calc
SAFETY_RECOMPUTE_EVERY = 3  #recompute safety map at most every N frames;

HUNT_SIGMA         = 5.0    #gaussian variance for attraction falloff toward ghosts
HUNT_CROWD_WEIGHT  = 0.4    #blend factor for crowd scoring vs proximal scoring: 0.0 = pure proximal, 1.0 = pure crowd, 0.4 = blend

class BeliefMap:
    """
    Safety ranking algorithm

    For every known ghost position g_i with staleness age_i (frames since last confirmed sighting):
        likelihood_i(c) = exp(-dist(c, g_i)**2 / (2 sigma**2))   [Gaussian falloff]
        weight_i        = exp(-age_i / STALENESS_DECAY)          [recency discount]
        danger_i(c)     = weight_i * likelihood_i(c)             [weighted evidence]

    Bayes update (log-space, multiplicative across independent ghosts):
        log P(c unsafe) = Σ_i  danger_i(c)

    We include a uniform prior (PRIOR_UNIFORM_WT) so that cells with no ghost evidence are not treated as perfectly safe.  After summing, we normalise the
    danger map to [0,1] and define:
        safety(c) = 1 - danger_norm(c)

    Cells are then ranked descending by safety(c) - highest safety first.
     
    Pacman in non-powered mode should generally move towards the safest cell in its visible neighbourhood, which we can check each frame.
    In powered mode, we can use the same safety map but invert it to get an "attraction" map and move towards the most attractive cell to hunt ghosts.
    """

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
        self._open_cells = world.prm_nodes
        self._neighbours = world.prm_graph
        
        self.n_nodes = len(self._open_cells)
        self._open_arr = np.array(self._open_cells, dtype=np.float32)
        self._open_idx_map = {n: i for i, n in enumerate(self._open_cells)}
        self._b_flat = np.zeros(self.n_nodes, dtype=np.float32)
        
        self._topology_dirty = True
        self._nbr_idx = np.empty((0, 0), dtype=np.int32)
        self._nbr_count = np.empty((0,), dtype=np.int32)
        self._compute_topology()
        
        #safetyMap: _safety[i] contains [0, 1] mapping to prm_nodes
        self._safety = np.ones(self.n_nodes, dtype=np.float32)
        self._last_ghost_snapshot: dict = {}
        self._ghost_last_seen: dict[int, int] = {}
        self._topology_dirty = False
        self._dirty_cells: set = set()
        self._last_safety_frame: int = -999
        self._last_powered: bool = False
        self._payload_cache: dict | None = None
        self._payload_dirty: bool = True

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

    def observe_clear(self, visible_nodes: set, pacman_pos=None):
        self._ensure_initialised()
        if not visible_nodes or self._open_arr.size == 0:
            return
        vis_arr = np.array(list(visible_nodes), dtype=np.float32)
        if pacman_pos is not None:
            keep = ~((np.abs(vis_arr[:, 0] - pacman_pos[0]) < 0.1) & (np.abs(vis_arr[:, 1] - pacman_pos[1]) < 0.1))
            vis_arr = vis_arr[keep]
        if vis_arr.size == 0:
            return
        # Find indices of all visible nodes
        idxs = [self._open_idx_map.get((r, c)) for r, c in zip(vis_arr[:, 0], vis_arr[:, 1])]
        idxs = [i for i in idxs if i is not None]
        if not idxs: return
        self._b_flat[idxs] = 0.0
        self._normalise()

    def diffuse(self, ghost_pos: tuple):
        if self._dirty_cells:
            self._rebuild_topology()
            self._dirty_cells.clear()
            self._topology_dirty = False
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting + 1, 9999)
        self._uniform_diffuse()
        if self.last_known_pos is not None:
            self._momentum_diffuse()
        self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):     #P(c | self, sender) = P(c | self)^(1−conf) x P(c | sender)^conf
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells or self._open_arr.size == 0:
            return
        confidence = max(MIN_CONFIDENCE, math.exp(-sender_fss / TAU_RECENCY))
        n = len(self._open_arr)
        s_flat = np.full(n, COMPRESS_THRESHOLD / 2.0, dtype=np.float32)
        
        idxs = []
        vals = []
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
        #vectorized: extract indices and values above threshold in one shot
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
        idx = self._closest_node(pos)
        if idx >= 0:
            return float(self._b_flat[idx])
        return 0.0

    def as_flat_list(self) -> list[float]:
        self._ensure_initialised()
        return self._b_flat.tolist()

    def update_safety_map(self, known_agents: dict, current_frame: int, powered: bool = False, hunt_mode: str = "blend"):
        new_snapshot = {gid: pos for gid, pos in known_agents.items() if pos != "UNKNOWN"}
        positions_changed = (new_snapshot != self._last_ghost_snapshot)
        mode_changed = (powered != getattr(self, "_last_powered", None))
        due = (current_frame - getattr(self, "_last_safety_frame", -999) >= SAFETY_RECOMPUTE_EVERY)
        if not (positions_changed or mode_changed or due):
            return
        self._last_ghost_snapshot = new_snapshot
        self._last_powered = powered
        self._last_safety_frame = current_frame
        for gid, pos in known_agents.items():
            if pos != "UNKNOWN":
                self._ghost_last_seen[gid] = current_frame
        n_open = len(self._open_cells)
        if n_open == 0:
            return
        known_positions: list[tuple] = []
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
        starts = []
        weights = []
        for gr, gc, weight in known_positions:
            idx = self._closest_node((gr, gc))
            if idx >= 0:
                starts.append(idx)
                weights.append(weight)
        
        for start_idx, weight in zip(starts, weights):
            dist_flat = csgraph.shortest_path(self._graph, directed=False, indices=start_idx, unweighted=True)
            mask_flat = dist_flat <= cutoff_steps
            contrib_vals = weight * np.exp(-(dist_flat[mask_flat] ** 2) / (2.0 * sigma ** 2))
            scores[mask_flat] += contrib_vals
            
            if powered:
                proximal[mask_flat] = np.maximum(proximal[mask_flat], contrib_vals)
        if not powered:
            prior = PRIOR_UNIFORM_WT / n_open
            flat_unknown = n_unknown * (UNSEEN_GHOST_PRIOR / n_open)
            scores += prior + flat_unknown
            max_score = np.max(scores)
            if max_score < MIN_SAFETY:
                max_score = MIN_SAFETY
            self._safety = 1.0 - (scores / max_score)
        else:
            max_crowd = np.max(scores)
            if max_crowd < MIN_SAFETY: max_crowd = MIN_SAFETY
            max_prox = np.max(proximal)
            if max_prox < MIN_SAFETY: max_prox = MIN_SAFETY
            c_norm = scores / max_crowd
            p_norm = proximal / max_prox
            if hunt_mode == "proximal":
                self._safety = p_norm
            elif hunt_mode == "crowd":
                self._safety = c_norm
            else:
                self._safety = ((1.0 - HUNT_CROWD_WEIGHT) * p_norm + HUNT_CROWD_WEIGHT * c_norm)

    def safety_at(self, pos: tuple) -> float:
        idx = self._closest_node(pos)
        if idx >= 0:
            return float(self._safety[idx])
        return 0.0

    def safest_cells(self, n: int = 5) -> list[tuple]:
        idx_map = {node: i for i, node in enumerate(self._open_cells)}
        ranked = sorted(self._open_cells, key=lambda node: self._safety[idx_map[node]], reverse=True)
        return ranked[:n]

    def safest_neighbour(self, pacman_pos: tuple) -> Optional[tuple]:
        candidates = self._neighbours.get(pacman_pos, [])
        if not candidates:
            return None
        return max(candidates, key=lambda rc: self._safety[rc[0]][rc[1]])

    def safety_as_flat_list(self) -> list[float]:
        return [self._safety[r][c] for r in range(self.rows) for c in range(self.cols)]

    def safety_payload(self) -> dict:
        return {(r, c): round(self._safety[r][c], 4) for r, c in self._open_cells if self._safety[r][c] < 0.95}

    def update_local_map_cell(self, pos: tuple, value: int):
        r, c = pos
        was_open = self._open_idx[r, c] >= 0
        self.grid[r][c] = value
        if value == WALL:
            mass = self._b[r][c]
            self._b[r][c] = 0.0
            self._safety[r][c] = 1.0
            if not was_open:
                return
            neighbours = [n for n in self._neighbours.get(pos, []) if self.grid[n[0]][n[1]] != WALL and self._open_idx[n[0], n[1]] >= 0]
            if mass > 0 and neighbours:
                weights = {}
                total_w = 0.0
                if self.last_known_pos is not None:
                    lr, lc = self.last_known_pos
                    dr, dc = lr - r, lc - c
                    dist = math.hypot(dr, dc)
                    if dist > 0:
                        dr /= dist
                        dc /= dist
                    for nr, nc in neighbours:
                        alignment = (nr - r) * dr + (nc - c) * dc
                        w = max(0.01, alignment + 1.0)
                        weights[(nr, nc)] = w
                        total_w += w
                else:
                    for nr, nc in neighbours:
                        weights[(nr, nc)] = 1.0
                        total_w += 1.0    
                if total_w > 0:
                    for (nr, nc), w in weights.items():
                        self._b[nr][nc] += mass * (w / total_w)
        now_open = value != WALL
        if was_open != now_open:
            if now_open:
                self._add_open_cell(pos)
            else:
                self._remove_open_cell(pos)
            self._dirty_cells.add(pos)

    def _remove_open_cell(self, pos: tuple):
        r, c = pos
        if pos not in self._open_cells:
            return
        self._open_cells.remove(pos)
        self._neighbours.pop(pos, None)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nbr_key = (nr, nc)
            if nbr_key in self._neighbours and pos in self._neighbours[nbr_key]:
                self._neighbours[nbr_key].remove(pos)

    def _add_open_cell(self, pos: tuple):
        r, c = pos
        if pos in self._open_cells or self.grid[r][c] == WALL:
            return
        neighbours = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != WALL):
                neighbours.append((nr, nc))
        self._open_cells.append(pos)
        self._neighbours[pos] = neighbours
        for nbr in neighbours:
            if nbr in self._neighbours and pos not in self._neighbours[nbr]:
                self._neighbours[nbr].append(pos)

    def _compute_topology(self):
        n = len(self._open_cells)
        if n == 0:
            self._nbr_idx = np.full((0, 0), -1, dtype=np.int32)
            self._nbr_count = np.zeros(0, dtype=np.int32)
            return
        nbr_idx_list = []
        nbr_count_list = []
        for cell in self._open_cells:
            nbrs = self._neighbours[cell]
            n_idx = [self._open_idx_map.get(n) for n in nbrs]
            n_idx = [i for i in n_idx if i is not None]
            nbr_count_list.append(len(n_idx))
            nbr_idx_list.append(n_idx)
        max_nbrs = max(nbr_count_list, default=0)
        self._nbr_idx = np.full((n, max_nbrs), -1, dtype=np.int32)
        for i, idxs in enumerate(nbr_idx_list):
            if idxs:
                self._nbr_idx[i, :len(idxs)] = idxs
        self._nbr_count = np.array(nbr_count_list, dtype=np.int32)
        
        row_idx = []
        col_idx = []
        for i, idxs in enumerate(nbr_idx_list):
            for j in idxs:
                row_idx.append(i)
                col_idx.append(j)
        data = np.ones(len(row_idx), dtype=np.float32)
        self._graph = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

    def _rebuild_topology(self):
        self._compute_topology()

    def _sync_b_to_flat(self):
        self._b_flat = self._b[self._open_arr[:, 0], self._open_arr[:, 1]].astype(np.float32)

    def _sync_flat_to_b(self):
        self._b.fill(0.0)
        if self._open_arr.size > 0:
            self._b[self._open_arr[:, 0], self._open_arr[:, 1]] = self._b_flat

    def _ensure_initialised(self):
        if self._topology_dirty or self._dirty_cells:
            self._rebuild_topology()
            self._topology_dirty = False
            self._dirty_cells.clear()
            
        if self._initialised:
            return
        if (self._pacman_start is not None and self._pacman_start in self._open_cells):
            self._b_flat.fill(0.0)
            idx = self._open_idx_map.get(self._pacman_start)
            if idx is not None:
                self._b_flat[idx] = 1.0
        else:
            self._b_flat.fill(0.0)
            n = len(self._open_cells)
            if n:
                self._b_flat[:] = 1.0 / n
        self._initialised = True

    def _normalise(self):
        self._ensure_initialised()
        if len(self._open_cells) == 0:
            return
        total = float(self._b_flat.sum())
        if total < 1e-12:
            n = len(self._open_cells)
            self._b_flat[:] = 1.0 / n
        else:
            self._b_flat /= total
        self._payload_dirty = True

    def _uniform_diffuse(self):
        if len(self._open_cells) == 0:
            return     
        outflow = self._b_flat * ALPHA_UNIFORM
        counts = self._nbr_count
        share = np.zeros_like(outflow)
        valid_mask = counts > 0
        share[valid_mask] = outflow[valid_mask] / counts[valid_mask]
        self._b_flat -= outflow
        for i in range(self._nbr_idx.shape[1]):
            nbrs = self._nbr_idx[:, i]
            valid_nbrs = nbrs >= 0
            np.add.at(self._b_flat, nbrs[valid_nbrs], share[valid_nbrs])
        self._b_flat = np.maximum(0.0, self._b_flat)

    def _momentum_diffuse(self):
        if self.last_known_pos is None or self.last_known_dir == (0, 0):
            return
        strength = ALPHA_MOMENTUM * math.exp(-self.frames_since_sighting / MOMENTUM_DECAY)
        if strength < 1e-4:
            return
        dr, dc = self.last_known_dir
        r = self._open_arr[:, 0]
        c = self._open_arr[:, 1]
        push = self._b_flat * strength
        fwd_mask = np.zeros(self._nbr_idx.shape, dtype=bool)
        for j in range(self._nbr_idx.shape[1]):
            nbr_idx = self._nbr_idx[:, j]
            valid = nbr_idx >= 0
            nr = self._open_arr[nbr_idx[valid], 0]
            nc = self._open_arr[nbr_idx[valid], 1]
            alignment = (nr - r[valid]) * dr + (nc - c[valid]) * dc
            fwd_mask[valid, j] = alignment > 0
        fwd_counts = fwd_mask.sum(axis=1)
        has_fwd = fwd_counts > 0
        share = np.zeros_like(push)
        valid_push = has_fwd & (self._b_flat >= 1e-4)
        share[valid_push] = push[valid_push] / fwd_counts[valid_push]
        self._b_flat[valid_push] -= push[valid_push]
        for j in range(self._nbr_idx.shape[1]):
            valid_receivers = valid_push & fwd_mask[:, j]
            receivers = self._nbr_idx[valid_receivers, j]
            np.add.at(self._b_flat, receivers, share[valid_receivers])
        self._b_flat = np.maximum(0.0, self._b_flat)