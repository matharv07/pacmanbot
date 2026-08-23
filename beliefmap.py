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
    
    IMPORTANT — Knowledge constraints (README rule #4):
    The belief map starts knowing ONLY Pacman's initial (t=0) position.
    Topology (navigable nodes and edges) is built incrementally from the
    ghost's own observations (prm_last_seen).  At t=0 the map contains a
    single node (pacman_start) with probability 1.0.  As the ghost explores,
    newly observed PRM nodes are added and the probability diffuses over the
    growing graph.
    """

    def __init__(self, gid: int, world, pacman_start: Optional[tuple] = None):
        self.gid = gid
        self.world = world           # kept for is_passable checks only
        self.rows = world.height
        self.cols = world.width
        self._initialised = False
        self.last_known_pos: Optional[tuple] = None
        self.last_known_dir: tuple = (0, 0)
        self.frames_since_sighting: int = 9999
        self._pacman_start: Optional[tuple] = pacman_start

        # --- Incremental topology: starts empty, grows from observations ---
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

        #safetyMap: _safety[i] contains [0, 1] mapping to known nodes
        self._safety = np.ones(0, dtype=np.float32)

        # Seed with pacman_start if known
        if pacman_start is not None:
            self._add_node(pacman_start)
        self._last_ghost_snapshot: dict = {}
        self._ghost_last_seen: dict[int, int] = {}
        self._dirty_cells: set = set()
        self._last_safety_frame: int = -999
        self._last_powered: bool = False
        self._payload_cache: dict | None = None
        self._payload_dirty: bool = True

    # --- Incremental node management ---

    def _add_node(self, node: tuple) -> bool:
        """Add a single node to the topology. Returns True if newly added."""
        if node in self._open_idx_map:
            return False
        idx = len(self._open_cells)
        self._open_cells.append(node)
        self._open_idx_map[node] = idx
        self._neighbours[node] = []
        self.n_nodes = len(self._open_cells)
        # Extend arrays
        self._b_flat = np.append(self._b_flat, 0.0).astype(np.float32)
        self._safety = np.append(self._safety, 1.0).astype(np.float32)
        if len(self._open_cells) == 1:
            self._open_arr = np.array([node], dtype=np.float32)
        else:
            self._open_arr = np.vstack([self._open_arr, np.array([node], dtype=np.float32)])
        self._topology_dirty = True
        return True

    def ingest_observed_nodes(self, observed_nodes: list, world_prm_graph: dict):
        """
        Called by the ghost after lidar sweep to feed newly observed PRM nodes
        and their world-level edges into the belief map's incremental topology.
        
        Only edges between two *observed* nodes are added — we never import
        edges to nodes we haven't seen.
        """
        any_new = False
        for node in observed_nodes:
            if self._add_node(node):
                any_new = True
        # Add edges between observed nodes
        if any_new:
            for node in observed_nodes:
                if node in world_prm_graph:
                    for nbr in world_prm_graph[node]:
                        if nbr in self._open_idx_map:
                            if nbr not in self._neighbours.get(node, []):
                                self._neighbours.setdefault(node, []).append(nbr)
                            if node not in self._neighbours.get(nbr, []):
                                self._neighbours.setdefault(nbr, []).append(node)
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

    def observe_clear(self, visible_nodes: set, pacman_pos=None):
        self._ensure_initialised()
        if not visible_nodes or self._open_arr.size == 0:
            return
        # Only clear nodes we actually have in our topology
        idxs = []
        for node in visible_nodes:
            idx = self._open_idx_map.get(node)
            if idx is not None:
                if pacman_pos is not None and abs(node[0] - pacman_pos[0]) < 0.1 and abs(node[1] - pacman_pos[1]) < 0.1:
                    continue
                idxs.append(idx)
        if not idxs: return
        self._b_flat[idxs] = 0.0
        self._normalise()

    def diffuse(self, ghost_pos: tuple):
        if self._topology_dirty:
            self._compute_topology()
            self._topology_dirty = False
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting + 1, 9999)
        if self.n_nodes > 0:
            self._uniform_diffuse()
            if self.last_known_pos is not None:
                self._momentum_diffuse()
            self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):     #P(c | self, sender) = P(c | self)^(1−conf) x P(c | sender)^conf
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells or self.n_nodes == 0:
            return
        confidence = max(MIN_CONFIDENCE, math.exp(-sender_fss / TAU_RECENCY))
        n = self.n_nodes
        s_flat = np.full(n, COMPRESS_THRESHOLD / 2.0, dtype=np.float32)
        
        idxs = []
        vals = []
        for pt, v in cells.items():
            # If sender mentions a node we haven't seen yet, skip it
            # (we can't reason about topology we don't know)
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
        
        # Batch all ghost indices into a single multi-source Dijkstra call
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

    def safety_at(self, pos: tuple) -> float:
        idx = self._closest_node(pos)
        if idx >= 0 and idx < len(self._safety):
            return float(self._safety[idx])
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
        
        row_idx = []
        col_idx = []
        data_list = []
        nodes = self._open_cells
        for i, idxs in enumerate(nbr_idx_list):
            n1 = nodes[i]
            for j in idxs:
                n2 = nodes[j]
                dist = math.hypot(n1[0] - n2[0], n1[1] - n2[1])
                row_idx.append(i)
                col_idx.append(j)
                data_list.append(dist)
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

    def _normalise(self):
        if self.n_nodes == 0:
            return
        total = float(self._b_flat.sum())
        if total < 1e-12:
            n = self.n_nodes
            self._b_flat[:] = 1.0 / n
        else:
            self._b_flat /= total
        self._payload_dirty = True

    def _uniform_diffuse(self):
        if self.n_nodes == 0 or self._nbr_idx.shape[1] == 0:
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
        if self.n_nodes == 0 or self._nbr_idx.shape[1] == 0:
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