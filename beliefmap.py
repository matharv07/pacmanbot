from __future__ import annotations
import math
from typing import Optional
import numpy as np

WALL = 1

ALPHA_UNIFORM      = 0.20   #fraction of mass diffused to each neighbour every frame
ALPHA_MOMENTUM     = 0.25   #direction-based mass sharing
MOMENTUM_DECAY     = 50     #lower value removes trust from older sightings
TAU_RECENCY        = 60     #lower value adds trust to older messages
MIN_CONFIDENCE     = 0.02   #minimum trust in any received message
LOS_CERTAINTY      = 0.99   #trust in a direct sighting
LOST_SPREAD        = 0.60   #how to spread out probability if we lose sight of pacman
COMPRESS_THRESHOLD = 0.0005 #cells below this are omitted from payload

DANGER_SIGMA       = 6.0    #Gaussian variance cells for ghost danger falloff
STALENESS_DECAY    = 40.0   #frames half-life for un-refreshed ghost positions
UNSEEN_GHOST_PRIOR = 0.30   #PRIOR danger weight for a ghost whose position is unknown
PRIOR_UNIFORM_WT   = 1.0    #weight of the uniform PRIOR
MIN_SAFETY         = 1e-6   #minimum safety per cell to avoid divide-by-zero in normalisation and -infinity in logloss calc
SAFETY_RECOMPUTE_EVERY = 3  #recompute safety map at most every N frames;

HUNT_SIGMA         = 5.0    #Gaussian variance for attraction falloff toward ghosts
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
        safety(c) = 1 − danger_norm(c)

    Cells are then ranked descending by safety(c) - highest safety first.
     
    Pacman in non-powered mode should generally move towards the safest cell in its visible neighbourhood, which we can check each frame.
    In powered mode, we can use the same safety map but invert it to get an "attraction" map and move towards the most attractive cell to hunt ghosts.
    """

    def __init__(self, gid: int, grid: list, pacman_start: Optional[tuple] = None):
        self.gid = gid
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self._b = np.zeros((self.rows, self.cols), dtype=np.float32)   #stores gridwise beliefmap
        self._initialised = False
        self.last_known_pos: Optional[tuple] = None
        self.last_known_dir: tuple = (0, 0)
        self.frames_since_sighting: int = 9999
        self._pacman_start: Optional[tuple] = pacman_start
        self._neighbours: dict[tuple, list[tuple]] = {}
        self._open_cells: list[tuple] = []
        self._open_arr = np.empty((0, 2), dtype=np.int32)
        self._open_idx = np.full((self.rows, self.cols), -1, dtype=np.int32)
        self._nbr_idx = np.empty((0, 0), dtype=np.int32)
        self._nbr_count = np.empty((0,), dtype=np.int32)
        self._b_flat = np.empty((0,), dtype=np.float32)
        self._compute_topology()
        self._sync_b_to_flat()
        #safetyMap: _safety[r][c] contains [0, 1], where 1 = perfectly safe, 0 = ghost present
        self._safety = np.ones((self.rows, self.cols), dtype=np.float32)
        self._last_ghost_snapshot: dict = {}     #store last known ghost positions for bayesian modelling
        self._ghost_last_seen: dict[int, int] = {}
        self._topology_dirty = False
        #throttle tracking - skips recompute if nothing changed
        self._last_safety_frame: int = -999
        self._last_powered: bool = False
        self._payload_cache: dict | None = None
        self._payload_dirty: bool = True

    def observe(self, pacman_pos: tuple, pacman_dir: tuple = (0, 0)):
        self._ensure_initialised()
        r, c = pacman_pos
        if self.grid[r][c] == WALL:
            return
        total = sum(self._b[rr][cc] for rr, cc in self._open_cells) or 1.0
        spike    = total * LOS_CERTAINTY
        residual = total * (1.0 - LOS_CERTAINTY)
        for rr, cc in self._open_cells:
            self._b[rr][cc] *= residual / total
        self._b[r][c] += spike
        self.last_known_pos        = pacman_pos
        self.last_known_dir        = pacman_dir
        self.frames_since_sighting = 0
        self._normalise()
        self._payload_dirty = True

    def observe_lost(self, last_pos: tuple):
        self._ensure_initialised()
        r, c       = last_pos
        if self.grid[r][c] == WALL:
            return
        outgoing   = self._b[r][c] * LOST_SPREAD
        neighbours = [n for n in self._neighbours.get((r, c), [])
                      if self.grid[n[0]][n[1]] != WALL]
        if neighbours and self.last_known_dir != (0, 0):
            dr, dc   = self.last_known_dir
            weights  = {}
            total_w  = 0.0
            for nr, nc in neighbours:
                alignment = (nr - r) * dr + (nc - c) * dc
                w = max(0.0, alignment + 1.0)
                weights[(nr, nc)] = w
                total_w += w
            if total_w > 0:
                for (nr, nc), w in weights.items():
                    self._b[nr][nc] += outgoing * (w / total_w)
                self._b[r][c] -= outgoing
        self.last_known_pos        = last_pos
        self.frames_since_sighting = 0
        self._normalise()
        self._payload_dirty = True

    def observe_clear(self, visible_cells: set, pacman_pos=None):
        self._ensure_initialised()
        changed = False
        for (r, c) in visible_cells:
            if (r, c) == pacman_pos:
                continue
            if self._b[r][c] > 1e-9:
                self._b[r][c] = 0.0
                changed = True
        if changed:
            self._normalise()

    def diffuse(self, ghost_pos: tuple):
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
        if not cells:
            return
        confidence = max(MIN_CONFIDENCE, math.exp(-sender_fss / TAU_RECENCY))
        n = len(self._open_arr)
        if n == 0:
            return
        s_flat = np.full(n, COMPRESS_THRESHOLD / 2.0, dtype=np.float32)
        s_total = 0.0
        for key, val in cells.items():
            r, c = key if isinstance(key, tuple) else (key[0], key[1])
            if 0 <= r < self.rows and 0 <= c < self.cols:
                idx = int(self._open_idx[r, c])
                if idx >= 0:
                    v = float(val)
                    s_flat[idx] = v
                    s_total += v
        if s_total < 1e-9:
            return
        s_flat /= s_flat.sum()
        prior_flat = self._b_flat.astype(np.float32)
        log_prior = np.log(np.maximum(prior_flat, 1e-12))
        log_sender = np.log(np.maximum(s_flat, 1e-12))
        new_flat = np.exp((1.0 - confidence) * log_prior + confidence * log_sender)
        self._b_flat = new_flat
        self._sync_flat_to_b()
        lkp = payload.get("lkp")
        if lkp is not None and sender_fss < self.frames_since_sighting:
            self.last_known_pos = tuple(lkp)
            self.last_known_dir = tuple(payload.get("lkd", (0, 0)))
            self.frames_since_sighting = sender_fss
        self._normalise()

    def get_payload(self) -> dict:
        self._ensure_initialised()
        if not self._payload_dirty and self._payload_cache is not None:
            return self._payload_cache
        cells = {}
        for r, c in self._open_cells:
            v = self._b[r][c]
            if v >= COMPRESS_THRESHOLD:
                cells[(r, c)] = round(v, 5)
        self._payload_cache = {"cells": cells, "fss": self.frames_since_sighting, "lkp": self.last_known_pos, "lkd": self.last_known_dir}
        self._payload_dirty = False
        return self._payload_cache
    
    def top_cells(self, n: int = 5) -> list[tuple]:
        self._ensure_initialised()
        ranked = sorted(self._open_cells, key=lambda rc: self._b[rc[0]][rc[1]], reverse=True)
        return ranked[:n]

    def probability_at(self, pos: tuple) -> float:
        self._ensure_initialised()
        return self._b[pos[0]][pos[1]]

    def as_flat_list(self) -> list[float]:
        self._ensure_initialised()
        return [self._b[r][c] for r in range(self.rows) for c in range(self.cols)]

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
        if not powered:
            prior = PRIOR_UNIFORM_WT / n_open
            flat_unknown = n_unknown * (UNSEEN_GHOST_PRIOR / n_open)
            scores: dict[tuple, float] = {(r, c): prior + flat_unknown for r, c in self._open_cells}
        else:
            scores = {(r, c): 0.0 for r, c in self._open_cells}
            if not known_positions:
                for r, c in self._open_cells:
                    self._safety[r][c] = 1.0 / n_open
                return
        DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        proximal: dict[tuple, float] = {(r, c): 0.0 for r, c in self._open_cells} if powered else {}
        for gr, gc, weight in known_positions:
            visited: set = set()
            queue: list = [(gr, gc, 0)] #track actual path steps instead of coordinate distance
            visited.add((gr, gc))
            while queue:
                next_queue = []
                for r, c, steps in queue:
                    if steps > cutoff_steps:
                        continue
                    contrib = weight * math.exp(-(steps ** 2) / (2.0 * sigma ** 2))
                    key = (r, c)
                    if key in scores:
                        scores[key] += contrib
                        if powered and contrib > proximal[key]:
                            proximal[key] = contrib        
                    for dr, dc in DIRS4:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in visited:
                            continue
                        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                            continue
                        if self.grid[nr][nc] == WALL:
                            visited.add((nr, nc))
                            continue
                        visited.add((nr, nc))
                        next_queue.append((nr, nc, steps + 1))
                queue = next_queue
        if not powered:
            max_score = max(scores.values()) if scores else 1.0
            if max_score < MIN_SAFETY:
                max_score = MIN_SAFETY
            for r, c in self._open_cells:
                self._safety[r][c] = 1.0 - (scores[(r, c)] / max_score)
        else:
            max_crowd = max(scores.values()) or MIN_SAFETY
            max_prox = max(proximal.values()) or MIN_SAFETY
            for r, c in self._open_cells:
                c_norm = scores[(r, c)] / max_crowd
                p_norm = proximal[(r, c)] / max_prox
                if hunt_mode == "proximal":
                    self._safety[r][c] = p_norm
                elif hunt_mode == "crowd":
                    self._safety[r][c] = c_norm
                else:
                    self._safety[r][c] = ((1.0 - HUNT_CROWD_WEIGHT) * p_norm + HUNT_CROWD_WEIGHT * c_norm)

    def safety_at(self, pos: tuple) -> float:
        return self._safety[pos[0]][pos[1]]

    def safest_cells(self, n: int = 5) -> list[tuple]:
        ranked = sorted(self._open_cells, key=lambda rc: self._safety[rc[0]][rc[1]], reverse=True)
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
            self._remove_open_cell(pos)
        elif not was_open and value != WALL:
            self._add_open_cell(pos)

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
        self._topology_dirty = True

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
        self._topology_dirty = True

    def _compute_topology(self):
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._open_cells = []
        self._neighbours = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == WALL:
                    continue
                self._open_cells.append((r, c))
                nbrs = []
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != WALL):
                        nbrs.append((nr, nc))
                self._neighbours[(r, c)] = nbrs
        n = len(self._open_cells)
        self._open_arr = np.array(self._open_cells, dtype=np.int32)
        self._open_idx = np.full((self.rows, self.cols), -1, dtype=np.int32)
        for i, (r, c) in enumerate(self._open_cells):
            self._open_idx[r, c] = i
        max_nbrs = max((len(v) for v in self._neighbours.values()), default=0)
        self._nbr_idx = np.full((n, max_nbrs), -1, dtype=np.int32)
        self._nbr_count = np.zeros(n, dtype=np.int32)
        for i, cell in enumerate(self._open_cells):
            nbrs = self._neighbours[cell]
            for j, nbr in enumerate(nbrs):
                self._nbr_idx[i, j] = self._open_idx[nbr]
            self._nbr_count[i] = len(nbrs)

    def _rebuild_topology(self):
        self._compute_topology()
        self._sync_b_to_flat()

    def _sync_b_to_flat(self):
        self._b_flat = self._b[self._open_arr[:, 0], self._open_arr[:, 1]].astype(np.float32)

    def _sync_flat_to_b(self):
        self._b.fill(0.0)
        if self._open_arr.size > 0:
            self._b[self._open_arr[:, 0], self._open_arr[:, 1]] = self._b_flat

    def _ensure_initialised(self):
        if self._topology_dirty:
            self._rebuild_topology()
            self._topology_dirty = False
            
        if self._initialised:
            return
        if (self._pacman_start is not None and self._pacman_start in self._open_cells):
            self._b.fill(0.0)
            r, c = self._pacman_start
            self._b[r, c] = 1.0
        else:
            self._b.fill(0.0)
            n = len(self._open_cells)
            if n:
                self._b_flat[:] = 1.0 / n
                self._sync_flat_to_b()
        self._sync_b_to_flat()
        self._initialised = True

    def _zero_walls(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == WALL:
                    self._b[r][c] = 0.0

    def _normalise(self):
        self._ensure_initialised()
        if len(self._open_cells) == 0:
            return
        self._zero_walls()
        self._sync_b_to_flat()
        total = float(self._b_flat.sum())
        if total < 1e-12:
            n = len(self._open_cells)
            self._b_flat[:] = 1.0 / n
        else:
            self._b_flat /= total
        self._sync_flat_to_b()
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
        self._sync_flat_to_b()

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
        self._sync_flat_to_b()