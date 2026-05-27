from __future__ import annotations
import math
from typing import Optional

WALL = 1

ALPHA_UNIFORM      = 0.12   #fraction of mass diffused to each neighbour every frame-ish
ALPHA_MOMENTUM     = 0.25   #direction-based mass sharing
MOMENTUM_DECAY     = 80     #lower value removes trust from older sightings
TAU_RECENCY        = 60     #lower value adds trust to older messages
MIN_CONFIDENCE     = 0.02   #minimum trust in any received message
LOS_CERTAINTY      = 0.99   #trust in a direct sighting
LOST_SPREAD        = 0.40   #how to spread out probability if we lose sight of pacman
COMPRESS_THRESHOLD = 0.001  #cells below this are omitted from payload

DANGER_SIGMA       = 6.0    #Gaussian variance cells for ghost danger falloff
STALENESS_DECAY    = 40.0   #frames half-life for un-refreshed ghost positions
UNSEEN_GHOST_PRIOR = 0.30   #PRIOR danger weight for a ghost whose position is unknown
PRIOR_UNIFORM_WT   = 1.0    #weight of the uniform PRIOR
MIN_SAFETY         = 1e-6   #minimum safety per cell to avoid divide-by-zero in normalisation and -infinity in logloss calc
SAFETY_RECOMPUTE_EVERY = 3  #recompute safety map at most every N frames;

HUNT_SIGMA         = 5.0    #Gaussian variance for attraction falloff toward ghosts
HUNT_CROWD_WEIGHT = 0.4     #blend factor for crowd scoring vs proximal scoring: 0.0 = pure proximal, 1.0 = pure crowd, 0.4 = blend

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
        self._b: list[list[float]] = [[0.0] * self.cols for _ in range(self.rows)]   #stores gridwise beliefmap
        self._initialised = False
        self.last_known_pos: Optional[tuple] = None
        self.last_known_dir: tuple = (0, 0)
        self.frames_since_sighting: int = 9999
        self._pacman_start: Optional[tuple] = pacman_start
        self._neighbours: dict[tuple, list[tuple]] = {}
        self._open_cells: list[tuple] = []
        self._compute_topology()
        #SafetyMap: _safety[r][c] contains [0, 1], where 1 = perfectly safe, 0 = ghost present
        self._safety: list[list[float]] = [[1.0] * self.cols for _ in range(self.rows)]
        self._last_ghost_snapshot: dict = {}     #store last known ghost positions for bayesian modelling
        self._ghost_last_seen: dict[int, int] = {}
        #Throttle tracking - skips recompute if nothing changed
        self._last_safety_frame: int = -999
        self._last_powered: bool = False

    def observe(self, pacman_pos: tuple, pacman_dir: tuple = (0, 0)):
        self._ensure_initialised()
        r, c = pacman_pos
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

    def observe_lost(self, last_pos: tuple):
        self._ensure_initialised()
        r, c       = last_pos
        outgoing   = self._b[r][c] * LOST_SPREAD
        neighbours = self._neighbours.get((r, c), [])
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
        #projecting recieved changes onto full map
        s = [[0.0] * self.cols for _ in range(self.rows)]
        s_total = 0.0
        for key, val in cells.items():
            r, c = key if isinstance(key, tuple) else (key[0], key[1])
            s[r][c] = float(val)
            s_total += float(val)
        if s_total < 1e-9:
            return
        for r, c in self._open_cells:
            s[r][c] /= s_total
        new_b = [row[:] for row in self._b]
        for r, c in self._open_cells:
            log_prior = math.log(max(self._b[r][c], 1e-12))
            log_sender = math.log(max(s[r][c], 1e-12))
            new_b[r][c] = math.exp((1.0 - confidence) * log_prior + confidence * log_sender)    #log-loss compute
        self._b = new_b
        lkp = payload.get("lkp")
        if lkp is not None and sender_fss < self.frames_since_sighting:
            self.last_known_pos = tuple(lkp)
            self.last_known_dir = tuple(payload.get("lkd", (0, 0)))
            self.frames_since_sighting = sender_fss
        self._normalise()

    def get_payload(self) -> dict:
        self._ensure_initialised()
        cells = {}
        for r, c in self._open_cells:
            v = self._b[r][c]
            if v >= COMPRESS_THRESHOLD:
                cells[(r, c)] = round(v, 5)
        return {"cells": cells, "fss":   self.frames_since_sighting, "lkp":   self.last_known_pos, "lkd":   self.last_known_dir}

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
        """
        Updated strategy to distribute probability mass across nearest cells from sighting:

            flight mode (unpowered):
            score(c) = (1 - max_i) * danger_i(c)
            We believe probability will be spread out to nearest neighbours and so on, and we do continuous calcuations cell wise to calculate the spread,
            just like the previous version.
            This time however, we rank cells by their maximum distance from all known ghosts, and favour their spread vs others - this allows for us to be 
            prepared for human player strategies that involve hiding in corners and such in our probability calculations.

            hunt mode (powered):
                can select between three strategies for ranking cells by their attraction to known ghost positions -
                "proximal": score(c) = max_i  attraction_i(c)
                            Go for the nearest ghost. Best for when ghosts are spread out.
                "crowd":    score(c) = Σ_i   attraction_i(c)
                            Go behind ghost clusters. Best for chaining kills.
                "blend":    (1−w)*proximal + w*crowd, w = HUNT_CROWD_WEIGHT (default 0.4)
                            Mostly proximal, crowd as tiebreaker. Recommended default.
        """
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
        known_positions: list[tuple] = []   #(gr, gc, recency_weight) - this decides spread for belief map
        n_unknown = 0
        for gid, pos in known_agents.items():
            if pos == "UNKNOWN":
                n_unknown += 1
                continue
            gr, gc = pos
            age = current_frame - self._ghost_last_seen.get(gid, current_frame)
            weight = math.exp(-age / STALENESS_DECAY)
            known_positions.append((gr, gc, weight))

        sigma      = HUNT_SIGMA if powered else DANGER_SIGMA
        cutoff_d2  = (3.0 * sigma) ** 2   #beyond 3sigma contribution < 1% — skip
        #Normal: start at uniform prior danger, BFS adds ghost-specific danger
        #Powered: start at zero, BFS adds ghost-specific attraction
        if not powered:
            prior = PRIOR_UNIFORM_WT / n_open
            flat_unknown = n_unknown * (UNSEEN_GHOST_PRIOR / n_open)    #spread unknown ghost prior evenly across all cells
            scores: dict[tuple, float] = {(r, c): prior + flat_unknown for r, c in self._open_cells}
        else:
            scores = {(r, c): 0.0 for r, c in self._open_cells}
            if not known_positions:
                #no ghost locations known at all — uniform score
                for r, c in self._open_cells:
                    self._safety[r][c] = 1.0 / n_open
                return
        DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        proximal: dict[tuple, float] = {(r, c): 0.0 for r, c in self._open_cells} if powered else {}    #stores max_i attraction_i(c) for hunt mode - proximality

        for gr, gc, weight in known_positions:
            visited: set   = set()
            queue:   list  = [(gr, gc, 0)]   #(row, col, dist²_so_far)
            visited.add((gr, gc))
            while queue:
                next_queue = []
                for r, c, d2 in queue:
                    if d2 > cutoff_d2:
                        continue
                    contrib = weight * math.exp(-d2 / (2.0 * sigma ** 2))
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
                            visited.add((nr, nc))   #mark walls visited to skip
                            continue
                        nd2 = (nr - gr) ** 2 + (nc - gc) ** 2
                        if nd2 <= cutoff_d2:
                            visited.add((nr, nc))
                            next_queue.append((nr, nc, nd2))
                queue = next_queue
        if not powered:
            max_score = max(scores.values()) if scores else 1.0
            if max_score < MIN_SAFETY:
                max_score = MIN_SAFETY
            for r, c in self._open_cells:
                self._safety[r][c] = 1.0 - (scores[(r, c)] / max_score)
        else:
            max_crowd = max(scores.values()) or MIN_SAFETY
            max_prox = max(proximal.values())  or MIN_SAFETY
            for r, c in self._open_cells:
                c_norm = scores[(r, c)]   / max_crowd
                p_norm = proximal[(r, c)] / max_prox
                if hunt_mode == "proximal":
                    self._safety[r][c] = p_norm
                elif hunt_mode == "crowd":
                    self._safety[r][c] = c_norm
                else:
                    self._safety[r][c] = ((1.0 - HUNT_CROWD_WEIGHT) * p_norm + HUNT_CROWD_WEIGHT  * c_norm)   #blend szn

    def safety_at(self, pos: tuple) -> float:
        return self._safety[pos[0]][pos[1]]

    def safest_cells(self, n: int = 5) -> list[tuple]:      #return n most optimal cells for pacman to move towards based on safety map
        ranked = sorted(self._open_cells, key=lambda rc: self._safety[rc[0]][rc[1]], reverse=True)
        return ranked[:n]

    def safest_neighbour(self, pacman_pos: tuple) -> Optional[tuple]:   #same as safest_cells[1]
        candidates = self._neighbours.get(pacman_pos, [])
        if not candidates:
            return None
        return max(candidates, key=lambda rc: self._safety[rc[0]][rc[1]])

    def safety_as_flat_list(self) -> list[float]:       #setup for RL strategizing
        return [self._safety[r][c]
                for r in range(self.rows) for c in range(self.cols)]

    def safety_payload(self) -> dict:
        return {(r, c): round(self._safety[r][c], 4) for r, c in self._open_cells if self._safety[r][c] < 0.95}   #only send cells that are below safety threshold for ghosts

    def update_local_map_cell(self, pos: tuple, value: int):
        r, c = pos
        old  = self.grid[r][c]
        if old == value:
            return
        self.grid[r][c] = value
        if value == WALL:
            self._remove_open_cell(pos)
            self._b[r][c] = 0.0
            self._safety[r][c] = 1.0
        elif old == WALL:
            self._add_open_cell(pos)
        self._normalise()

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
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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

    def _ensure_initialised(self):
        if self._initialised:
            return
        if (self._pacman_start is not None and self._pacman_start in self._open_cells):
            r, c = self._pacman_start
            self._b[r][c] = 1.0
        else:
            n = len(self._open_cells)
            p = 1.0 / n if n else 0.0
            for rr, cc in self._open_cells:
                self._b[rr][cc] = p
        self._initialised = True

    def _normalise(self):
        total = sum(self._b[r][c] for r, c in self._open_cells)
        if total < 1e-12:
            n = len(self._open_cells)
            p = 1.0 / n if n else 0.0
            for r, c in self._open_cells:
                self._b[r][c] = p
        else:
            for r, c in self._open_cells:
                self._b[r][c] /= total

    def _uniform_diffuse(self):
        delta = [[0.0] * self.cols for _ in range(self.rows)]
        for r, c in self._open_cells:
            nbrs = self._neighbours[(r, c)]
            if not nbrs:
                continue
            outflow = self._b[r][c] * ALPHA_UNIFORM
            share = outflow / len(nbrs)
            delta[r][c] -= outflow
            for nr, nc in nbrs:
                delta[nr][nc] += share
        for r, c in self._open_cells:
            self._b[r][c] = max(0.0, self._b[r][c] + delta[r][c])

    def _momentum_diffuse(self):
        if self.last_known_pos is None or self.last_known_dir == (0, 0):
            return
        strength = ALPHA_MOMENTUM * math.exp(
            -self.frames_since_sighting / MOMENTUM_DECAY)
        if strength < 1e-4:
            return
        lr, lc = self.last_known_pos
        dr, dc = self.last_known_dir
        candidates = [(r, c) for r, c in self._neighbours.get((lr, lc), []) if (r - lr) * dr + (c - lc) * dc > 0]
        if not candidates:
            return
        source_mass = self._b[lr][lc] * strength
        if source_mass < 1e-9:
            return
        share = source_mass / len(candidates)
        self._b[lr][lc] -= source_mass
        for r, c in candidates:
            self._b[r][c] += share