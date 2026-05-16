from __future__ import annotations
import math
from typing import Optional

WALL = 1

ALPHA_UNIFORM  = 0.12     #fraction of mass given to each neighbor every frameish
ALPHA_MOMENTUM = 0.25     #direction based mass sharing
MOMENTUM_DECAY = 80       #lower value removes trust from older sightings
TAU_RECENCY    = 60       #lower value adds trust to older messages
MIN_CONFIDENCE = 0.02     #minimum trust in any message
LOS_CERTAINTY  = 0.99     #trust in a direct sighting
LOST_SPREAD    = 0.40     #how to spread out probability if we lose sight of pacman

COMPRESS_THRESHOLD = 0.001   #cells below this are omitted from payload


class BeliefMap:
    def __init__(self, gid: int, grid: list, pacman_start: Optional[tuple] = None):
        self.gid  = gid
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self._b: list[list[float]] = [[0.0] * self.cols for _ in range(self.rows)]  #stores probabilistic beliefs per cell
        self._initialised = False
        self.last_known_pos: Optional[tuple] = None
        self.last_known_dir: tuple = (0, 0)
        self.frames_since_sighting: int = 9999
        self._pacman_start: Optional[tuple] = pacman_start   #pass actual player_start location | None => uniform prior
        self._neighbours: dict[tuple, list[tuple]] = {}
        self._open_cells: list[tuple] = []
        self._compute_topology()

    def observe(self, pacman_pos: tuple, pacman_dir: tuple = (0, 0)):
        self._ensure_initialised()
        r, c = pacman_pos
        total = sum(self._b[rr][cc] for rr, cc in self._open_cells)
        if total == 0:
            total = 1.0
        spike = total * LOS_CERTAINTY
        residual = total * (1.0 - LOS_CERTAINTY)
        for rr, cc in self._open_cells:
            self._b[rr][cc] *= residual / total
        self._b[r][c] += spike
        self.last_known_pos = pacman_pos
        self.last_known_dir = pacman_dir
        self.frames_since_sighting = 0
        self._normalise()

    def observe_lost(self, last_pos: tuple):
        self._ensure_initialised()
        r, c = last_pos
        outgoing = self._b[r][c] * LOST_SPREAD
        neighbours = self._neighbours.get((r, c), [])
        if neighbours and self.last_known_dir != (0, 0):
            dr, dc = self.last_known_dir
            weights = {}    #weighing neighbours for spread considering pacman's velocity direction
            total_w = 0.0
            for nr, nc in neighbours:
                alignment = (nr - r) * dr + (nc - c) * dc
                w = max(0.0, alignment+1.0)
                weights[(nr, nc)] = w
                total_w += w
            if total_w > 0:
                for (nr, nc), w in weights.items():
                    self._b[nr][nc] += outgoing * (w / total_w)
                self._b[r][c] -= outgoing
        self.last_known_pos = last_pos      #assuming pacman continued in the same direction for now
        self.frames_since_sighting = 0
        self._normalise()

    def observe_clear(self, visible_cells: set, pacman_pos=None):   #checking if any of the visible cells are not pacman, and zeroing their P
        self._ensure_initialised()
        changed = False
        for (r, c) in visible_cells:
            if (r, c) == pacman_pos:
                continue
            if self._b[r][c] > 0.0:
                self._b[r][c] = 0.0
                changed = True
        if changed:
            self._normalise()

    def diffuse(self, ghost_pos: tuple):
        self._ensure_initialised()
        self.frames_since_sighting = min(self.frames_since_sighting+1, 9999)
        self._uniform_diffuse()
        if self.last_known_pos is not None:
            self._momentum_diffuse()
        self._normalise()

    def merge(self, sender_gid: int, payload: dict, frame: int):
        #payload keys => 'cells':{(r,c): probability}, 'fss':frames_since_sighting, 'lkp':last_known_pos or None, 'lkd':last_known_dir
        self._ensure_initialised()
        sender_fss = payload.get("fss", 9999)
        cells: dict = payload.get("cells", {})
        if not cells:
            return
        confidence = max(MIN_CONFIDENCE, math.exp(-sender_fss/TAU_RECENCY))   #calculates confidence based on recency of sighting data

        s: list[list[float]] = [[0.0] * self.cols for _ in range(self.rows)]  #mapping recieved data into a grid
        s_total = 0.0
        for key, val in cells.items():
            r, c = key if isinstance(key, tuple) else (key[0], key[1])
            s[r][c] = float(val)
            s_total += float(val)
        if s_total < 1e-9:
            return
        for r, c in self._open_cells:
            s[r][c] /= s_total

        #log(posterior) proportional to (1-confidence)*log(prior) + confidence*log(sender)
        EPS = 1e-12   #adding miniscule value to prevent log(0)
        new_b = [row[:] for row in self._b]
        for r, c in self._open_cells:
            log_prior  = math.log(max(self._b[r][c], EPS))
            log_sender = math.log(max(s[r][c], EPS))
            new_b[r][c] = math.exp((1.0-confidence)*log_prior + confidence*log_sender)
        self._b = new_b

        #update movement state if recieved data is more recent
        lkp = payload.get("lkp")
        if lkp is not None and sender_fss < self.frames_since_sighting:
            self.last_known_pos        = tuple(lkp)
            self.last_known_dir        = tuple(payload.get("lkd", (0, 0)))
            self.frames_since_sighting = sender_fss

        self._normalise()

    def get_payload(self) -> dict:
        self._ensure_initialised()
        cells = {}
        for r, c in self._open_cells:
            v = self._b[r][c]
            if v >= COMPRESS_THRESHOLD:         #ignore rest to keep updates small
                cells[(r, c)] = round(v, 5)
        return {"cells": cells, "fss": self.frames_since_sighting, "lkp": self.last_known_pos, "lkd": self.last_known_dir}

    def top_cells(self, n: int = 5) -> list[tuple]:
        self._ensure_initialised()
        ranked = sorted(self._open_cells, key=lambda rc: self._b[rc[0]][rc[1]], reverse=True)
        return ranked[:n]

    def probability_at(self, pos: tuple) -> float:
        self._ensure_initialised()
        return self._b[pos[0]][pos[1]]

    def as_flat_list(self) -> list[float]:      #converting 2D belief grid into flat list for RL agent
        self._ensure_initialised()
        return [self._b[r][c] for r in range(self.rows) for c in range(self.cols)]

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

    def update_local_map_cell(self, pos: tuple, value: int):
        r, c = pos
        old = self.grid[r][c]
        if old == value:
            return
        self.grid[r][c] = value
        if value == WALL:
            self._remove_open_cell(pos)
            self._b[r][c] = 0.0
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
            neighbour_key = (nr, nc)
            if neighbour_key in self._neighbours:
                if pos in self._neighbours[neighbour_key]:
                    self._neighbours[neighbour_key].remove(pos)

    def _add_open_cell(self, pos: tuple):
        r, c = pos
        if pos in self._open_cells or self.grid[r][c] == WALL:
            return
        neighbours = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != WALL:
                neighbours.append((nr, nc))
        self._open_cells.append(pos)
        self._neighbours[pos] = neighbours
        for neighbour in neighbours:
            if neighbour in self._neighbours and pos not in self._neighbours[neighbour]:
                self._neighbours[neighbour].append(pos)

    def _ensure_initialised(self):
        if self._initialised:
            return
        if self._pacman_start is not None and self._pacman_start in self._open_cells:
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
            share   = outflow / len(nbrs)
            delta[r][c] -= outflow
            for nr, nc in nbrs:
                delta[nr][nc] += share
        for r, c in self._open_cells:
            self._b[r][c] = max(0.0, self._b[r][c] + delta[r][c])

    def _momentum_diffuse(self):
        if self.last_known_pos is None or self.last_known_dir == (0, 0):
            return
        momentum_strength = (ALPHA_MOMENTUM * math.exp(-self.frames_since_sighting/MOMENTUM_DECAY))
        if momentum_strength < 1e-4:
            return
        lr, lc = self.last_known_pos
        dr, dc  = self.last_known_dir
        candidates = []
        for r, c in self._neighbours.get((lr, lc), []):
            alignment = (r - lr) * dr + (c - lc) * dc
            if alignment > 0:
                candidates.append((r, c))
        if not candidates:
            return
        source_mass = self._b[lr][lc] * momentum_strength
        if source_mass < 1e-9:
            return
        share = source_mass / len(candidates)
        self._b[lr][lc] -= source_mass
        for r, c in candidates:
            self._b[r][c] += share