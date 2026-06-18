"""
Headless training environment that wraps the Pacman game loop.

Steps the game by 5 frames per RL decision (matching CBBA auction cadence),
collects per-ghost rewards, and exposes observations + heuristic BC targets.
"""

import os
import random
import numpy as np
import pacman as _pac
from pacman import generate_map, Player, WALL, PELLET, POWER, EMPTY
from ghost  import Ghost, GHOST_COLORS
import pathfinder
from obs import (build_spatial, build_global_spatial, build_vector, build_valid_mask, actions_to_tasks, MAX_H, MAX_W, MAX_GHOSTS, UNKNOWN, SPATIAL_CH, VEC_DIM)
from reward import RewardShaper
from allocator import generate_tasks as heuristic_generate_tasks

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_VIDEODRIVER'] = "dummy"
_pac.AUTO_MODE = True

DECISION_INTERVAL = 6      #frames between RL decisions (= CBBA AUCTION_EVERY)
NOM_DECAY = 0.8            #exponential decay on recent-nomination map

_DEFAULT_ROWS = 33
_DEFAULT_COLS = 41
_DEFAULT_GHOSTS = 7
_DEFAULT_POWER = 28

class Env:
    def __init__(self, env_id: int = 0, num_ghosts: int = _DEFAULT_GHOSTS, grid_rows: int = _DEFAULT_ROWS, grid_cols: int = _DEFAULT_COLS, n_power: int = _DEFAULT_POWER):
        self.env_id     = env_id
        self.num_ghosts = num_ghosts
        self.grid_rows  = grid_rows
        self.grid_cols  = grid_cols
        self.n_power    = n_power
        self.grid       = None
        self.player     = None
        self.ghosts: dict[int, Ghost] = {}
        self.frame      = 0
        self.shaper     = RewardShaper()
        self.recent_nom: dict[int, np.ndarray] = {}
        self._cached_ht: dict[int, np.ndarray] = {}   #heuristic targets cached at auction boundary
        self.static_pacman = False

    def reset(self):
        self.grid, self._player_start = generate_map(
            rows=self.grid_rows, cols=self.grid_cols, n_power=self.n_power)
        pathfinder.build_scipy_graph(self.grid)
        self.player = Player(self.grid, self._player_start)
        if self.static_pacman:
            self.player.stationary = True
        open_cells = np.argwhere(self.grid != WALL)
        pac = np.array(self._player_start)
        d_pac = np.sum(np.abs(open_cells - pac), axis=1)
        avail = np.ones(len(open_cells), dtype=bool)
        d_ghosts = np.full(len(open_cells), np.inf)
        starts = [tuple(open_cells[np.argmax(d_pac)])]
        avail[np.argmax(d_pac)] = False
        for _ in range(self.num_ghosts - 1):
            d_last = np.sum(np.abs(open_cells - np.array(starts[-1])), axis=1)
            d_ghosts = np.minimum(d_ghosts, d_last)
            scores = np.minimum(d_pac, d_ghosts)
            scores[~avail] = -1
            best = np.argmax(scores)
            starts.append(tuple(open_cells[best]))
            avail[best] = False
        self.ghosts = { i: Ghost(i, self.grid, pos, GHOST_COLORS[i % len(GHOST_COLORS)], self._player_start) for i, pos in enumerate(starts) }
        self.frame = 0
        self.shaper.reset()
        self.recent_nom = { i: np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32) for i in range(self.num_ghosts) }
        self._cached_ht = {}
        #pre-populate heuristic targets for the initial observation
        for gid in self.ghosts:
            g = self.ghosts[gid]
            if not g.dead:
                h_tasks, _ = heuristic_generate_tasks(g, self.frame)
                target = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
                for t in h_tasks[:3]:
                    r, c = t.target_pos
                    if r < self.grid_rows and c < self.grid_cols:
                        target[r, c] = t.score
                        #gaussian blur for soft BC targets
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid[nr][nc] != 1:
                                target[nr, nc] += t.score * 0.5
                        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid[nr][nc] != 1:
                                target[nr, nc] += t.score * 0.25
                self._cached_ht[gid] = target
        return self.observe()

    def observe(self):
        """
        Returns
        -------
        alive_gids       : list[int]
        spatial          : (N, C, H, W) float32   — trimmed to actual grid size
        vector           : (N, D) float32
        valid_masks      : (N, H, W) bool         — trimmed
        heuristic_targets: (N, H, W) float32      — trimmed
        grid_shape       : (rows, cols) int tuple — for padding on GPU side
        """
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        sp, ve, vm, ht = [], [], [], []
        R, C = self.grid_rows, self.grid_cols
        global_sp = build_global_spatial(self, R, C)
        for gid in alive:
            g = self.ghosts[gid]
            sp.append(build_spatial(g, self.recent_nom[gid], R, C))
            ve.append(build_vector(g))
            vm.append(build_valid_mask(g, R, C))
            cached = self._cached_ht.get(gid)
            if cached is not None:
                ht.append(cached[:R, :C])
            else:
                ht.append(np.zeros((R, C), dtype=np.float32))
        if not alive:
            z = lambda s: np.zeros(s, dtype=np.float32)
            return ([], z((0, SPATIAL_CH, R, C)), z((0, VEC_DIM)),
                    np.zeros((0, R, C), dtype=bool),
                    z((0, R, C)), z((5, R, C)), (R, C))
        return (alive, np.stack(sp), np.stack(ve), np.stack(vm), np.stack(ht), global_sp, (R, C))

    def step(self, action_dict: dict, bc_prob: float = 0.0):
        info_heuristic_merges = 0
        info_total_auctions = 0
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        for gid in alive:
            g = self.ghosts[gid]
            need_h_tasks = (self.frame % DECISION_INTERVAL == 0) or (gid not in self._cached_ht)
            h_tasks = None
            if need_h_tasks:
                h_tasks, h_dists = heuristic_generate_tasks(g, self.frame)
                target = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
                if h_tasks:
                    for t in h_tasks[:3]:
                        r, c = t.target_pos
                        if r < self.grid_rows and c < self.grid_cols:
                            target[r, c] = t.score
                            #gaussian blur for soft BC targets
                            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid[nr][nc] != 1:
                                    target[nr, nc] += t.score * 0.5
                            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid[nr][nc] != 1:
                                    target[nr, nc] += t.score * 0.25
                self._cached_ht[gid] = target
            if gid in action_dict:      #merge RL tasks with CBBA
                indices, scores_map = action_dict[gid]
                self.recent_nom[gid] *= NOM_DECAY
                for r, c in indices:
                    if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                        self.recent_nom[gid][r, c] = 1.0
                if self.frame % DECISION_INTERVAL == 0:
                    tasks = actions_to_tasks(g, scores_map, indices, self.frame)
                    g.cbba_agent._last_auction = self.frame + DECISION_INTERVAL
                    if random.random() < bc_prob:
                        all_tasks = h_tasks + tasks
                        info_heuristic_merges += 1
                    else:
                        all_tasks = tasks
                        h_dists = {}
                    info_total_auctions += 1
                    g.cbba_agent._task_map.clear()
                    #only calculate Dijkstra for RL tasks, h_dists already has heuristic distances
                    if tasks:
                        all_targets = [t.target_pos for t in tasks]
                        from pathfinder import dijkstra_multi
                        dists = dijkstra_multi(g.grid, (g.row, g.col), all_targets)
                        h_dists.update(dists)  
                    g.cbba_agent._phase1(g, all_tasks, h_dists)
        rewards = {gid: 0.0 for gid in alive}
        done = False
        for _ in range(DECISION_INTERVAL):
            self.frame += 1
            self.player.update(self.ghosts)
            powered = self.player.powered
            for gid, ghost in list(self.ghosts.items()):
                new_cells, stale_refresh = ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                if gid in rewards:
                    explore_r = (new_cells * 0.01) + (stale_refresh * 0.005)
                    rewards[gid] += min(explore_r, 0.04)  #caps exploration reward per frame to keep it below 0.05 step cost
            if not self.player.dead:
                for gid, ghost in list(self.ghosts.items()):
                    if ghost.dead:
                        continue
                    same = (ghost.row == self.player.row and ghost.col == self.player.col)
                    swap = (ghost.row == self.player.prev_row and ghost.col == self.player.prev_col
                            and self.player.row == ghost.prev_row and self.player.col == ghost.prev_col)
                    if same or swap:
                        if self.player.powered:
                            ghost.kill()
                            if gid in rewards:
                                rewards[gid] -= 60.0
                        else:
                            self.player.die()
                            done = True
                            if gid in rewards:
                                rewards[gid] += 100.0
                            break
            if done:
                break
            if int(np.sum(np.isin(self.grid, (PELLET, POWER)))) == 0:
                done = True
                for o in rewards:
                    rewards[o] -= 20.0
                break
            for gid in rewards:
                rewards[gid] -= 0.05    #per-frame step cost
        for gid, g in self.ghosts.items():
            if not g.dead and gid in rewards:
                rewards[gid] += self.shaper.shaping(g, self.ghosts)
        obs = self.observe() if not done else None
        return obs, rewards, done, {"pacman_score": getattr(self.player, "score", 0), "heuristic_merges": info_heuristic_merges, "total_auctions": info_total_auctions}