"""
Headless training environment that wraps the Pacman game loop.

Steps the game by 5 frames per RL decision (matching CBBA auction cadence),
collects per-ghost rewards, and exposes observations + heuristic BC targets.
"""

import os
import random
import math
import numpy as np
import pacman as _pac
from pacman import generate_map, Player, WALL, PELLET, POWER, EMPTY
from ghost  import Ghost, GHOST_COLORS
import pathfinder
from obs import (build_spatial, build_global_spatial, build_vector, build_valid_mask, actions_to_tasks, MAX_H, MAX_W, MAX_GHOSTS, UNKNOWN, SPATIAL_CH, GLOBAL_SPATIAL_CH, VEC_DIM)
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
    def __init__(self, env_id: int = 0, num_ghosts: int = _DEFAULT_GHOSTS, world_height: float = float(_DEFAULT_ROWS), world_width: float = float(_DEFAULT_COLS), obs_resolution: float = 1.0, n_power: int = _DEFAULT_POWER):
        self.env_id     = env_id
        self.num_ghosts = num_ghosts
        self.world_height = world_height
        self.world_width  = world_width
        self.obs_resolution = obs_resolution
        self.n_power    = n_power
        self.grid       = None
        self.player     = None
        self.ghosts: dict[int, Ghost] = {}
        self.frame      = 0
        self.shaper     = RewardShaper()
        self.recent_nom: dict[int, np.ndarray] = {}
        self._cached_ht: dict[int, np.ndarray] = {}   #heuristic targets cached at auction boundary
        self._cached_hspeed: dict[int, float] = {}
        self.static_pacman = False

    def reset(self):
        self.grid, self._player_start, self.world = generate_map(
            world_height=self.world_height, world_width=self.world_width, n_power=self.n_power, random_spawn=self.static_pacman, obs_resolution=self.obs_resolution)
        self.player = Player(self.grid, self._player_start, self.world)
        if self.static_pacman:
            self.player.stationary = True
        open_cells = np.array(self.world.prm_nodes) if hasattr(self.world, 'prm_nodes') and self.world.prm_nodes else np.array([[float(self._player_start[0]), float(self._player_start[1])]])
        if len(open_cells) < self.num_ghosts:
            open_cells = np.array([self.world.random_open_point() for _ in range(self.num_ghosts * 2)])
        pac = np.array(self._player_start)
        d_pac = np.sum(np.square(open_cells - pac), axis=1)
        avail = np.ones(len(open_cells), dtype=bool)
        d_ghosts = np.full(len(open_cells), np.inf)
        starts = [tuple(open_cells[np.argmax(d_pac)])]
        avail[np.argmax(d_pac)] = False
        for _ in range(self.num_ghosts - 1):
            d_last = np.sum(np.square(open_cells - np.array(starts[-1])), axis=1)
            d_ghosts = np.minimum(d_ghosts, d_last)
            scores = np.minimum(d_pac, d_ghosts)
            scores[~avail] = -1
            best = np.argmax(scores)
            starts.append(tuple(open_cells[best]))
            avail[best] = False
        self.ghosts = { i: Ghost(i, self.grid, pos, GHOST_COLORS[i % len(GHOST_COLORS)], self._player_start, self.world) for i, pos in enumerate(starts) }
        self.frame = 0
        self.shaper.reset()
        r = int(self.world_height * self.obs_resolution)
        c = int(self.world_width * self.obs_resolution)
        self.recent_nom = { i: np.zeros((r, c), dtype=np.float32) for i in range(self.num_ghosts) }
        self._cached_ht = {}
        #pre-populate heuristic targets for the initial observation
        for gid in self.ghosts:
            g = self.ghosts[gid]
            if not g.dead:
                h_tasks, _ = heuristic_generate_tasks(g, self.frame)
                target = np.zeros((r, c), dtype=np.float32)
                for t in h_tasks[:3]:
                    r_t, c_t = int(t.target_pos[0] * self.obs_resolution), int(t.target_pos[1] * self.obs_resolution)
                    if 0 <= r_t < r and 0 <= c_t < c:
                        target[r_t, c_t] = t.score
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0: continue
                                nr, nc = r_t + dr, c_t + dc
                                if 0 <= nr < r and 0 <= nc < c:
                                    wy = (float(nr) + 0.5) / self.obs_resolution
                                    wx = (float(nc) + 0.5) / self.obs_resolution
                                    if self.world.is_passable(wx, wy, radius=0.35):
                                        target[nr, nc] += t.score * 0.5
                self._cached_ht[gid] = target
        for gid in self.ghosts:
            self.ghosts[gid].cbba_agent.reset_caches()
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
        sp, ve, vm, ht, hs = [], [], [], [], []
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        global_sp = build_global_spatial(self, R, C, self.obs_resolution)
        for gid in alive:
            g = self.ghosts[gid]
            sp.append(build_spatial(g, self.recent_nom[gid], R, C, self.obs_resolution))
            ve.append(build_vector(g))
            vm.append(build_valid_mask(g, R, C, self.obs_resolution))
            cached = self._cached_ht.get(gid)
            if cached is not None:
                ht.append(cached[:R, :C])
                hs.append(np.array([self._cached_hspeed.get(gid, 1.0)], dtype=np.float32))
            else:
                ht.append(np.zeros((R, C), dtype=np.float32))
                hs.append(np.array([1.0], dtype=np.float32))
        if not alive:
            z = lambda s: np.zeros(s, dtype=np.float32)
            return ([], z((0, SPATIAL_CH, R, C)), z((0, VEC_DIM)),
                    np.zeros((0, R, C), dtype=bool),
                    z((0, R, C)), z((0, 1)), z((GLOBAL_SPATIAL_CH, R, C)), (R, C))
        return (alive, np.stack(sp), np.stack(ve), np.stack(vm), np.stack(ht), np.stack(hs), global_sp, (R, C))

    def step(self, action_dict: dict, bc_prob: float = 0.0):
        info_heuristic_merges = 0
        info_total_auctions = 0
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        for gid in alive:
            g = self.ghosts[gid]
            HEURISTIC_EVERY = DECISION_INTERVAL * 2
            need_h_tasks = (self.frame % HEURISTIC_EVERY == 0) or (gid not in self._cached_ht)
            h_tasks = []
            h_dists = {}
            if need_h_tasks:
                h_tasks = heuristic_generate_tasks(g, self.frame)[0]
                target = np.zeros((R, C), dtype=np.float32)
                if h_tasks:
                    self._cached_hspeed[gid] = h_tasks[0].target_speed
                    for t in h_tasks[:3]:
                        r_t, c_t = int(t.target_pos[0] * self.obs_resolution), int(t.target_pos[1] * self.obs_resolution)
                        if 0 <= r_t < R and 0 <= c_t < C:
                            target[r_t, c_t] = t.score
                else:
                    self._cached_hspeed[gid] = 1.0
                self._cached_ht[gid] = target
            if gid in action_dict:      #merge RL tasks with CBBA
                indices, scores_map, speed = action_dict[gid]
                g.current_speed_mult = speed
                self.recent_nom[gid] *= NOM_DECAY
                for r, c in indices:
                    if 0 <= r < R and 0 <= c < C:
                        self.recent_nom[gid][r, c] = 1.0
                if self.frame % DECISION_INTERVAL == 0:
                    tasks = actions_to_tasks(g, scores_map, indices, self.frame, self.obs_resolution)
                    g.cbba_agent._last_auction = self.frame + DECISION_INTERVAL
                    if random.random() < bc_prob and h_tasks:
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
                        dists = dijkstra_multi(g.world, (g.y, g.x), all_targets)
                        h_dists.update(dists)  
                    g.cbba_agent._phase1(g, all_tasks, h_dists)
        rewards = {gid: 0.0 for gid in alive}
        done = False
        for _ in range(DECISION_INTERVAL):
            self.frame += 1
            self.player.update(self.ghosts)
            powered = self.player.powered
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                ghost.update((self.player.y, self.player.x), powered, self.ghosts, speed_mult=getattr(ghost, 'current_speed_mult', 1.0))
            if not self.player.dead:
                for gid, ghost in list(self.ghosts.items()):
                    if ghost.dead:
                        continue
                    #continuous radius-based swept-path collision (matches pacman.py visualizer)
                    collision_radius = self.player.radius + ghost.radius + 0.15
                    collided = False
                    p_path = getattr(self.player, 'path_this_frame', [(self.player.x, self.player.y)])
                    g_path = getattr(ghost, 'path_this_frame', [(ghost.x, ghost.y)])
                    n_p = len(p_path)
                    n_g = len(g_path)
                    max_segs = max(1, n_p - 1, n_g - 1)
                    samples = max_segs * 3 + 1
                    for step in range(samples):
                        t = step / (samples - 1) if samples > 1 else 0.0
                        if n_p == 1:
                            px, py = p_path[0]
                        else:
                            fp = t * (n_p - 1)
                            ip = min(int(fp), n_p - 2)
                            rem_p = fp - ip
                            px = p_path[ip][0] * (1 - rem_p) + p_path[ip + 1][0] * rem_p
                            py = p_path[ip][1] * (1 - rem_p) + p_path[ip + 1][1] * rem_p
                        if n_g == 1:
                            gx, gy = g_path[0]
                        else:
                            fg = t * (n_g - 1)
                            ig = min(int(fg), n_g - 2)
                            rem_g = fg - ig
                            gx = g_path[ig][0] * (1 - rem_g) + g_path[ig + 1][0] * rem_g
                            gy = g_path[ig][1] * (1 - rem_g) + g_path[ig + 1][1] * rem_g
                        if math.hypot(gx - px, gy - py) < collision_radius:
                            collided = True
                            break
                    if collided:
                        if self.player.powered:
                            ghost.kill()
                            if gid in rewards:
                                rewards[gid] -= 40.0
                        else:
                            self.player.die()
                            done = True
                            if gid in rewards:
                                rewards[gid] += 100.0
                            TEAM_KILL_SHARE = 0.60   #60% of kill reward shared
                            for other_gid, other_ghost in self.ghosts.items():
                                if other_gid != gid and not other_ghost.dead and other_gid in rewards:
                                    dist = math.hypot(other_ghost.y - self.player.y, other_ghost.x - self.player.x)
                                    proximity_scale = math.exp(-dist / 5.0)
                                    rewards[other_gid] += 100.0 * TEAM_KILL_SHARE * proximity_scale
                            break
                if not any(not g.dead for g in self.ghosts.values()):
                    done = True
            if done:
                break
            if (len(self.world.pellets) + len(self.world.power_pellets)) == 0:
                done = True
                for o in rewards:
                    rewards[o] -= 20.0
                break
            if not done and not self.player.dead:
                for gid_prox, ghost_prox in self.ghosts.items():
                    if ghost_prox.dead or gid_prox not in rewards:
                        continue
                    if not self.player.powered:
                        dist = math.hypot(ghost_prox.y - self.player.y, ghost_prox.x - self.player.x)
                        #reward peaks at 0.20 when adjacent, decays to about 0 beyond 8 cells
                        prox = 0.20 * math.exp(-dist / 3.0)
                        rewards[gid_prox] += prox

            grid_area = self.world.height * self.world.width
            base_area = 33 * 41
            step_cost = 0.05 * (grid_area / base_area)   #0.009 on 7x9, scales to 0.05 on 33x41
            for gid in rewards:
                if self.ghosts[gid].dead:
                    continue
                rewards[gid] -= step_cost    #per-frame step cost
                speed_mult = getattr(self.ghosts[gid], 'current_speed_mult', 1.0)
                active_task = self.ghosts[gid].cbba_agent.get_active_task()
                is_hunting = (active_task is not None and int(active_task.task_type) == 0) or \
                             (self.ghosts[gid].known_pacman is not None and not self.ghosts[gid].pacman_powered)
                if not is_hunting:
                    rewards[gid] -= 0.01 * speed_mult
        for gid, g in self.ghosts.items():
            if not g.dead and gid in rewards:
                rewards[gid] += self.shaper.shaping(g, self.ghosts)
        obs = self.observe() if not done else None
        return obs, rewards, done, {"pacman_score": getattr(self.player, "score", 0), "heuristic_merges": info_heuristic_merges, "total_auctions": info_total_auctions}