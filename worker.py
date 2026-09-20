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
from beliefmap import extract_movement_features
from net import speed_to_mult, mult_to_throttle

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_VIDEODRIVER'] = "dummy"
_pac.AUTO_MODE = True

DECISION_INTERVAL = 6      #frames between RL decisions (= CBBA AUCTION_EVERY)
NOM_DECAY = 0.8            #exponential decay on recent-nomination map

_DEFAULT_ROWS = 33
_DEFAULT_COLS = 41
_DEFAULT_GHOSTS = 7
_DEFAULT_POWER = 28

KILL_BASE    = 16.0
KILL_SPEED_W = 0.5
SURVIVOR_W   = 10.0
DEATH_SELF   = -12.0
DEATH_PEER   = -1.5

class Env:
    def __init__(self, env_id: int = 0, num_ghosts: int = _DEFAULT_GHOSTS, world_height: float = float(_DEFAULT_ROWS), world_width: float = float(_DEFAULT_COLS), obs_resolution: float = 1.0, n_power: int = _DEFAULT_POWER, randomize_opponent: bool = True):
        self.env_id     = env_id
        self.randomize_opponent = randomize_opponent
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
        self._cached_ht: dict[int, np.ndarray] = {}   #heuristic BC targets for the CURRENT observation
        self._cached_hspeed: dict[int, float] = {}
        self._cached_htasks: dict[int, list] = {}     #heuristic candidate tasks for the CURRENT observation
        self._cached_hdists: dict[int, dict] = {}     #and the belief-space distances computed for them
        self.max_frames = int(world_height * world_width * 2) + 1000
        self._pending_pred = None
        self._stored_predictor_weights = None

    def sync_predictor(self, state_dict):
        """Synchronize trained MovementPredictor weights across all active ghosts' belief maps."""
        self._stored_predictor_weights = state_dict
        for g in self.ghosts.values():
            if hasattr(g, 'belief_map') and hasattr(g.belief_map, 'predictor'):
                try:
                    g.belief_map.predictor.load_state_dict(state_dict)
                except Exception:
                    pass

    def reset(self):
        self.max_frames = int(self.world_height * self.world_width * 2) + 1000
        self.grid, self._player_start, self.world = generate_map(
            world_height=self.world_height, world_width=self.world_width, n_power=self.n_power, random_spawn=False, obs_resolution=self.obs_resolution)
        self.player = Player(self.grid, self._player_start, self.world, obs_resolution=self.obs_resolution)
        if self.randomize_opponent:
            self._randomize_opponent()
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
        for g in self.ghosts.values():
            g.rl_mode = True
            g.cbba_agent.rl_mode = True
        self.frame = 0
        self.shaper.reset()
        for g in self.ghosts.values():
            g._had_los_prev = False
        self._pending_pred = None
        if self._stored_predictor_weights is not None:
            for g in self.ghosts.values():
                if hasattr(g, 'belief_map') and hasattr(g.belief_map, 'predictor'):
                    try:
                        g.belief_map.predictor.load_state_dict(self._stored_predictor_weights)
                    except Exception:
                        pass
        r = int(self.world_height * self.obs_resolution)
        c = int(self.world_width * self.obs_resolution)
        self.recent_nom = { i: np.zeros((r, c), dtype=np.float32) for i in range(self.num_ghosts) }
        #heuristic BC targets for the initial observation (cheap: once per episode)
        self._refresh_bc_targets()
        for gid in self.ghosts:
            self.ghosts[gid].cbba_agent.reset_caches()
        return self.observe()

    def _randomize_opponent(self):
        """Draw a fresh Pacman personality each episode.

        Training against one fixed controller teaches the swarm that controller's quirks, not pursuit.
        The ranges span reckless-greedy (ignores ghosts, ignores power pellets) to cautious-power-hungry
        (detours hard around ghosts, beelines for power pellets, hunts aggressively once powered).
        Drawn from np.random so a seeded episode is still exactly reproducible.
        """
        p = self.player
        p.power_weight   = float(np.random.uniform(0.25, 1.40))
        p.danger_weight  = float(np.random.uniform(5.0, 35.0))
        p.danger_radius  = float(np.random.uniform(3.0, 6.5))
        p.replan_age     = int(np.random.randint(8, 27))
        p.emergency_dist = float(np.random.uniform(1.5, 4.5))
        p.chase_margin   = float(np.random.uniform(5.0, 30.0))

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
            s_map = build_spatial(g, self.recent_nom[gid], R, C, self.obs_resolution)
            sp.append(s_map)
            ve.append(build_vector(g))
            vm.append(build_valid_mask(g, R, C, self.obs_resolution, spatial_walls=s_map[0]))
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

    def _refresh_bc_targets(self):
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        self._cached_ht = {}
        self._cached_hspeed = {}
        self._cached_htasks = {}
        self._cached_hdists = {}
        for gid, g in self.ghosts.items():
            if g.dead:
                continue
            h_tasks, _h_dists = heuristic_generate_tasks(g, self.frame)
            self._cached_htasks[gid] = h_tasks
            self._cached_hdists[gid] = _h_dists
            g._rl_candidates = h_tasks
            target = np.zeros((R, C), dtype=np.float32)
            if h_tasks:
                self._cached_hspeed[gid] = mult_to_throttle(h_tasks[0].target_speed)
                for t in h_tasks[:3]:
                    r_t, c_t = int(t.target_pos[0] * self.obs_resolution), int(t.target_pos[1] * self.obs_resolution)
                    if 0 <= r_t < R and 0 <= c_t < C:
                        target[r_t, c_t] = t.score
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0: continue
                                nr, nc = r_t + dr, c_t + dc
                                if 0 <= nr < R and 0 <= nc < C:
                                    wy = (float(nr) + 0.5) / self.obs_resolution
                                    wx = (float(nc) + 0.5) / self.obs_resolution
                                    if self.world.is_passable(wx, wy, radius=0.35):
                                        target[nr, nc] += t.score * 0.5
            else:
                self._cached_hspeed[gid] = mult_to_throttle(1.0)
            self._cached_ht[gid] = target

    def step(self, action_dict: dict, want_bc: bool = False):
        """The heuristic candidate set is rebuilt for every returned observation regardless of want_bc: it is
        the actor's channel 11 and the auction floor, not just the BC target. RL nominations are pooled on top
        of it, so the swarm always has hunt/flank/evade/convert/explore vocabulary and the actor can only add."""
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        for gid in alive:
            g = self.ghosts[gid]
            if gid in action_dict:      #merge RL tasks with CBBA
                act_data = action_dict[gid]
                indices, scores_map, speed = act_data[0], act_data[1], act_data[2]
                g.current_rl_dir = act_data[3] if len(act_data) > 3 else None
                g.rl_hijack      = bool(act_data[4]) if len(act_data) > 4 else False
                g.rl_mode = True
                #the [0,1] throttle is a fraction of the ghost speed cap, never a free-fall to zero
                g.current_speed_mult = speed_to_mult(speed)
                self.recent_nom[gid] *= NOM_DECAY
                for r, c in indices:
                    if 0 <= r < R and 0 <= c < C:
                        self.recent_nom[gid][r, c] = 1.0
        if self.frame % DECISION_INTERVAL == 0:
            from cbba import _task_key
            pooled_tasks = {}
            for gid in alive:
                if gid not in action_dict:
                    continue
                g = self.ghosts[gid]
                act_data = action_dict[gid]
                indices = act_data[0]
                scores_map = act_data[1]
                speed = speed_to_mult(act_data[2])
                tasks = actions_to_tasks(g, scores_map, indices, self.frame, self.obs_resolution, target_speed=speed)
                cand_tasks = tasks
                cur_active = g.cbba_agent.get_active_task()
                if cur_active is not None and (self.frame - cur_active.created_frame < 24):
                    d_cur = math.hypot(cur_active.target_pos[0] - g.y, cur_active.target_pos[1] - g.x)
                    if d_cur > 0.6 and cur_active not in cand_tasks:
                        cand_tasks.append(cur_active)
                for t in cand_tasks:
                    k = _task_key(t)
                    if k in pooled_tasks and pooled_tasks[k].owner != t.owner:
                        #nominated by more than one ghost: nobody gets the own-waypoint bid edge, distance decides
                        t.assigned_to = -1
                        pooled_tasks[k].assigned_to = -1
                    if k not in pooled_tasks or t.score > pooled_tasks[k].score:
                        pooled_tasks[k] = t
            all_pooled_tasks = list(pooled_tasks.values())
            all_targets = [t.target_pos for t in all_pooled_tasks]
            for gid in alive:
                if gid not in action_dict:
                    continue
                g = self.ghosts[gid]
                own_h = list(self._cached_htasks.get(gid, []))
                if not all_pooled_tasks and not own_h:
                    continue
                g.cbba_agent._last_auction = self.frame + DECISION_INTERVAL
                h_dists = dict(self._cached_hdists.get(gid, {}))
                if all_targets:
                    h_dists.update(g.plan_dists(all_targets))
                g.cbba_agent._phase1(g, all_pooled_tasks + own_h, h_dists)
        rewards = {gid: 0.0 for gid in alive}
        done = False
        pred_samples = []
        for _ in range(DECISION_INTERVAL):
            self.frame += 1
            score_before = getattr(self.player, 'score', 0)
            powered_before = getattr(self.player, 'powered', False)
            self.player.update(self.ghosts)
            score_diff = getattr(self.player, 'score', 0) - score_before
            if score_diff > 0:
                for a_gid in alive:
                    if a_gid in rewards and not self.ghosts[a_gid].dead:
                        rewards[a_gid] -= 0.004 * score_diff
            if not powered_before and getattr(self.player, 'powered', False):
                for a_gid in alive:
                    if a_gid in rewards and not self.ghosts[a_gid].dead:
                        g_a = self.ghosts[a_gid]
                        d_pel = math.hypot(g_a.y - self.player.y, g_a.x - self.player.x)
                        rewards[a_gid] -= 0.5 + 2.0 * math.exp(-d_pel / 6.0)
            powered = self.player.powered
            new_pac_v = np.array([float(self.player.vy), float(self.player.vx)], dtype=np.float32)
            if self._pending_pred is not None:
                p_feats, p_base_v = self._pending_pred
                pred_samples.append((p_feats, p_base_v, new_pac_v))
                self._pending_pred = None
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                has_los = (ghost.known_pacman is not None)
                if has_los and not getattr(ghost, '_had_los_prev', False) and not powered:
                    if gid in rewards:
                        rewards[gid] += 0.5    #LOS discovery reward
                ghost._had_los_prev = has_los
                ghost.update((self.player.y, self.player.x), powered, self.ghosts, speed_mult=getattr(ghost, 'current_speed_mult', 1.0))
            if not self.player.dead:
                seeing_ghosts = [g for g in self.ghosts.values() if not g.dead and g.known_pacman is not None]
                if seeing_ghosts:
                    best_ghost = min(seeing_ghosts, key=lambda g: math.hypot(g.y - self.player.y, g.x - self.player.x))
                    feats = extract_movement_features(
                        pacman_pos=best_ghost.known_pacman,
                        current_vel=(float(self.player.vy), float(self.player.vx)),
                        prev_vel=(float(getattr(self.player, 'prev_vy', 0.0)), float(getattr(self.player, 'prev_vx', 0.0))),
                        known_walls=best_ghost.lidar_memory,
                        map_width=self.world.width,
                        map_height=self.world.height,
                        known_ghosts=[(g.y, g.x) for g in self.ghosts.values() if not g.dead],
                        known_pellets=best_ghost.known_pellets,
                        is_powered=self.player.powered)
                    base_v = new_pac_v.copy()
                    self._pending_pred = (feats, base_v)
                else:
                    self._pending_pred = None
            else:
                self._pending_pred = None
            self.player.prev_vy = self.player.vy
            self.player.prev_vx = self.player.vx
            if not self.player.dead:
                for gid, ghost in list(self.ghosts.items()):
                    if ghost.dead:
                        continue
                    #coarse proximity filter before expensive swept-path interpolation
                    if abs(ghost.x - self.player.x) > 2.0 or abs(ghost.y - self.player.y) > 2.0:
                        continue
                    #continuous radius-based swept-path collision
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
                            kill_y, kill_x = ghost.y, ghost.x
                            ghost.kill()
                            for other_gid, og in self.ghosts.items():
                                if other_gid != gid and not og.dead:
                                    d_w = math.hypot(og.y - kill_y, og.x - kill_x)
                                    witnesses = False
                                    if d_w <= 12.0:
                                        if og.world and hasattr(og.world, 'line_of_sight'):
                                            witnesses = og.world.line_of_sight((og.x, og.y), (kill_x, kill_y), radius=og.radius, step_size=0.5)
                                        else:
                                            witnesses = True
                                    if witnesses:
                                        og.witness_death(gid, self.ghosts)
                            if gid in rewards:
                                rewards[gid] += DEATH_SELF
                            for o_gid, og in self.ghosts.items():
                                if o_gid != gid and not og.dead and o_gid in rewards:
                                    rewards[o_gid] += DEATH_PEER   #losing a node costs the whole mesh
                        else:
                            self.player.die()
                            done = True
                            alive_now = [g2 for g2 in self.ghosts.values() if not g2.dead]
                            time_left = 1.0 - min(1.0, self.frame / max(1.0, float(self.max_frames)))
                            kill_pay = KILL_BASE * (1.0 + KILL_SPEED_W * time_left)
                            surv_frac = len(alive_now) / max(1, self.num_ghosts)
                            for a_g in alive_now:
                                if a_g.gid in rewards:
                                    rewards[a_g.gid] += kill_pay + SURVIVOR_W * surv_frac
                            break
                if not any(not g.dead for g in self.ghosts.values()):
                    done = True
                    for o in rewards:
                        rewards[o] -= 15.0   #whole swarm eliminated
            if done:
                break
            if (len(self.world.pellets) + len(self.world.power_pellets)) == 0:
                done = True
                for o in rewards:
                    rewards[o] -= 15.0   #Pacman cleared the board
                break
            if self.frame >= self.max_frames:
                done = True
                for o in rewards:
                    rewards[o] -= 10.0   #ran out of time
                break
            step_cost = 0.010 / DECISION_INTERVAL
            for gid in rewards:
                if self.ghosts[gid].dead:
                    continue
                rewards[gid] -= step_cost
                conv = getattr(self.ghosts[gid], 'power_pellets_converted_this_frame', 0)
                if conv > 0:
                    rewards[gid] += 2.5 * conv
                    for ogid in alive:
                        if ogid != gid and not self.ghosts[ogid].dead and ogid in rewards:
                            rewards[ogid] += 0.6 * conv
                    self.ghosts[gid].power_pellets_converted_this_frame = 0
        for gid, g in self.ghosts.items():
            if gid not in rewards:
                continue
            if g.dead:
                if gid in self.shaper._prev:
                    #for Ng et al. shaping, terminal potential upon death must be 0 (scaled consistently by 0.3)
                    rewards[gid] += (0.0 - self.shaper._prev.pop(gid, 0.0)) * 0.3
            else:
                if done:
                    #terminal potential must be 0 for absorbing end-of-episode state (scaled consistently by 0.3)
                    rewards[gid] += (0.0 - self.shaper._prev.pop(gid, 0.0)) * 0.3
                else:
                    rewards[gid] += self.shaper.shaping(g, self.ghosts)
        if done:
            obs = None
        else:
            self._refresh_bc_targets()
            obs = self.observe()
        pacman_caught = bool(getattr(self.player, "dead", False))
        return obs, rewards, done, {"pacman_score": getattr(self.player, "score", 0), "pacman_caught": pacman_caught, "pred_samples": pred_samples}