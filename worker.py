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
from os import environ as _env
from pacman import generate_map, Player, WALL, PELLET, POWER, EMPTY
from ghost  import Ghost, GHOST_COLORS
import pathfinder
from obs import (authoritative_task, MAX_CANDIDATES, build_spatial, build_global_spatial, build_vector, build_valid_mask, build_candidates, select_candidates, flatten_cand_cells,
                 actions_to_tasks, MAX_H, MAX_W, MAX_GHOSTS, UNKNOWN, SPATIAL_CH, GLOBAL_SPATIAL_CH, VEC_DIM,
                 MAX_CANDIDATES, CAND_FEAT_DIM)
from reward import RewardShaper
from allocator import generate_tasks as heuristic_generate_tasks
from beliefmap import extract_movement_features
from net import speed_to_mult, mult_to_throttle

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_VIDEODRIVER'] = "dummy"
_pac.AUTO_MODE = True

DECISION_INTERVAL = 6      #frames between RL decisions (= CBBA AUCTION_EVERY)
STAGGER_AUCTION = int(os.environ.get("STAGGER_AUCTION", "0"))
NOM_DECAY = 0.8            #exponential decay on recent-nomination map

_DEFAULT_ROWS = 33
_DEFAULT_COLS = 41
_DEFAULT_GHOSTS = 7
_DEFAULT_POWER = 28

KILL_TACKLER    = float(_env.get("KILL_TACKLER", "35.0"))
KILL_ASSIST     = float(_env.get("KILL_ASSIST", "15.0")) 
KILL_BASE       = float(_env.get("KILL_BASE", "20.0"))
KILL_SPEED_W    = float(_env.get("KILL_SPEED_W", "1.5"))
KILL_TIME_SCALE = float(_env.get("KILL_TIME_SCALE", "220.0"))
SURVIVOR_W      = float(_env.get("SURVIVOR_W", "10.0"))
DEATH_SELF      = float(_env.get("DEATH_SELF", "-20.0"))
DEATH_PEER      = float(_env.get("DEATH_PEER", "-2.0"))
STEP_COST       = float(_env.get("STEP_COST", "0.12"))

class Env:
    def __init__(self, env_id: int = 0, num_ghosts: int = _DEFAULT_GHOSTS, world_height: float = float(_DEFAULT_ROWS), world_width: float = float(_DEFAULT_COLS), obs_resolution: float = 1.0, n_power: int = _DEFAULT_POWER, randomize_opponent: bool = True, static_pacman: bool = False):
        self.env_id     = env_id
        self.randomize_opponent = randomize_opponent
        self.static_pacman = static_pacman
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
        self._prev_pac_dist: dict[int, float] = {}
        self._cached_ht: dict[int, np.ndarray] = {}
        self._cached_hspeed: dict[int, float] = {}
        self._cached_htasks: dict[int, list] = {}
        self._cached_full_htasks: dict[int, list] = {}
        self._cached_hdists: dict[int, dict] = {}
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
        self.player.stationary = self.static_pacman
        if self.randomize_opponent and not self.static_pacman:
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
        self._prev_pac_dist.clear()
        self.frame = 0
        self._killer_gid = -1
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
        cand_feat        : (N, MAX_CANDIDATES, CAND_FEAT_DIM) float32
        cand_cell        : (N, MAX_CANDIDATES) int64   — flattened r*cols+c at the TRIMMED width
        cand_mask        : (N, MAX_CANDIDATES) bool
        cand_bc          : (N, MAX_CANDIDATES) float32 — heuristic scores, the BC target over the set
        grid_shape       : (rows, cols) int tuple — for padding on GPU side
        """
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        sp, ve, vm, ht, hs = [], [], [], [], []
        cf, cc, cm, cbc = [], [], [], []
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
            f_c, c_c, m_c, b_c = build_candidates(g, R, C, self.obs_resolution, self._cached_hdists.get(gid))
            cf.append(f_c); cc.append(c_c); cm.append(m_c); cbc.append(b_c)
        if not alive:
            z = lambda s: np.zeros(s, dtype=np.float32)
            return ([], z((0, SPATIAL_CH, R, C)), z((0, VEC_DIM)),
                    np.zeros((0, R, C), dtype=bool),
                    z((0, R, C)), z((0, 1)),
                    z((0, MAX_CANDIDATES, CAND_FEAT_DIM)), np.zeros((0, MAX_CANDIDATES, 2), dtype=np.int64),
                    np.zeros((0, MAX_CANDIDATES), dtype=bool), z((0, MAX_CANDIDATES)),
                    z((GLOBAL_SPATIAL_CH, R, C)), (R, C))
        return (alive, np.stack(sp), np.stack(ve), np.stack(vm), np.stack(ht), np.stack(hs),
                np.stack(cf), np.stack(cc), np.stack(cm), np.stack(cbc), global_sp, (R, C))

    def _refresh_bc_targets(self):
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        self._cached_ht = {}
        self._cached_hspeed = {}
        self._cached_htasks = {}
        self._cached_full_htasks = {}
        self._cached_hdists = {}
        for gid, g in self.ghosts.items():
            if g.dead:
                continue
            h_tasks, _h_dists = heuristic_generate_tasks(g, self.frame)
            self._cached_full_htasks[gid] = list(h_tasks)
            h_cands = select_candidates(h_tasks, seed=self.frame * MAX_GHOSTS + gid)
            self._cached_htasks[gid] = h_cands
            self._cached_hdists[gid] = _h_dists
            g._rl_candidates = h_cands
            target = np.zeros((R, C), dtype=np.float32)
            if h_tasks:
                h_top = sorted(h_tasks, key=lambda z: -float(z.score))[:3]
                self._cached_hspeed[gid] = mult_to_throttle(h_top[0].target_speed)
                for t in h_top:
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

    def _run_auction_for(self, gid, tasks, h_dists):
        from cbba import _task_key
        g = self.ghosts[gid]
        g.cbba_agent._last_auction = self.frame + DECISION_INTERVAL
        g.cbba_agent._phase1(g, tasks, h_dists)
        active = g.cbba_agent.get_active_task()
        slot = -1
        if active is not None:
            ak = _task_key(active)
            cands = (getattr(g, '_rl_candidates', None) or [])[:MAX_CANDIDATES]
            for ci, t in enumerate(cands):
                if _task_key(t) == ak:
                    slot = ci
                    break
        self._last_exec_cand[gid] = slot

    def step(self, action_dict: dict, want_bc: bool = False):
        alive = [gid for gid, g in self.ghosts.items() if not g.dead]
        R = int(self.world_height * self.obs_resolution)
        C = int(self.world_width * self.obs_resolution)
        for gid in alive:
            g = self.ghosts[gid]
            if gid in action_dict:
                act_data = action_dict[gid]
                if len(act_data) >= 3 and isinstance(act_data[1], np.ndarray):
                    indices, scores_map, speed_val = act_data[0], act_data[1], act_data[2]
                    from net import speed_idx_to_mult
                    g.current_speed_mult = speed_idx_to_mult(speed_val)
                    g.rl_mode = True
                    self.recent_nom[gid] *= NOM_DECAY
                    for item in indices:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            r, c = int(item[0]), int(item[1])
                        else:
                            r, c = int(item) // C, int(item) % C
                        if 0 <= r < R and 0 <= c < C:
                            self.recent_nom[gid][r, c] = 1.0
                elif len(act_data) >= 5:
                    cand_picks, novel_pairs, speed = act_data[0], act_data[2], act_data[4]
                    g.current_rl_dir = None
                    g.rl_hijack      = False
                    g.rl_mode = True
                    g.current_speed_mult = speed_to_mult(speed)
                    self.recent_nom[gid] *= NOM_DECAY
                    cands = getattr(g, '_rl_candidates', None) or []
                    marks = list(novel_pairs)
                    for slot in cand_picks:
                        if 0 <= int(slot) < len(cands):
                            t = cands[int(slot)]
                            marks.append((int(t.target_pos[0] * self.obs_resolution), int(t.target_pos[1] * self.obs_resolution)))
                    for r, c in marks:
                        if 0 <= r < R and 0 <= c < C:
                            self.recent_nom[gid][r, c] = 1.0
        self._last_exec_cand = {}
        self._last_used_pick = {}
        self._pending_auction = {}
        if self.frame % DECISION_INTERVAL == 0:
            spatial_gids = [gid for gid in alive if gid in action_dict and len(action_dict[gid]) in (3, 4) and isinstance(action_dict[gid][1], np.ndarray)]
            if spatial_gids:
                any_restruct = any(
                    (isinstance(action_dict[gid][3], (bool, int, float, np.bool_, np.number)) and bool(action_dict[gid][3]))
                    for gid in spatial_gids if len(action_dict[gid]) >= 4
                ) or (self.frame == 0)
                pool_tasks = []
                from net import speed_idx_to_mult
                from obs import actions_to_tasks
                for gid in spatial_gids:
                    g = self.ghosts[gid]
                    act_data = action_dict[gid]
                    indices, scores_map, speed_val = act_data[0], act_data[1], act_data[2]
                    speed = speed_idx_to_mult(speed_val)
                    g.current_speed_mult = speed
                    tasks = actions_to_tasks(g, scores_map, indices, self.frame, target_speed=speed)
                    if tasks:
                        pool_tasks.extend(tasks)
                for gid in alive:
                    own_h = list(self._cached_full_htasks.get(gid, self._cached_htasks.get(gid, [])))
                    if own_h:
                        pool_tasks.extend(own_h)
                from cbba import _deduplicate_tasks
                deduped_pool = _deduplicate_tasks(pool_tasks, threshold=1.5)
                from pathfinder import dijkstra_multi
                all_targets = [t.target_pos for t in deduped_pool]
                for gid in spatial_gids:
                    g = self.ghosts[gid]
                    h_dists = dict(self._cached_hdists.get(gid, {}))
                    if all_targets:
                        d_rl = dijkstra_multi(g.world, (g.y, g.x), all_targets)
                        h_dists.update(d_rl)
                    if any_restruct or g.cbba_agent.get_active_task() is None:
                        g.cbba_agent._last_auction = self.frame + DECISION_INTERVAL
                        g.cbba_agent._phase1(g, deduped_pool, h_dists)
                if any_restruct and len(alive) > 1:
                    for _ in range(2):
                        payloads = {gid: self.ghosts[gid].cbba_agent.get_consensus_payload() for gid in alive}
                        for gid_i in alive:
                            agent_i = self.ghosts[gid_i].cbba_agent
                            for gid_j in alive:
                                if gid_i != gid_j:
                                    p_j = payloads[gid_j]
                                    agent_i.receive_consensus(gid_j, p_j["y"], p_j["z"], p_j["s"], self.frame, p_j.get("meta"))
                from cbba import _task_key
                for gid in spatial_gids:
                    g = self.ghosts[gid]
                    active = g.cbba_agent.get_active_task()
                    slot = -1
                    if active is not None:
                        ak = _task_key(active)
                        cands = (getattr(g, '_rl_candidates', None) or [])[:MAX_CANDIDATES]
                        for ci, t in enumerate(cands):
                            if _task_key(t) == ak:
                                slot = ci
                                break
                    self._last_exec_cand[gid] = slot
            else:
                for gid in alive:
                    if gid not in action_dict:
                        continue
                    g = self.ghosts[gid]
                    act_data = action_dict[gid]
                    speed = speed_to_mult(act_data[4]) if len(act_data) > 4 else 1.0
                    use_pick = bool(act_data[7]) if len(act_data) > 7 else False
                    own_h = list(self._cached_full_htasks.get(gid, self._cached_htasks.get(gid, [])))
                    auth = None
                    if use_pick and act_data[0] is not None and len(act_data[0]) > 0:
                        from obs import authoritative_task
                        auth = authoritative_task(g, int(act_data[0][0]), self.frame, target_speed=speed)
                    if auth is None and not own_h:
                        continue
                    self._last_used_pick[gid] = auth is not None
                    tasks = ([auth] if auth is not None else []) + own_h
                    h_dists = dict(self._cached_hdists.get(gid, {}))
                    if STAGGER_AUCTION:
                        self._pending_auction[gid] = (tasks, h_dists)
                    else:
                        self._run_auction_for(gid, tasks, h_dists)
        rewards = {gid: 0.0 for gid in alive}
        for gid in alive:
            if gid in action_dict and len(action_dict[gid]) in (3, 4) and len(action_dict[gid]) >= 4:
                val = action_dict[gid][3]
                if isinstance(val, (bool, int, float, np.bool_, np.number)) and bool(val):
                    rewards[gid] -= 0.005  # micro restructure communication cost
        done = False
        pred_samples = []
        for _ in range(DECISION_INTERVAL):
            self.frame += 1
            if self._pending_auction:
                for gid in list(self._pending_auction.keys()):
                    if (self.frame + gid) % DECISION_INTERVAL == 0:
                        tasks, h_dists = self._pending_auction.pop(gid)
                        if gid in self.ghosts and not self.ghosts[gid].dead:
                            self._run_auction_for(gid, tasks, h_dists)
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
                if not powered and not self.player.dead:
                    d_pac = math.hypot(ghost.y - self.player.y, ghost.x - self.player.x)
                    p_vy = float(getattr(self.player, 'vy', 0.0))
                    p_vx = float(getattr(self.player, 'vx', 0.0))
                    p_spd = math.hypot(p_vy, p_vx)
                    if p_spd > 0.05:
                        lead_y = self.player.y + (p_vy / p_spd) * 2.5
                        lead_x = self.player.x + (p_vx / p_spd) * 2.5
                        d_lead = math.hypot(ghost.y - lead_y, ghost.x - lead_x)
                        eff_d = min(d_pac, d_lead + 0.5)
                    else:
                        eff_d = d_pac
                    prev_d = self._prev_pac_dist.get(gid, eff_d)
                    d_delta = prev_d - eff_d
                    self._prev_pac_dist[gid] = eff_d
                    if gid in rewards:
                        if d_delta > 0.005:
                            rewards[gid] += 0.15 * min(d_delta, 1.0)
                        if d_pac < 5.0:
                            rewards[gid] += 0.040 * math.exp(-d_pac / 2.5)
                        v_mag = math.hypot(ghost.vx, ghost.vy)
                        spd = getattr(ghost, 'current_speed_mult', 1.0)
                        if v_mag > 0.01:
                            ux = (self.player.x - ghost.x) / max(d_pac, 1e-4)
                            uy = (self.player.y - ghost.y) / max(d_pac, 1e-4)
                            cos_theta = (ghost.vx * ux + ghost.vy * uy) / v_mag
                            if p_spd > 0.05:
                                ux_l = (lead_x - ghost.x) / max(d_lead, 1e-4)
                                uy_l = (lead_y - ghost.y) / max(d_lead, 1e-4)
                                cos_l = (ghost.vx * ux_l + ghost.vy * uy_l) / v_mag
                                cos_eff = max(cos_theta, cos_l)
                            else:
                                cos_eff = cos_theta
                            if cos_eff > 0.0:
                                rewards[gid] += 0.040 * spd * cos_eff
                            if spd < 0.85 and cos_eff > -0.3:
                                rewards[gid] -= 0.020 * (1.0 - spd)
            if not powered and not self.player.dead:
                close_gids = [g.gid for g in self.ghosts.values() if not g.dead and math.hypot(g.y - self.player.y, g.x - self.player.x) <= 6.0]
                if len(close_gids) >= 2:
                    angles = [math.atan2(self.ghosts[cg].y - self.player.y, self.ghosts[cg].x - self.player.x) for cg in close_gids]
                    R = math.hypot(sum(math.cos(a) for a in angles) / len(angles), sum(math.sin(a) for a in angles) / len(angles))
                    encirclement = max(0.0, 1.0 - R)
                    if encirclement > 0.25:
                        pincer_rew = 0.035 * encirclement * min(len(close_gids) - 1, 3)
                        for cg in close_gids:
                            if cg in rewards:
                                rewards[cg] += pincer_rew
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
                            speed_bonus = math.exp(-self.frame / max(1.0, KILL_TIME_SCALE))
                            kill_pay = KILL_BASE * (1.0 + KILL_SPEED_W * speed_bonus)
                            surv_frac = len(alive_now) / max(1, self.num_ghosts)
                            for a_g in alive_now:
                                if a_g.gid in rewards:
                                    rewards[a_g.gid] += kill_pay + SURVIVOR_W * surv_frac
                                    if a_g.gid == gid:
                                        self._killer_gid = gid
                                        rewards[a_g.gid] += KILL_TACKLER
                                    else:
                                        d_peer = math.hypot(a_g.y - self.player.y, a_g.x - self.player.x)
                                        rewards[a_g.gid] += KILL_ASSIST * math.exp(-d_peer / 6.0)
                            break
                if not any(not g.dead for g in self.ghosts.values()):
                    done = True
                    for o in rewards:
                        rewards[o] -= 25.0   #whole swarm eliminated
            if done:
                break
            if (len(self.world.pellets) + len(self.world.power_pellets)) == 0:
                done = True
                for o in rewards:
                    rewards[o] -= 20.0       #Pacman cleared the board
                break
            if self.frame >= self.max_frames:
                done = True
                for o in rewards:
                    rewards[o] -= 15.0       #ran out of time
                break
            step_cost = STEP_COST / DECISION_INTERVAL
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
        pacman_caught = bool(getattr(self.player, "dead", False))
        for gid, g in self.ghosts.items():
            if gid not in rewards:
                continue
            if g.dead:
                if gid in self.shaper._prev:
                    rewards[gid] += (0.0 - self.shaper._prev.pop(gid, 0.0)) * 0.3
            else:
                if done:
                    if pacman_caught:
                        prev_phi = self.shaper._prev.pop(gid, 0.0)
                        rewards[gid] += max(0.0, (prev_phi * self.shaper.gamma - prev_phi) * 0.3)
                    else:
                        rewards[gid] += (0.0 - self.shaper._prev.pop(gid, 0.0)) * 0.3
                else:
                    rewards[gid] += self.shaper.shaping(g, self.ghosts)
        if done:
            obs = None
        else:
            self._refresh_bc_targets()
            obs = self.observe()
        return obs, rewards, done, {"pacman_score": getattr(self.player, "score", 0), "pacman_caught": pacman_caught, "frames": self.frame, 
                                    "ghosts_dead": sum(1 for g in self.ghosts.values() if g.dead), "pred_samples": pred_samples,
                                    "killer_gid": getattr(self, '_killer_gid', -1), "exec_cand": dict(getattr(self, '_last_exec_cand', {})), "used_pick": dict(getattr(self, '_last_used_pick', {}))}