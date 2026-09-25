import pygame
import random
import math
from collections import deque
import numpy as np
from pathfinder import next_step, path_belief, ghost_dists, find_topological_flee_target_belief
from cbba import CBBA_Agent
from beliefmap import BeliefMap
from allocator import TaskType

CELL = 20
COLS = 41
ROWS = 33
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + 48

BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (255, 220, 0)
BLUE   = (30, 30, 180)
RED    = (220, 30, 30)
PINK   = (255, 100, 180)
CYAN   = (0, 220, 220)
ORANGE = (255, 160, 30)
DKBLUE = (10, 10, 60)
GREY   = (80, 80, 80)
POWERED_COLOR = (0, 120, 255)
GHOST_COLORS  = [RED, PINK, CYAN, ORANGE, (180, 0, 180), (0, 180, 80), (220, 220, 0)]

WALL   = 1
EMPTY  = 0
PELLET = 2
POWER  = 3

UP    = (-1,  0)
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS  = [UP, DOWN, LEFT, RIGHT]

import os as _os
RADIUS            = float(_os.environ.get("GHOST_RADIO", "12"))
GHOST_SPEED       = float(_os.environ.get("GHOST_SPEED", "0.50"))
SPEED_RATIO       = 1.0 / max(1e-6, GHOST_SPEED)   #how many ghost-cells Pacman covers per ghost-cell
RAY_COUNT         = 90
MAX_RAY_DIST      = float(_os.environ.get("GHOST_LIDAR", "10"))
MAX_RAY_DIST_SQ   = MAX_RAY_DIST * MAX_RAY_DIST
UNKNOWN           = -1
MEMORY_FRAMES     = 10
HEARTBEAT_EVERY   = 5
HEARTBEAT_TIMEOUT = 25
RESYNC_EVERY      = 100
OSCILLATION_WINDOW = 8     #position history length to prevent oscillations
RL_MAX_DEVIATION   = 1.05  #max residual rotation (rad) the policy may apply to the heuristic heading
RL_TACTICAL_RADIUS = 6.0   #range to Pacman inside which a steering hijack is allowed
EVADE_TRIGGER_DIST = 16.0  #start running before Pacman is close enough to lock on
LIDAR_SWEEP_EVERY  = 3     #lidar sweep + LOS checks every N frames
BELIEF_DIFFUSE_EVERY = 4   #belief map diffusion every N frames

_ANGLES = np.linspace(0, 2*math.pi, RAY_COUNT, endpoint=False)
_DX = np.cos(_ANGLES) * 0.5
_DY = np.sin(_ANGLES) * 0.5

def find_closest_pellet(p, world_obj, is_power=False):      #O(1) lookup to recover original float64 tuple from float32 array row
    cache_key = '_power_lookup' if is_power else '_pellet_lookup'
    lookup = getattr(world_obj, cache_key, None)
    if lookup is None:
        source_list = getattr(world_obj, 'power_pellets' if is_power else 'pellets', [])
        lookup = {(round(pt[0], 2), round(pt[1], 2)): pt for pt in source_list}
        setattr(world_obj, cache_key, lookup)
    k = (round(p[0], 2), round(p[1], 2))
    return lookup.get(k)

class Ghost:
    def __init__(self, gid, grid, pos, color, player_start, world=None):
        self.gid = gid
        self.grid = grid
        self.world = world
        self.color = color
        self.radius = 0.4
        self.x, self.y = float(pos[1]), float(pos[0])
        self.prev_x, self.prev_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0
        self.max_speed = GHOST_SPEED
        self.target_cell = pos
        self.color = color
        self.dead = False
        self.rl_mode = False
        self.in_fallback_mode = False
        self.move_every = 1
        self.last_dir = random.choice(DIRS)
        self.known_pellets = set()
        self.known_power_pellets = set()
        self.lidar_memory = set()
        self.prm_last_seen = {}                 #belief-grid node -> frame last observed (-1 = never)
        self.prm_known_count = 0
        self.frame = 0
        self.message_queue = []
        self.seen_message_ids = {}
        self.seq = 0
        self.known_agents = {}                  #(row, col) | UNKNOWN for dead/out of reach agents
        self._last_known_agent_pos = {}         #last known physical (y, x) coordinates for ghosts
        self.dead_agents = set()                #set of gids confirmed dead
        self.last_heartbeat = {}                #frame of last received heartbeat from every ghost
        self.last_sync_frame = {}               #frame of last full sync sent to every ghost
        self.known_pacman = None                #(row, col) | None for not seen yet
        self.pacman_powered = False             #normal | powered | unknown
        self.pacman_power_timer = 0
        self.pacman_last_seen = -1              #frame of when pacman was last seen for tiebreaks
        self.last_lost_pacman = None            #(row, col) of last invalidated pacman pos
        self.prev_pac_row: int = -1             #pacman's row on previous frame - belief map
        self.prev_pac_col: int = -1             #pacman's col on previous frame - belief map
        self.cbba_agent = CBBA_Agent(gid)       #CBBA auction agent for this ghost
        self.pos_history: deque = deque(maxlen=OSCILLATION_WINDOW)  #rolling position window for oscillation detection
        p_start = None
        if player_start:
            p_start = (float(np.float32(player_start[0])), float(np.float32(player_start[1])))
        self.belief_map = BeliefMap(gid, rows=int(self.world.height), cols=int(self.world.width), pacman_start=p_start)
        self.belief_map.init_full_topology()
        self.prm_last_seen = {n: -1 for n in self.belief_map._open_cells}
        self._proximity_channel_cache = None
        self._proximity_channel_frame = -1
        self._proximity_channel_target = None
        self._last_synced_map: dict[int, np.ndarray] = {}
        self.power_pellets_converted_this_frame = 0
        self.callout: Optional[str] = None
        self.callout_timer: int = 0
        self._prev_seen_pacman: Optional[tuple] = None
        self._player_dir = (0.0, 0.0)
        self.current_rl_dir: Optional[float] = None
        self.rl_hijack: bool = False        #policy asked to take over micro-navigation this step
        self.rl_mode: bool = False

    def update(self, player_pos, powered, all_ghosts, skip_movement=False, speed_mult=1.0):
        self.frame += 1
        if self.callout_timer > 0:
            self.callout_timer -= 1
            if self.callout_timer == 0:
                self.callout = None
        if getattr(self, 'pacman_power_timer', 0) > 0:
            self.pacman_power_timer -= 1
            if self.pacman_power_timer <= 0:
                self.pacman_powered = False
        if self.last_lost_pacman is not None and (self.frame - self.pacman_last_seen > 25):
            self.last_lost_pacman = None
        newly_discovered = 0
        stale_refreshed = 0.0
        if self.dead:
            return newly_discovered, stale_refreshed
        self._check_liveness(all_ghosts)
        if self.frame % LIDAR_SWEEP_EVERY == 0:
            diffs, newly_discovered, stale_refreshed = self._update_lidar_memory(all_ghosts, player_pos, powered)
        else:
            diffs = []
        if self.frame % HEARTBEAT_EVERY == 0:
            diffs.append(("heartbeat", self.gid, int(self.y), int(self.x), self.frame))
        self._broadcast(diffs, all_ghosts)
        self._process_messages(all_ghosts)
        self.belief_map.update_safety_map(self.known_agents, self.frame, powered=self.pacman_powered, pacman_pos=self.known_pacman or self.last_lost_pacman)
        if skip_movement:
            self.pos_history.append((self.y, self.x))
            self._check_oscillation()
            return newly_discovered, stale_refreshed
        active_task = self.cbba_agent.step(self, self.frame)
        if self.pacman_powered and not getattr(self, 'rl_mode', False):
            pac_danger_pos = self.known_pacman or self.last_lost_pacman
            drop_keys = []
            for k in list(self.cbba_agent.bundle):
                if k[0] == TaskType.HUNT:
                    drop_keys.append(k)
                elif pac_danger_pos is not None:
                    t_obj = self.cbba_agent._task_map.get(k)
                    if t_obj is not None and math.hypot(t_obj.target_pos[0] - pac_danger_pos[0], t_obj.target_pos[1] - pac_danger_pos[1]) < 8.0:
                        drop_keys.append(k)
            for dk in drop_keys:
                if dk in self.cbba_agent.bundle:
                    self.cbba_agent.bundle.remove(dk)
                if dk in self.cbba_agent.path:
                    self.cbba_agent.path.remove(dk)
            active_task = self.cbba_agent.get_active_task()
        if not self.pacman_powered and self.known_pacman is not None and not getattr(self, 'rl_mode', False):
            if active_task is not None and active_task.task_type in (TaskType.EXPLORE, TaskType.DYNAMIC):
                self.cbba_agent.emergency_preempt_explore()
                active_task = self.cbba_agent.get_active_task()
        #use tolerance-based comparison
        if active_task is not None:
            tpr, tpc = active_task.target_pos
            if abs(self.y - tpr) < 0.5 and abs(self.x - tpc) < 0.5:
                is_hunt = (active_task.task_type == TaskType.HUNT)
                pac_near = False
                if is_hunt and not self.pacman_powered and self.known_pacman is not None:
                    d_kp = math.hypot(self.known_pacman[0] - self.y, self.known_pacman[1] - self.x)
                    if d_kp <= 6.0:
                        pac_near = True
                        self._reached_hunt_target_near_pacman = True
                if not pac_near:
                    #cleanly remove completed task without mutating task_pos in place (preserves CBBA key matching)
                    self.cbba_agent.remove_task(active_task)
                    active_task = self.cbba_agent.get_active_task()
                else:
                    #dynamically repath to live Pacman position instead of stalling
                    self._committed_target = self.known_pacman
                    self._committed_path = []
        desired_vx = 0.0
        desired_vy = 0.0
        moved = False
        dist_pac = 999.0
        self._is_striking = False
        #Terminal Evasion: actively flee away from powered Pacman when nearby (heuristics only)
        if not moved and self.pacman_powered and not getattr(self, 'rl_mode', False):
            pac_target = self.known_pacman or self.last_lost_pacman
            if pac_target is None and hasattr(self, 'belief_map') and self.belief_map is not None:
                top = self.belief_map.top_cells(n=1)
                pac_target = top[0] if top else None
            if pac_target is not None:
                pr, pc = float(pac_target[0]), float(pac_target[1])
                dist_pac = math.hypot(pr - self.y, pc - self.x)
                if dist_pac < EVADE_TRIGGER_DIST:
                    fc = getattr(self, '_flee_cache', None)
                    if (fc is not None and self.frame - fc[0] < 4
                            and math.hypot(fc[1][0] - pr, fc[1][1] - pc) < 1.5):
                        flee_target = fc[2]
                    else:
                        flee_target = find_topological_flee_target_belief(self.belief_map, (self.y, self.x), (pr, pc))
                        self._flee_cache = (self.frame, (pr, pc), flee_target)
                    if flee_target is not None:
                        path = self.plan_path(flee_target)
                        if len(path) >= 2:
                            next_pt = path[1]
                            dx = next_pt[1] - self.x
                            dy = next_pt[0] - self.y
                            d = math.hypot(dx, dy)
                            if d > 0.01:
                                desired_vx = dx / d
                                desired_vy = dy / d
                                moved = True
                                self._committed_path = path[1:]
                                self._committed_target = flee_target
                    if not moved:
                        best_evade_vx, best_evade_vy = 0.0, 0.0
                        best_evade_score = -math.inf
                        d_away_x = self.x - pc
                        d_away_y = self.y - pr
                        d_mag = math.hypot(d_away_x, d_away_y) + 1e-6
                        dir_away_x = d_away_x / d_mag
                        dir_away_y = d_away_y / d_mag
                        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                            rvx, rvy = math.cos(angle), math.sin(angle)
                            chk_x = self.x + rvx * 1.2
                            chk_y = self.y + rvy * 1.2
                            if self.world and not self.world.is_passable(chk_x, chk_y, radius=self.radius):
                                continue
                            new_dist = math.hypot(chk_y - pr, chk_x - pc)
                            align = rvx * dir_away_x + rvy * dir_away_y
                            evade_score = new_dist * 2.0 + align * 3.0
                            if evade_score > best_evade_score:
                                best_evade_score = evade_score
                                best_evade_vx = rvx
                                best_evade_vy = rvy
                        if best_evade_score > -math.inf:
                            desired_vx = best_evade_vx
                            desired_vy = best_evade_vy
                            moved = True
                            if hasattr(self, '_committed_path'):
                                self._committed_path = []
        #Terminal strike reflex: if unpowered Pacman in line-of-sight within 2.5 cells, steer directly into Pacman at max speed (1.0)
        if not moved and not self.pacman_powered:
            pac_strike_target = None
            if self.known_pacman is not None:
                pac_strike_target = self.known_pacman
            elif active_task is not None:
                pac_strike_target = active_task.target_pos
            if pac_strike_target is not None:
                pac_y, pac_x = float(pac_strike_target[0]), float(pac_strike_target[1])
                dist_pac = math.hypot(pac_y - self.y, pac_x - self.x)
                if dist_pac <= 2.5:
                    has_los = False
                    if self.world and hasattr(self.world, 'line_of_sight'):
                        has_los = self.world.line_of_sight((self.x, self.y), (pac_x, pac_y), radius=self.radius, step_size=0.3)
                    else:
                        has_los = True
                    if has_los and dist_pac > 0.01:
                        desired_vx = (pac_x - self.x) / dist_pac
                        desired_vy = (pac_y - self.y) / dist_pac
                        speed_mult = 1.0
                        self._is_striking = True
                        moved = True
                        if hasattr(self, '_committed_path'):
                            self._committed_path = []
        if not moved and active_task is not None:
            target = active_task.target_pos
            replan = False
            prev_target = getattr(self, '_committed_target', None)
            if not getattr(self, '_committed_path', []):
                replan = True
            elif prev_target != target:
                if self.frame - getattr(self, '_last_replan_frame', -999) > 10:
                    replan = True
                elif prev_target and math.hypot(target[0] - prev_target[0], target[1] - prev_target[1]) > 3.0:
                    replan = True
            if replan:
                full_path = self.plan_path(target)
                if len(full_path) >= 2:
                    self._committed_path = full_path[1:]
                    self._committed_target = target
                    self._last_replan_frame = self.frame
                else:
                    self._committed_path = []
                    d_target = math.hypot(self.y - target[0], self.x - target[1])
                    if d_target < 1.0:
                        if active_task.task_type != TaskType.HUNT:
                            self.cbba_agent.remove_task(active_task)
                            active_task = None
                    else:
                        self.cbba_agent.mark_unreachable(target, self.frame)
                        self.cbba_agent.remove_task(active_task)
                        active_task = None
            if hasattr(self, '_committed_path') and self._committed_path:
                next_cell = self._committed_path[0]
                if abs(self.y - next_cell[0]) < 0.4 and abs(self.x - next_cell[1]) < 0.4:
                    self._committed_path.pop(0)
                    if self._committed_path:
                        next_cell = self._committed_path[0]
                    else:
                        if active_task.task_type != TaskType.HUNT:
                            self.cbba_agent.remove_task(active_task)
                            active_task = None
                if self._committed_path:
                    target_y, target_x = next_cell[0], next_cell[1]
                    dx, dy = target_x - self.x, target_y - self.y
                    d = math.hypot(dx, dy)
                    if d > 0:
                        desired_vx = dx / d
                        desired_vy = dy / d
                        moved = True
        if not moved:
            target = None
            if self.pacman_powered:
                pac_pos = self.known_pacman or self.last_lost_pacman
                if pac_pos is None and hasattr(self, 'belief_map') and self.belief_map is not None:
                    top = self.belief_map.top_cells(n=1)
                    pac_pos = top[0] if top else None
                if pac_pos is not None:
                    from allocator import _find_flee_pos
                    target = _find_flee_pos(self, pac_pos)
            else:
                if self.known_pacman is not None:
                    pr, pc = self.known_pacman
                    target = (float(pr), float(pc))
                elif self.belief_map._initialised and len(self.belief_map._b_flat) > 0:
                    best_idx = int(np.argmax(self.belief_map._b_flat))
                    if self.belief_map._b_flat[best_idx] > 1e-4:
                        best_r, best_c = self.belief_map._open_cells[best_idx]
                        target = (float(best_r), float(best_c))
            if target is not None:
                replan = False
                prev_target = getattr(self, '_committed_target', None)
                if not getattr(self, '_committed_path', []):
                    replan = True
                elif prev_target is None or math.hypot(target[0] - prev_target[0], target[1] - prev_target[1]) > 2.0:
                    if self.frame - getattr(self, '_last_replan_frame', -999) >= 8:
                        replan = True
                elif self.frame - getattr(self, '_last_replan_frame', -999) >= 30:
                    replan = True
                if replan:
                    full_path = self.plan_path(target)
                    if len(full_path) >= 2:
                        self._committed_path = full_path[1:]
                        self._committed_target = target
                        self._last_replan_frame = self.frame
                    else:
                        self._committed_path = []
                if getattr(self, '_committed_path', None):
                    next_cell = self._committed_path[0]
                    if abs(self.y - next_cell[0]) < 0.4 and abs(self.x - next_cell[1]) < 0.4:
                        self._committed_path.pop(0)
                        if self._committed_path:
                            next_cell = self._committed_path[0]
                    if self._committed_path:
                        target_y, target_x = next_cell[0], next_cell[1]
                        dx, dy = target_x - self.x, target_y - self.y
                        d = math.hypot(dx, dy)
                        if d > 0:
                            desired_vx = dx / d
                            desired_vy = dy / d
                            moved = True
        self.in_fallback_mode = not moved
        #fallback (maintain forward momentum along corridor instead of spinning, or use RL direction)
        if not moved:
            if hasattr(self, '_committed_path'):
                self._committed_path = []
            cur_speed = math.hypot(self.vx, self.vy)
            if cur_speed > 0.01:
                desired_vx = self.vx / cur_speed
                desired_vy = self.vy / cur_speed
            else:
                angle = random.uniform(0, 2*math.pi)
                desired_vx = math.cos(angle)
                desired_vy = math.sin(angle)
        if (getattr(self, 'rl_mode', False) and getattr(self, 'rl_hijack', False)
                and getattr(self, 'current_rl_dir', None) is not None
                and (desired_vx != 0.0 or desired_vy != 0.0) and self._in_tactical_envelope(active_task)):
            off = (float(self.current_rl_dir) - 0.5) * 2.0 * RL_MAX_DEVIATION
            ca, sa = math.cos(off), math.sin(off)
            desired_vx, desired_vy = desired_vx * ca - desired_vy * sa, desired_vx * sa + desired_vy * ca
        #context steering and momentum, cached every 3 frames to reduce jitter/CPU load
        _STEER_CACHE_TTL = 3
        best_vx, best_vy = desired_vx, desired_vy
        if desired_vx != 0.0 or desired_vy != 0.0:
            prev_desired = getattr(self, '_prev_desired', (0.0, 0.0))
            desired_changed = (abs(desired_vx - prev_desired[0]) > 0.05 or abs(desired_vy - prev_desired[1]) > 0.05)
            cache_stale = (self.frame - getattr(self, '_steer_cache_frame', -999)) >= _STEER_CACHE_TTL
            if desired_changed or cache_stale:
                num_rays = 16
                current_speed = self.max_speed * speed_mult
                cur_speed_mag = math.hypot(self.vx, self.vy) + 1e-6
                cur_vx_norm = self.vx / cur_speed_mag
                cur_vy_norm = self.vy / cur_speed_mag
                angles = np.linspace(0, 2*math.pi, num_rays, endpoint=False)
                ray_vx_arr = np.cos(angles)
                ray_vy_arr = np.sin(angles)
                check_dist_max = current_speed * 1.5 + self.radius
                n_steps = max(2, int(math.ceil(check_dist_max / 0.2)))
                fracs = np.linspace(1/n_steps, 1.0, n_steps)
                cc_grid = self.x + np.outer(ray_vx_arr, fracs) * check_dist_max
                cr_grid = self.y + np.outer(ray_vy_arr, fracs) * check_dist_max
                passable = self.world.batch_is_passable(cc_grid.flatten(), cr_grid.flatten(), self.radius).reshape((num_rays, n_steps))
                hit_mask = ~passable
                hit_indices = np.argmax(hit_mask, axis=1)
                has_hit = np.any(hit_mask, axis=1)
                hit_fracs = (hit_indices + 1) / n_steps
                ray_penalties = np.where(has_hit, 1000.0 / hit_fracs, 0.0)
                interests = 1.5 * (ray_vx_arr * desired_vx + ray_vy_arr * desired_vy)
                hysteresis = 0.8 * (ray_vx_arr * cur_vx_norm + ray_vy_arr * cur_vy_norm)
                peer_penalties = np.zeros(num_rays, dtype=np.float32)
                if not getattr(self, '_is_striking', False) and all_ghosts:
                    for ogid, og in all_ghosts.items():
                        if ogid == self.gid or getattr(og, 'dead', False):
                            continue
                        d_peer = math.hypot(og.y - self.y, og.x - self.x)
                        if d_peer < 1.8:
                            ux = (og.x - self.x) / max(d_peer, 1e-4)
                            uy = (og.y - self.y) / max(d_peer, 1e-4)
                            cos_align = ray_vx_arr * ux + ray_vy_arr * uy
                            fwd_mask = cos_align > 0.0
                            if np.any(fwd_mask):
                                peer_penalties[fwd_mask] += 1.5 * ((1.8 - d_peer) / 1.8) * cos_align[fwd_mask]
                scores = interests + hysteresis - ray_penalties - peer_penalties
                best_idx = int(np.argmax(scores))
                best_vx, best_vy = float(ray_vx_arr[best_idx]), float(ray_vy_arr[best_idx])
                self._steer_cache = (best_vx, best_vy)
                self._steer_cache_frame = self.frame
                self._prev_desired = (desired_vx, desired_vy)
            else:
                best_vx, best_vy = getattr(self, '_steer_cache', (desired_vx, desired_vy))
        target_vy = best_vy * self.max_speed * speed_mult
        target_vx = best_vx * self.max_speed * speed_mult
        if getattr(self, '_is_striking', False):
            smooth_vy = target_vy
            smooth_vx = target_vx
        else:
            smooth_vy = self.vy * 0.25 + target_vy * 0.75
            smooth_vx = self.vx * 0.25 + target_vx * 0.75
        smooth_safe = True
        if self.world and hasattr(self.world, 'batch_is_passable'):
            smooth_mag = math.hypot(smooth_vx, smooth_vy)
            if smooth_mag > 1e-6:
                check_dist = smooth_mag * 1.5 + self.radius
                n_steps_s = max(2, int(math.ceil(check_dist / 0.2)))
                s_vy_norm = smooth_vy / smooth_mag
                s_vx_norm = smooth_vx / smooth_mag
                fracs = np.linspace(1/n_steps_s, 1.0, n_steps_s)
                cc_arr = self.x + s_vx_norm * check_dist * fracs
                cr_arr = self.y + s_vy_norm * check_dist * fracs
                passable = self.world.batch_is_passable(cc_arr, cr_arr, self.radius)
                smooth_safe = np.all(passable)
        if smooth_safe:
            self.vy = smooth_vy
            self.vx = smooth_vx
        else:
            self.vy = target_vy
            self.vx = target_vx
        self.path_this_frame = [(self.x, self.y)]
        steps = max(1, int(math.ceil(math.hypot(self.vx, self.vy) / 0.2)))
        if steps > 0:
            step_vx = self.vx / steps
            step_vy = self.vy / steps
            for _ in range(steps):
                self.x += step_vx
                self.y += step_vy
                self.x, self.y = self.world.resolve_collision(self.x, self.y, self.radius, max_iters=3)
                self.path_this_frame.append((self.x, self.y))
        power_arr = getattr(self.world, 'power_pellets_arr', None)
        if power_arr is not None and len(power_arr) > 0:
            dist = np.hypot(power_arr[:, 0] - self.x, power_arr[:, 1] - self.y)
            close_idx = np.where(dist < self.radius + 0.5)[0]
            if len(close_idx) > 0:
                for idx in close_idx:
                    arr_xy = power_arr[idx]   #(x, y) as float32
                    #tolerance-based removal to handle float32 vs float64 mismatch
                    pt = None
                    for pp in list(self.world.power_pellets):
                        if abs(pp[0] - float(arr_xy[0])) < 0.02 and abs(pp[1] - float(arr_xy[1])) < 0.02:
                            pt = pp
                            break
                    if pt is None:
                        pt = (float(arr_xy[0]), float(arr_xy[1]))
                    if getattr(self, 'world', None):
                        for pp in list(self.world.power_pellets):
                            if abs(pp[0] - pt[0]) < 0.05 and abs(pp[1] - pt[1]) < 0.05:
                                self.world.power_pellets.remove(pp)
                                pt = pp
                                break
                        else:
                            if pt in self.world.power_pellets:
                                self.world.power_pellets.remove(pt)
                        if pt not in self.world.pellets:
                            self.world.pellets.append(pt)
                        if hasattr(self.world, '_update_pellet_arrays'):
                            self.world._update_pellet_arrays()
                    if getattr(self, 'grid', None) is not None:
                        obs_res = len(self.grid) / self.world.height if getattr(self, 'world', None) else 1.0
                        gr = int(pt[1] * obs_res)
                        gc = int(pt[0] * obs_res)
                        if 0 <= gr < len(self.grid) and 0 <= gc < len(self.grid[0]):
                            if self.grid[gr][gc] == POWER:
                                self.grid[gr][gc] = PELLET
                    self.known_power_pellets.discard(pt)
                    if pt not in self.known_pellets:
                        self.known_pellets.add(pt)
                    self.power_pellets_converted_this_frame += 1
                    self._broadcast([("power_eaten", pt), ("pellet", pt)], all_ghosts)
        self._check_oscillation()
        return newly_discovered, stale_refreshed

    def plan_path(self, target) -> list:
        return path_belief(self.belief_map, (float(self.y), float(self.x)), (float(target[0]), float(target[1])))

    def plan_dists(self, targets) -> dict:
        return ghost_dists(self, (float(self.y), float(self.x)), list(targets))

    def _in_tactical_envelope(self, active_task) -> bool:
        if getattr(self, 'in_fallback_mode', False):
            return True
        pac = self.known_pacman or self.last_lost_pacman
        if pac is not None and math.hypot(pac[0] - self.y, pac[1] - self.x) < RL_TACTICAL_RADIUS:
            return True
        if active_task is not None:
            if math.hypot(active_task.target_pos[0] - self.y, active_task.target_pos[1] - self.x) < 2.0:
                return True
        return False

    def _check_oscillation(self):
        if len(self.pos_history) < OSCILLATION_WINDOW:
            return
        cur_y, cur_x = self.y, self.x
        tol = 0.3               #tolerance for float coordinate comparison
        matches = sum(1 for py, px in self.pos_history if abs(py - cur_y) < tol and abs(px - cur_x) < tol)
        if matches >= 2:
            if self.known_pacman is None and self.last_lost_pacman is not None:
                self.last_lost_pacman = None
                self.pos_history.clear()
        if matches >= 3:        #drop current task to force re-evaluation if found oscillating
            active_task = self.cbba_agent.get_active_task()
            if active_task:
                self.cbba_agent.mark_unreachable(active_task.target_pos, self.frame)
            self.cbba_agent.bundle.clear()
            self.cbba_agent.path.clear()
            self.pos_history.clear()

    def is_agent_dead(self, gid: int) -> bool:
        if gid == self.gid:
            return self.dead
        if gid in self.dead_agents:
            return True
        return False

    def _check_liveness(self, all_ghosts):
        for gid in list(self.last_heartbeat.keys()):
            silence = self.frame - self.last_heartbeat[gid]
            if silence > HEARTBEAT_TIMEOUT:
                if self.known_agents.get(gid) != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    self._broadcast([("agent_lost", gid)], all_ghosts)

    def witness_death(self, dead_gid: int, all_ghosts: dict):
        """Called when this ghost directly witnesses a peer ghost getting eaten by Pacman."""
        self.dead_agents.add(dead_gid)
        self.known_agents[dead_gid] = "UNKNOWN"
        self.callout = f"Ghost {dead_gid} DOWN!"
        self.callout_timer = 60
        self._broadcast([("agent_dead", dead_gid)], all_ghosts)

    def _lidar_sweep(self, all_ghosts, player_pos, powered=False):
        directions = np.column_stack((_DX, _DY))
        if hasattr(self.world, 'batch_raycast'):
            hit_x, hit_y = self.world.batch_raycast((self.x, self.y), directions, max_dist=15.0)
            if len(hit_x) > 0:
                hits = set(zip(np.round(hit_y, 1), np.round(hit_x, 1)))
                new_walls = hits - self.lidar_memory
                if new_walls:
                    self.lidar_memory.update(new_walls)
                    self.belief_map.observe_walls_batch(list(new_walls))
                    wall_diffs = [("wall", w) for w in new_walls]
                    self._broadcast(wall_diffs, all_ghosts)
        visible_prm = []
        visible_belief_idxs = set()
        impassable_belief_nodes = []
        bm_arr = getattr(self.belief_map, '_open_arr', None)
        if bm_arr is not None and len(bm_arr) > 0:
            dx = bm_arr[:, 1] - self.x
            dy = bm_arr[:, 0] - self.y
            dist_sq = dx * dx + dy * dy
            valid_mask = dist_sq <= MAX_RAY_DIST_SQ
            if np.any(valid_mask):
                valid_nodes = bm_arr[valid_mask]
                valid_idxs = np.where(valid_mask)[0]
                valid_targets = np.column_stack((valid_nodes[:, 1], valid_nodes[:, 0]))
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid_targets, radius=0.0, step_size=0.5)
                if hasattr(self.belief_map, '_walkable_mask') and len(self.belief_map._walkable_mask) > 0:
                    walkable = self.belief_map._walkable_mask[valid_idxs]
                    visible_belief_idxs.update(valid_idxs[is_los & walkable].tolist())
                else:
                    disabled_nodes = getattr(self.belief_map, '_disabled_wall_nodes', set())
                    for idx, node, vis in zip(valid_idxs, valid_nodes, is_los):
                        if vis and tuple(node) not in disabled_nodes:
                            visible_belief_idxs.add(idx)
        if visible_belief_idxs:
            cells = self.belief_map._open_cells
            visible_prm = [cells[i] for i in visible_belief_idxs if i < len(cells)]
        pellet_diffs = []
        pellets_arr = getattr(self.world, 'pellets_arr', None)
        pellets_tup = getattr(self.world, 'pellets_tuples', None)
        if pellets_arr is not None and len(pellets_arr) > 0:
            dx = pellets_arr[:, 0] - self.x
            dy = pellets_arr[:, 1] - self.y
            dist_sq = dx * dx + dy * dy
            valid_mask = dist_sq <= MAX_RAY_DIST_SQ
            if np.any(valid_mask):
                valid = pellets_arr[valid_mask]
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid, radius=0.0, step_size=0.5)
                if pellets_tup is not None and len(pellets_tup) == len(pellets_arr):
                    valid_idxs = np.where(valid_mask)[0]
                    for idx, v in zip(valid_idxs, is_los):
                        if v:
                            pt = pellets_tup[idx]
                            if pt not in self.known_pellets:
                                self.known_pellets.add(pt)
                                pellet_diffs.append(("pellet", pt))
                else:
                    for p, v in zip(valid, is_los):
                        if v:
                            pt = find_closest_pellet(p, self.world, is_power=False)
                            if pt is None:
                                pt = tuple(p)
                            if pt not in self.known_pellets:
                                self.known_pellets.add(pt)
                                pellet_diffs.append(("pellet", pt))
        power_arr = getattr(self.world, 'power_pellets_arr', None)
        power_tup = getattr(self.world, 'power_pellets_tuples', None)
        if power_arr is not None and len(power_arr) > 0:
            dx = power_arr[:, 0] - self.x
            dy = power_arr[:, 1] - self.y
            dist_sq = dx * dx + dy * dy
            valid_mask = dist_sq <= MAX_RAY_DIST_SQ
            if np.any(valid_mask):
                valid = power_arr[valid_mask]
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid, radius=0.0, step_size=0.5)
                if power_tup is not None and len(power_tup) == len(power_arr):
                    valid_idxs = np.where(valid_mask)[0]
                    for idx, v in zip(valid_idxs, is_los):
                        if v:
                            pt = power_tup[idx]
                            if pt not in self.known_power_pellets:
                                self.known_power_pellets.add(pt)
                                pellet_diffs.append(("power", pt))
                else:
                    for p, v in zip(valid, is_los):
                        if v:
                            pt = find_closest_pellet(p, self.world, is_power=True)
                            if pt is None:
                                pt = tuple(p)
                            if pt not in self.known_power_pellets:
                                self.known_power_pellets.add(pt)
                                pellet_diffs.append(("power", pt))
        agent_diffs = []
        alive_ghosts = []
        alive_gids = []
        for gid, ghost in all_ghosts.items():
            if gid == self.gid: continue
            if getattr(ghost, 'dead', False):
                continue
            alive_ghosts.append(ghost)
            alive_gids.append(gid)
        if alive_ghosts:
            targets = np.array([[(g.x, g.y)] for g in alive_ghosts]).reshape(-1, 2)
            dx = targets[:, 0] - self.x
            dy = targets[:, 1] - self.y
            dists_sq = dx * dx + dy * dy
            valid_mask = dists_sq <= MAX_RAY_DIST_SQ
            if np.any(valid_mask):
                valid_gids = np.array(alive_gids)[valid_mask]
                valid_targets = targets[valid_mask]
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid_targets, radius=0.4, step_size=0.5)
                los_gids = valid_gids[is_los]
            else:
                los_gids = []
            for gid, ghost in zip(alive_gids, alive_ghosts):
                if gid in los_gids:
                    self.last_heartbeat[gid] = self.frame
                    old = self.known_agents.get(gid)
                    if old != (ghost.y, ghost.x):
                        self.known_agents[gid] = (ghost.y, ghost.x)
                        self._last_known_agent_pos[gid] = (ghost.y, ghost.x)
                        agent_diffs.append(("agent", gid, ghost.y, ghost.x))
            for gid, pos in list(self.known_agents.items()):
                if pos == "UNKNOWN" or gid in los_gids:
                    continue
                pr, pc = pos
                dist = math.hypot(pr - self.y, pc - self.x)
                if dist <= MAX_RAY_DIST and self.world.line_of_sight((self.x, self.y), (pc, pr), radius=0.4, step_size=0.5):
                    #direct visual confirmation that peer ghost is NOT at last known position
                    self.known_agents[gid] = "UNKNOWN"
                    if self.is_agent_dead(gid):
                        agent_diffs.append(("agent_dead", gid))
                    else:
                        agent_diffs.append(("agent_lost", gid))
        pacman_diff = None
        pr, pc = player_pos
        pac_d = math.hypot(pr - self.y, pc - self.x)
        if pac_d <= MAX_RAY_DIST and self.world.line_of_sight((self.x, self.y), (pc, pr), radius=0.4, step_size=0.5):
            if getattr(self, '_prev_seen_pacman', None) is not None:
                dy = pr - self._prev_seen_pacman[0]
                dx = pc - self._prev_seen_pacman[1]
                n = math.hypot(dx, dy)
                if n > 0.01:
                    self._player_dir = (dy / n, dx / n)
            self._prev_seen_pacman = (pr, pc)
            if self.known_pacman != (pr, pc) or self.pacman_powered != powered:
                self.known_pacman = (pr, pc)
                if powered and not self.pacman_powered:
                    self.pacman_power_timer = 40
                self.pacman_powered = powered
                if not powered: self.pacman_power_timer = 0
                self.pacman_last_seen = self.frame
                p_dir = getattr(self, '_player_dir', (0.0, 0.0))
                pacman_diff = ("pacman", pr, pc, powered, self.frame, p_dir[0], p_dir[1])
            else:
                self.pacman_last_seen = self.frame
        else:
            self._prev_seen_pacman = None
            if self.known_pacman is not None:
                kr, kc = self.known_pacman
                self.last_lost_pacman = (kr, kc)
                self.pacman_last_seen = self.frame 
                self.known_pacman = None
                pacman_diff = ("pacman_lost", kr, kc, self.frame)
        return [], visible_prm, agent_diffs, pacman_diff, pellet_diffs, visible_belief_idxs, impassable_belief_nodes

    def _update_lidar_memory(self, all_ghosts, player_pos, powered=False):
        _, visible_prm, agent_diffs, pacman_diff, pellet_diffs, visible_belief_idxs, impassable_belief_nodes = self._lidar_sweep(all_ghosts, player_pos, powered)
        diffs = []
        newly_discovered = 0
        stale_refreshed = 0.0
        diffs.extend(pellet_diffs)
        for n in visible_prm:
            last = self.prm_last_seen.get(n, -1)
            if last == -1:
                self.prm_known_count += 1
                newly_discovered += 1
            else:
                staleness = min(self.frame - last, 200) / 200.0
                if staleness > 0.25: stale_refreshed += staleness
            self.prm_last_seen[n] = self.frame
            diffs.append(("prm_refresh", n))
        diffs.extend(agent_diffs)
        if pacman_diff: 
            diffs.append(pacman_diff)
        pr, pc = player_pos
        pacman_in_los = (self.known_pacman is not None)
        pacman_just_lost = pacman_diff is not None and pacman_diff[0] == "pacman_lost"
        if pacman_in_los:
            pac_dir = (0, 0)
            if self.prev_pac_row >= 0:
                pac_dir = (pr - self.prev_pac_row, pc - self.prev_pac_col)
            cur_v = pac_dir
            p_v = getattr(self, '_prev_pac_dir', (0.0, 0.0))
            self._prev_pac_dir = pac_dir
            known_ghost_coords = [pos for gid, pos in self.known_agents.items() if pos != "UNKNOWN"]
            self.belief_map.observe((float(pr), float(pc)), pac_dir, current_vel=cur_v, prev_vel=p_v, 
                                known_ghosts=known_ghost_coords, known_pellets=self.known_pellets,
                                known_walls=self.lidar_memory, is_powered=self.pacman_powered)
            self.prev_pac_row, self.prev_pac_col = pr, pc
        elif pacman_just_lost:
            _, kr, kc, _ = pacman_diff
            self.belief_map.observe_lost((float(kr), float(kc)))
        #throttle belief diffusion to every BELIEF_DIFFUSE_EVERY frames and stagger by ghost ID
        if (self.frame + self.gid) % BELIEF_DIFFUSE_EVERY == 0:
            known_ghost_coords = [pos for gid, pos in self.known_agents.items() if pos != "UNKNOWN"]
            self.belief_map.diffuse((float(self.y), float(self.x)), self.known_pellets, self.known_power_pellets,
                                    ghost_positions=known_ghost_coords, is_powered=self.pacman_powered)
        pac_pos = (float(pr), float(pc)) if pacman_in_los else None
        self.belief_map.observe_clear(visible_belief_idxs, impassable_belief_nodes, pac_pos)
        #cleanup eaten pellets from memory only if within LOS (sensor-scoped eviction)
        world_pellets = getattr(self.world, 'pellet_set', None)
        if world_pellets is None:
            world_pellets = set(getattr(self.world, 'pellets', []))
        world_power_pellets = getattr(self.world, 'power_pellet_set', None)
        if world_power_pellets is None:
            world_power_pellets = set(getattr(self.world, 'power_pellets', []))
        if self.known_pellets and hasattr(self.world, 'batch_line_of_sight'):
            missing_pellets = [pt for pt in self.known_pellets if pt not in world_pellets]
            if missing_pellets:
                p_arr = np.array(missing_pellets, dtype=np.float32)
                dx = p_arr[:, 0] - self.x
                dy = p_arr[:, 1] - self.y
                dists_sq = dx * dx + dy * dy
                close_mask = dists_sq <= MAX_RAY_DIST_SQ
                if np.any(close_mask):
                    close_pts = p_arr[close_mask]
                    close_indices = np.where(close_mask)[0]
                    is_los = self.world.batch_line_of_sight((self.x, self.y), close_pts, radius=0.0, step_size=0.5)
                    for idx, los in zip(close_indices, is_los):
                        if los:
                            self.known_pellets.discard(missing_pellets[idx])
        if self.known_power_pellets and hasattr(self.world, 'batch_line_of_sight'):
            missing_power = [pt for pt in self.known_power_pellets if pt not in world_power_pellets]
            if missing_power:
                pow_arr = np.array(missing_power, dtype=np.float32)
                dx = pow_arr[:, 0] - self.x
                dy = pow_arr[:, 1] - self.y
                pow_dists_sq = dx * dx + dy * dy
                close_pow = pow_dists_sq <= MAX_RAY_DIST_SQ
                if np.any(close_pow):
                    close_pow_pts = pow_arr[close_pow]
                    close_pow_indices = np.where(close_pow)[0]
                    pow_los = self.world.batch_line_of_sight((self.x, self.y), close_pow_pts, radius=0.0, step_size=0.5)
                    for idx, los in zip(close_pow_indices, pow_los):
                        if los:
                            self.known_power_pellets.discard(missing_power[idx])
        self._last_visible_belief_idxs = visible_belief_idxs
        return diffs, newly_discovered, stale_refreshed

    def _broadcast(self, diffs, all_ghosts, msg_id=None, hop=0):
        if not diffs:
            return
        is_new_msg = msg_id is None
        if is_new_msg:
            msg_id = (self.gid, self.frame, self.seq)
            self.seq += 1
            cbba_payload   = self.cbba_agent.get_consensus_payload()
            belief_payload = self.belief_map.get_payload()
            diffs = list(diffs) + [("cbba", self.gid, cbba_payload), ("belief", self.gid, belief_payload)]
        self.seen_message_ids[msg_id] = True
        msg = {"id": msg_id, "diffs": diffs, "hop": hop}
        for ghost in all_ghosts.values():
            if ghost.gid == self.gid:
                continue
            dist = math.hypot(ghost.y - self.y, ghost.x - self.x)
            if dist <= RADIUS:
                ghost.message_queue.append(msg)
                if is_new_msg:
                    last = self.last_sync_frame.get(ghost.gid, -1)
                    if self.frame - last >= RESYNC_EVERY:
                        self.last_sync_frame[ghost.gid] = self.frame
                        ghost.last_sync_frame[self.gid] = self.frame
                        self._send_full_sync(ghost)
                        ghost._send_full_sync(self)

    def _send_full_sync(self, target_ghost):
        sync_diffs = []
        for n, last_seen in self.prm_last_seen.items():
            if last_seen != -1:
                sync_diffs.append(("prm_refresh", n))
        tgt_pellets = getattr(target_ghost, 'known_pellets', None)
        if tgt_pellets is None: tgt_pellets = set()
        for p in self.known_pellets:
            if p not in tgt_pellets:
                sync_diffs.append(("pellet", p)) 
        tgt_power = getattr(target_ghost, 'known_power_pellets', None)
        if tgt_power is None: tgt_power = set()
        for p in self.known_power_pellets:
            if p not in tgt_power:
                sync_diffs.append(("power", p))        
        tgt_lidar = getattr(target_ghost, 'lidar_memory', None)
        if tgt_lidar is None: tgt_lidar = set()
        for w in self.lidar_memory:
            if w not in tgt_lidar:
                sync_diffs.append(("wall", w))
        for dead_gid in self.dead_agents:
            sync_diffs.append(("agent_dead", dead_gid))
        for gid, pos in self.known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            elif pos is not None:
                sync_diffs.append(("agent", gid, pos[0], pos[1]))
        for gid, hb_frame in self.last_heartbeat.items():
            frames_ago = self.frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))
        if self.known_pacman is not None:
            p_dir = getattr(self, '_player_dir', (0.0, 0.0))
            sync_diffs.append(("pacman", self.known_pacman[0], self.known_pacman[1], self.pacman_powered, self.pacman_last_seen, p_dir[0], p_dir[1]))
        elif self.last_lost_pacman is not None and self.pacman_last_seen > -1:
            sync_diffs.append(("pacman_lost", self.last_lost_pacman[0], self.last_lost_pacman[1], self.pacman_last_seen))
        if sync_diffs:
            sync_id = ("sync", self.gid, target_ghost.gid, self.frame)
            self.seen_message_ids[sync_id] = True
            target_ghost.message_queue.append({"id": sync_id, "diffs": sync_diffs, "hop": 0})

    def _process_messages(self, all_ghosts):
        new_peer_walls = []
        for msg in self.message_queue:
            if msg["id"] in self.seen_message_ids:
                continue
            self.seen_message_ids[msg["id"]] = True
            hop  = msg.get("hop", 0)
            relay_diffs = []
            for diff in msg["diffs"]:
                dtype = diff[0]
                if dtype == "prm_refresh":
                    _, n = diff
                    old = self.prm_last_seen.get(n, -1)
                    if old < self.frame - MEMORY_FRAMES:
                        if old == -1:
                            self.prm_known_count += 1
                        self.prm_last_seen[n] = self.frame
                        relay_diffs.append(diff)
                elif dtype == "agent":
                    _, gid, r, c = diff
                    if gid == self.gid:
                        continue
                    old = self.known_agents.get(gid)
                    if old != (r, c):
                        self.known_agents[gid] = (r, c)
                        self._last_known_agent_pos[gid] = (r, c)
                        relay_diffs.append(diff)
                elif dtype == "pellet":
                    _, p = diff
                    if p not in self.known_pellets:
                        self.known_pellets.add(p)
                        relay_diffs.append(diff)
                elif dtype == "power":
                    _, p = diff
                    if p not in self.known_power_pellets and p not in self.known_pellets:
                        self.known_power_pellets.add(p)
                        relay_diffs.append(diff)
                elif dtype == "power_eaten":
                    _, p = diff
                    self.known_power_pellets.discard(p)
                    relay_diffs.append(diff)
                elif dtype == "wall":
                    _, w = diff
                    if w not in self.lidar_memory:
                        self.lidar_memory.add(w)
                        new_peer_walls.append(w)
                        relay_diffs.append(diff)
                elif dtype == "agent_dead":
                    _, gid = diff
                    self.dead_agents.add(gid)
                    self.known_agents[gid] = "UNKNOWN"
                    relay_diffs.append(diff)
                elif dtype == "agent_lost":
                    _, gid = diff
                    if gid == self.gid:
                        continue
                    if gid in self.dead_agents:
                        self.known_agents[gid] = "UNKNOWN"
                    elif self.frame - self.last_heartbeat.get(gid, -1) > HEARTBEAT_TIMEOUT:
                        if self.known_agents.get(gid) != "UNKNOWN":
                            self.known_agents[gid] = "UNKNOWN"
                            relay_diffs.append(diff)
                elif dtype == "heartbeat":
                    _, gid, r, c, origin_frame = diff
                    if gid == self.gid:
                        continue
                    existing = self.last_heartbeat.get(gid, -1)
                    if origin_frame > existing:
                        self.last_heartbeat[gid] = origin_frame
                    self.dead_agents.discard(gid)
                    if r != 0 or c != 0:
                        old = self.known_agents.get(gid)
                        if old != (r, c):
                            self.known_agents[gid] = (r, c)
                            self._last_known_agent_pos[gid] = (r, c)
                            relay_diffs.append(("agent", gid, r, c))
                    relay_diffs.append(diff)
                elif dtype == "hb_sync":
                    _, gid, frames_ago = diff
                    if gid == self.gid:
                        continue
                    reconstructed = self.frame - frames_ago
                    existing = self.last_heartbeat.get(gid, -1)
                    if reconstructed > existing:
                        self.last_heartbeat[gid] = reconstructed
                        relay_diffs.append(diff)
                elif dtype == "pacman":
                    r, c, powered, obs_frame = diff[1], diff[2], diff[3], diff[4]
                    p_vy = diff[5] if len(diff) > 5 else 0.0
                    p_vx = diff[6] if len(diff) > 6 else 0.0
                    if obs_frame > self.pacman_last_seen:
                        self.known_pacman     = (r, c)
                        if (p_vy != 0.0 or p_vx != 0.0):
                            self._player_dir = (p_vy, p_vx)
                        if powered and not self.pacman_powered:
                            self.pacman_power_timer = 40
                        self.pacman_powered   = powered
                        if not powered:
                            self.pacman_power_timer = 0
                        self.pacman_last_seen = obs_frame
                        self.last_lost_pacman = None  #new sighting clears lost marker
                        relay_diffs.append(diff)
                elif dtype == "pacman_lost":
                    _, lr, lc, obs_frame = diff
                    if obs_frame > self.pacman_last_seen:
                        if self.known_pacman == (lr, lc):
                            self.known_pacman = None
                        self.last_lost_pacman = (lr, lc)
                        self.pacman_last_seen = obs_frame
                        relay_diffs.append(diff)
                elif dtype == "cbba":
                    _, sender_gid, payload = diff
                    if sender_gid == self.gid:
                        continue
                    self.cbba_agent.receive_consensus(sender_gid, payload["y"], payload["z"], payload["s"], self.frame, payload.get("meta"))
                elif dtype == "belief":
                    _, sender_gid, payload = diff
                    if sender_gid == self.gid:
                        continue
                    self.belief_map.merge(sender_gid, payload, self.frame)
            if relay_diffs and hop < 2:
                MAX_RELAY_SIZE = 50
                for idx, i in enumerate(range(0, len(relay_diffs), MAX_RELAY_SIZE)):
                    chunk = relay_diffs[i : i + MAX_RELAY_SIZE]
                    if idx == 0:
                        chunk_msg_id = msg["id"]
                    else:
                        chunk_msg_id = tuple(list(msg["id"]) + [f"chunk_{idx}"])                        
                    self._broadcast(chunk, all_ghosts, msg_id=chunk_msg_id, hop=hop+1)
        if new_peer_walls:
            self.belief_map.observe_walls_batch(new_peer_walls)
        self.message_queue.clear()
        self.belief_map._ensure_initialised()
        #rolling prune - keep newest 250, discarding rest post 500 messages
        if len(self.seen_message_ids) > 500:
            to_remove = list(self.seen_message_ids)[:250]
            for item in to_remove:
                self.seen_message_ids.pop(item, None)

    def kill(self):
        self.dead = True
        self.dead_agents.add(self.gid)

    def draw(self, surf, scale=None, offset_x=0, offset_y=0):
        if scale is None:
            scale = CELL
        if self.dead:
            return
        x = int(self.x * scale) + offset_x
        y = int(self.y * scale) + offset_y
        r = scale // 2 - 2
        color = self.color
        pygame.draw.circle(surf, color, (x, y - 2), r)
        pygame.draw.rect(surf, color, (x - r, y - 2, r * 2, r + 2))
        wave_r = max(2, scale // 8)
        for i in range(3):
            wx = x - r + wave_r + i * (r * 2 - wave_r * 2) // 2
            wy = y + r - 1
            pygame.draw.circle(surf, BLACK, (wx, wy), wave_r)
        speed = math.hypot(self.vx, self.vy)
        if speed > 0.01:
            dx_n = self.vx / speed
            dy_n = self.vy / speed
        else:
            dx_n, dy_n = 1.0, 0.0
        eye_sep = max(3, scale // 5)
        pupil_off = max(1, scale // 12)
        eye_r = max(2, scale // 7)
        pupil_r = max(1, eye_r - 1)
        pygame.draw.circle(surf, WHITE, (x - eye_sep, y - eye_sep // 2), eye_r)
        pygame.draw.circle(surf, WHITE, (x + eye_sep, y - eye_sep // 2), eye_r)
        px_off = int(dx_n * pupil_off)
        py_off = int(dy_n * pupil_off)
        pygame.draw.circle(surf, BLACK, (x - eye_sep + px_off, y - eye_sep // 2 + py_off), pupil_r)
        pygame.draw.circle(surf, BLACK, (x + eye_sep + px_off, y - eye_sep // 2 + py_off), pupil_r)
        if getattr(self, 'callout_timer', 0) > 0 and getattr(self, 'callout', None):
            font = getattr(self, '_callout_font', None)
            if font is None:
                try:
                    font = pygame.font.SysFont('Arial', 10, bold=True)
                except Exception:
                    font = pygame.font.Font(None, 12)
                self._callout_font = font
            txt_surf = font.render(self.callout, True, (255, 255, 50))
            tw, th = txt_surf.get_size()
            bubble_rect = pygame.Rect(x - tw // 2 - 4, y - r - th - 8, tw + 8, th + 4)
            pygame.draw.rect(surf, (180, 20, 20), bubble_rect, border_radius=3)
            pygame.draw.rect(surf, (255, 255, 255), bubble_rect, width=1, border_radius=3)
            surf.blit(txt_surf, (x - tw // 2, y - r - th - 6))