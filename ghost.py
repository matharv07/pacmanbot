import pygame
import random
import math
from collections import deque
import numpy as np
from pathfinder import dijkstra_multi, next_step
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

RADIUS            = 12
RAY_COUNT         = 90
MAX_RAY_DIST      = 10
UNKNOWN           = -1
MEMORY_FRAMES     = 10
HEARTBEAT_EVERY   = 5
HEARTBEAT_TIMEOUT = 25
RESYNC_EVERY      = 50
OSCILLATION_WINDOW = 8   #position history length to prevent oscillations

_ANGLES = np.linspace(0, 2*math.pi, RAY_COUNT, endpoint=False)
_DX = np.cos(_ANGLES) * 0.5
_DY = np.sin(_ANGLES) * 0.5

class Ghost:
    def __init__(self, gid, grid, pos, color, player_start, world=None):
        self.gid = gid
        self.grid = grid
        self.world = world
        self.radius = 0.4
        self.x, self.y = float(pos[1]), float(pos[0])
        self.prev_x, self.prev_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0
        self.max_speed = 0.5
        self.target_cell = pos
        self.color = color
        self.dead = False
        self.in_fallback_mode = False
        self.move_every = 1
        self.last_dir = random.choice(DIRS)
        rows = len(grid)
        cols = len(grid[0])
        self.lidar_memory = set()
        self.known_pellets = set()
        self.known_power_pellets = set()
        self.prm_last_seen = {n: -1 for n in getattr(world, 'prm_nodes', [])}
        self.frame = 0
        self.message_queue = []
        self.seen_message_ids = {}
        self.seq = 0
        self.known_agents = {}                  #(row, col) | UNKNOWN for dead/out of reach agents
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
            import numpy as np
            p_start = (float(np.float32(player_start[0])), float(np.float32(player_start[1])))
        self.belief_map = BeliefMap(gid, self.world, pacman_start=p_start)
        self.belief_map.init_full_topology(getattr(self.world, 'prm_graph', {}))
        self._proximity_channel_cache = None
        self._proximity_channel_frame = -1
        self._proximity_channel_target = None
        self._last_synced_map: dict[int, np.ndarray] = {}   # per-peer snapshot for delta sync
        self._tail_pacman_remaining = 0         #post-pop number of ghosts that will be tailing
        self._personal_map_cache = None         #cached personal map; invalidated on observation changes
        self._personal_map_dirty = True

    @property
    def personal_map(self):
        """Cached discrete 2D map from continuous memory for the RL CNN."""
        if not self._personal_map_dirty and self._personal_map_cache is not None:
            return self._personal_map_cache
        rows = len(self.grid)
        cols = len(self.grid[0])
        pmap = np.full((rows, cols), -1, dtype=np.int8)
        
        # Mark walls from lidar hits
        if self.lidar_memory:
            pts = np.array(list(self.lidar_memory), dtype=np.float32)
            rs = pts[:, 0].astype(np.int32)
            cs = pts[:, 1].astype(np.int32)
            valid = (rs >= 0) & (rs < rows) & (cs >= 0) & (cs < cols)
            pmap[rs[valid], cs[valid]] = 1
                
        # Fill in empty spaces from PRM visibility
        for n, last_seen in self.prm_last_seen.items():
            if last_seen != -1:
                r, c = int(n[0]), int(n[1])
                if 0 <= r < rows and 0 <= c < cols and pmap[r, c] == -1:
                    pmap[r, c] = 0
                    
        # Mark known pellets
        for px, py in self.known_pellets:
            r, c = int(py), int(px)
            if 0 <= r < rows and 0 <= c < cols:
                pmap[r, c] = 2
                
        # Mark known power pellets
        for px, py in self.known_power_pellets:
            r, c = int(py), int(px)
            if 0 <= r < rows and 0 <= c < cols:
                pmap[r, c] = 3
        
        self._personal_map_cache = pmap
        self._personal_map_dirty = False
        return pmap

    def update(self, player_pos, powered, all_ghosts, skip_movement=False, speed_mult=1.0):
        self.frame += 1
        if getattr(self, 'pacman_power_timer', 0) > 0:
            self.pacman_power_timer -= 1
            if self.pacman_power_timer <= 0:
                self.pacman_powered = False
        newly_discovered = 0
        stale_refreshed = 0.0
        if self.dead:
            return newly_discovered, stale_refreshed
        self._check_liveness(all_ghosts)
        diffs, newly_discovered, stale_refreshed = self._update_lidar_memory(all_ghosts, player_pos, powered)
        if self.frame % HEARTBEAT_EVERY == 0:
            diffs.append(("heartbeat", self.gid, int(self.y), int(self.x), self.frame))
        self._broadcast(diffs, all_ghosts)
        self._process_messages(all_ghosts)
        self.belief_map.update_safety_map(self.known_agents, self.frame, powered=self.pacman_powered)
        if skip_movement:
            if skip_movement:
                self.pos_history.append((self.y, self.x))
                self._check_oscillation()
            return newly_discovered, stale_refreshed
        active_task = self.cbba_agent.step(self, self.frame)
        if active_task and self.pacman_powered and active_task.task_type == TaskType.HUNT:
            active_task = None
        if active_task is not None and (int(self.y), int(self.x)) == active_task.target_pos:
            key = (int(active_task.task_type), active_task.target_pos, getattr(active_task, 'owner', -1))
            if key in self.cbba_agent.path: self.cbba_agent.path.remove(key)
            if key in self.cbba_agent.bundle: self.cbba_agent.bundle.remove(key)
            active_task = None
        desired_vx = 0.0
        desired_vy = 0.0
        moved = False
        #power pellet area denial
        if not moved and not self.pacman_powered and self.known_pacman:
            pr, pc = self.known_pacman
            for p_pos in (self.world.power_pellets if hasattr(self, 'world') and self.world else []):
                p_r, p_c = float(p_pos[0]), float(p_pos[1])
                dist_pac_to_power = abs(pr - p_r) + abs(pc - p_c)
                if dist_pac_to_power < 8:
                    my_dist = abs(self.y - p_r) + abs(self.x - p_c)
                    is_closest = True
                    for _gid, pos in self.known_agents.items():
                        if pos != "UNKNOWN":
                            other_dist = abs(pos[0] - p_r) + abs(pos[1] - p_c)
                            if other_dist < my_dist:
                                is_closest = False
                                break
                    if is_closest:
                        active_task = type('DummyTask', (), {'target_pos': (p_r, p_c), 'task_type': -1})()
                        break
        #chase override
        CHASE_RADIUS = 5.0
        GRAB_DIST = 2.0
        dist_pac = 999
        if not moved and not self.pacman_powered and self.known_pacman:
            pr, pc = self.known_pacman
            pac_y, pac_x = pr, pc
            dist_pac = math.hypot(pac_y - self.y, pac_x - self.x)
            if dist_pac < CHASE_RADIUS:
                has_los = True
                if self.world and hasattr(self.world, 'line_of_sight'):
                    has_los = self.world.line_of_sight((self.x, self.y), (pac_x, pac_y), radius=self.radius, step_size=0.5)
                closest_to_pac = True
                nearby_ghosts = 1
                for _gid, pos in self.known_agents.items():
                    if pos != "UNKNOWN":
                        d_other = math.hypot(pos[0] + 0.5 - pac_y, pos[1] + 0.5 - pac_x)
                        if d_other < dist_pac:
                            closest_to_pac = False
                        if math.hypot(pos[0] - self.y, pos[1] - self.x) <= 6:
                            nearby_ghosts += 1
                self._tail_pacman_remaining = 2 if nearby_ghosts <= 2 else 0
                if closest_to_pac and has_los:
                    self.cbba_agent.bundle.clear()
                    self.cbba_agent.path.clear()
                    active_task = None
                    if dist_pac > 0:
                        desired_vx = (pac_x - self.x) / dist_pac
                        desired_vy = (pac_y - self.y) / dist_pac
                    moved = True
                else:
                    tr, tc = pr, pc
                    if not closest_to_pac:
                        dr = pr - self.prev_pac_row if self.prev_pac_row >= 0 else 0
                        dc = pc - self.prev_pac_col if self.prev_pac_col >= 0 else 0
                        dr = max(-1, min(1, dr))
                        dc = max(-1, min(1, dc))
                        if abs(dr) + abs(dc) > 0:
                            tr = max(0, min(len(self.grid)-1, pr + dr * 4))
                            tc = max(0, min(len(self.grid[0])-1, pc + dc * 4))
                    active_task = type('DummyTask', (), {'target_pos': (int(tr), int(tc)), 'task_type': -1})()
        #power pellet grab override
        if not moved and (not self.known_pacman or self.pacman_powered or dist_pac > CHASE_RADIUS):
            r, c = int(self.y), int(self.x)
            best_power = None
            best_pd = float('inf')
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[0]) and self.grid[nr][nc] == POWER:
                        pd = math.hypot((nr+0.5)-self.y, (nc+0.5)-self.x)
                        if pd < GRAB_DIST and pd < best_pd:
                            best_power = (nr+0.5, nc+0.5)
                            best_pd = pd
            if best_power is not None:
                dx, dy = best_power[1] - self.x, best_power[0] - self.y
                d = math.hypot(dx, dy)
                if d > 0:
                    desired_vx = dx / d
                    desired_vy = dy / d
                moved = True
                if hasattr(self, '_committed_path'):
                    self._committed_path = []
        #tailing
        if not moved and self._tail_pacman_remaining > 0:
            if self.known_pacman and not self.pacman_powered:
                pr, pc = self.known_pacman
                pac_y, pac_x = float(pr), float(pc)
                dist_pac = math.hypot(pac_y - self.y, pac_x - self.x)
                if dist_pac > 0:
                    desired_vx = (pac_x - self.x) / dist_pac
                    desired_vy = (pac_y - self.y) / dist_pac
                moved = True
                self._tail_pacman_remaining -= 1
                if hasattr(self, '_committed_path'):
                    self._committed_path = []
            else:
                self._tail_pacman_remaining = 0
        #belief-map coordinated search
        if not moved and self.known_pacman is None and active_task is None:
            if self.belief_map._initialised and self.belief_map._open_cells:
                probs = self.belief_map._b_flat.tolist()
                if probs:
                    max_p = max(probs)
                    if max_p > 1e-4:
                        best_idx = probs.index(max_p)
                        best_r, best_c = self.belief_map._open_cells[best_idx]
                        active_task = type('DummyTask', (), {'target_pos': (best_r, best_c), 'task_type': -1})()
        #normal task execution
        if not moved and active_task is not None:
            target = active_task.target_pos
            if getattr(self, '_committed_target', None) != target or not getattr(self, '_committed_path', []):
                from pathfinder import astar
                full_path = astar(self.world, (float(self.y), float(self.x)), target)
                if len(full_path) >= 2:
                    self._committed_path = full_path[1:]
                    self._committed_target = target
                else:
                    self._committed_path = []
            if hasattr(self, '_committed_path') and self._committed_path:
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
        #fallback
        if not moved:
            if hasattr(self, '_committed_path'):
                self._committed_path = []
            if random.random() < 0.1 or (self.vx == 0 and self.vy == 0):
                angle = random.uniform(0, 2*math.pi)
                desired_vx = math.cos(angle)
                desired_vy = math.sin(angle)
            else:
                cur_speed = math.hypot(self.vx, self.vy)
                if cur_speed > 0:
                    desired_vx = self.vx / cur_speed
                    desired_vy = self.vy / cur_speed
        #context steering and momentum
        best_score = -float('inf')
        best_vx, best_vy = desired_vx, desired_vy
        if desired_vx != 0.0 or desired_vy != 0.0:
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
            if self.world and hasattr(self.world, 'batch_is_passable'):
                passable = self.world.batch_is_passable(cc_grid.flatten(), cr_grid.flatten(), self.radius).reshape((num_rays, n_steps))
            else:
                r_c = cr_grid.astype(int)
                c_c = cc_grid.astype(int)
                valid = (r_c >= 0) & (r_c < len(self.grid)) & (c_c >= 0) & (c_c < len(self.grid[0]))
                safe_r = np.where(valid, r_c, 0)
                safe_c = np.where(valid, c_c, 0)
                grid_arr = np.array(self.grid)
                cells = grid_arr[safe_r, safe_c]
                passable = valid & (cells != WALL)
            hit_mask = ~passable
            hit_indices = np.argmax(hit_mask, axis=1)
            has_hit = np.any(hit_mask, axis=1)
            hit_fracs = (hit_indices + 1) / n_steps
            ray_penalties = np.where(has_hit, 1000.0 / hit_fracs, 0.0)
            interests = 1.5 * (ray_vx_arr * desired_vx + ray_vy_arr * desired_vy)
            hysteresis = 0.3 * (ray_vx_arr * cur_vx_norm + ray_vy_arr * cur_vy_norm)
            scores = interests + hysteresis - ray_penalties
            best_idx = np.argmax(scores)
            best_vx, best_vy = ray_vx_arr[best_idx], ray_vy_arr[best_idx]
        target_vy = best_vy * self.max_speed * speed_mult
        target_vx = best_vx * self.max_speed * speed_mult
        smooth_vy = self.vy * 0.7 + target_vy * 0.3
        smooth_vx = self.vx * 0.7 + target_vx * 0.3
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
        if self.world and hasattr(self.world, 'resolve_collision'):
            steps = max(1, int(math.ceil(math.hypot(self.vx, self.vy) / 0.2)))
            if steps > 0:
                step_vx = self.vx / steps
                step_vy = self.vy / steps
                for _ in range(steps):
                    self.x += step_vx
                    self.y += step_vy
                    self.x, self.y = self.world.resolve_collision(self.x, self.y, self.radius, max_iters=3)
                    self.path_this_frame.append((self.x, self.y))
        else:
            self.x += self.vx
            self.y += self.vy
            self.x = max(self.radius, min(len(self.grid[0]) - self.radius, self.x))
            self.y = max(self.radius, min(len(self.grid) - self.radius, self.y))
        r, c = int(self.y), int(self.x)
        if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]):
            if self.grid[r][c] == POWER:
                self.grid[r][c] = PELLET
                pt = (float(c) + 0.5, float(r) + 0.5)
                if getattr(self, 'world', None):
                    if pt in self.world.power_pellets:
                        self.world.power_pellets.remove(pt)
                    if pt not in self.world.pellets:
                        self.world.pellets.append(pt)
                    if hasattr(self.world, '_update_pellet_arrays'):
                        self.world._update_pellet_arrays()
                # Manually update local memory and force network sync
                p_tup = pt
                self.known_power_pellets = {p for p in self.known_power_pellets if int(p[0]) != c or int(p[1]) != r}
                if p_tup not in self.known_pellets:
                    self.known_pellets.add(p_tup)
                    # broadcast instantly via the actual network method so all peers get it
                    self._broadcast([("pellet", p_tup)], all_ghosts)
        self.pos_history.append((self.y, self.x))
        self._check_oscillation()
        return newly_discovered, stale_refreshed

    def _check_oscillation(self):
        if len(self.pos_history) < OSCILLATION_WINDOW:
            return
        cur_y, cur_x = self.y, self.x
        tol = 0.3  # tolerance for float coordinate comparison
        matches = sum(1 for py, px in self.pos_history if abs(py - cur_y) < tol and abs(px - cur_x) < tol)
        if matches >= 2:
            if self.known_pacman is None and self.last_lost_pacman is not None:
                self.last_lost_pacman = None
                self.pos_history.clear()
        if matches >= 3:
            #drop current task to force re-evaluation if found oscillating
            self.cbba_agent.bundle.clear()
            self.cbba_agent.path.clear()
            self.pos_history.clear()

    def _check_liveness(self, all_ghosts):
        for gid in list(self.last_heartbeat.keys()):
            if self.frame - self.last_heartbeat[gid] > HEARTBEAT_TIMEOUT:
                if self.known_agents.get(gid) != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    self._broadcast([("agent_lost", gid)], all_ghosts)

    def _lidar_sweep(self, all_ghosts, player_pos, powered=False):
        visible_prm = []
        prm_arr = getattr(self.world, 'prm_nodes_arr', None)
        if prm_arr is not None and len(prm_arr) > 0:
            dx = prm_arr[:, 1] - self.x
            dy = prm_arr[:, 0] - self.y
            dist = np.hypot(dx, dy)
            valid_mask = dist <= MAX_RAY_DIST
            if np.any(valid_mask):
                valid_nodes = prm_arr[valid_mask]
                valid_targets = np.column_stack((valid_nodes[:, 1], valid_nodes[:, 0]))
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid_targets, radius=0.4)
                visible_prm = [tuple(n) for n, vis in zip(valid_nodes, is_los) if vis]
        directions = np.column_stack((_DX, _DY))
        hit_x, hit_y = self.world.batch_raycast((self.x, self.y), directions, max_dist=MAX_RAY_DIST)
        lidar_hits = set(zip(np.round(hit_y, 2), np.round(hit_x, 2)))
        
        pellet_diffs = []
        pellets_arr = getattr(self.world, 'pellets_arr', None)
        if pellets_arr is not None and len(pellets_arr) > 0:
            dx = pellets_arr[:, 0] - self.x
            dy = pellets_arr[:, 1] - self.y
            dist = np.hypot(dx, dy)
            valid = pellets_arr[dist <= MAX_RAY_DIST]
            if len(valid) > 0:
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid, radius=0)
                for p, v in zip(valid, is_los):
                    if v and tuple(p) not in self.known_pellets:
                        self.known_pellets.add(tuple(p))
                        pellet_diffs.append(("pellet", tuple(p)))
                        
        power_arr = getattr(self.world, 'power_pellets_arr', None)
        if power_arr is not None and len(power_arr) > 0:
            dx = power_arr[:, 0] - self.x
            dy = power_arr[:, 1] - self.y
            dist = np.hypot(dx, dy)
            valid = power_arr[dist <= MAX_RAY_DIST]
            if len(valid) > 0:
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid, radius=0)
                for p, v in zip(valid, is_los):
                    if v and tuple(p) not in self.known_power_pellets:
                        self.known_power_pellets.add(tuple(p))
                        pellet_diffs.append(("power", tuple(p)))
        agent_diffs = []
        alive_ghosts = []
        alive_gids = []
        for gid, ghost in all_ghosts.items():
            if gid == self.gid: continue
            if getattr(ghost, 'dead', False):
                last_known = self.known_agents.get(gid)
                if last_known is not None and last_known != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    agent_diffs.append(("agent_lost", gid))
                continue
            alive_ghosts.append(ghost)
            alive_gids.append(gid)
            
        if alive_ghosts:
            targets = np.array([[(g.x, g.y)] for g in alive_ghosts]).reshape(-1, 2)
            dx = targets[:, 0] - self.x
            dy = targets[:, 1] - self.y
            dists = np.hypot(dx, dy)
            valid_mask = dists <= MAX_RAY_DIST
            if np.any(valid_mask):
                valid_gids = np.array(alive_gids)[valid_mask]
                valid_targets = targets[valid_mask]
                is_los = self.world.batch_line_of_sight((self.x, self.y), valid_targets, radius=0.4)
                los_gids = valid_gids[is_los]
            else:
                los_gids = []
                
            for gid, ghost in zip(alive_gids, alive_ghosts):
                if gid in los_gids:
                    old = self.known_agents.get(gid)
                    if old != (ghost.y, ghost.x):
                        self.known_agents[gid] = (ghost.y, ghost.x)
                        agent_diffs.append(("agent", gid, ghost.y, ghost.x))
        pacman_diff = None
        pr, pc = player_pos
        pac_d = math.hypot(pr - self.y, pc - self.x)
        if pac_d <= MAX_RAY_DIST and self.world.line_of_sight((self.x, self.y), (pc, pr), radius=0.4):
            if self.known_pacman != (pr, pc) or self.pacman_powered != powered:
                self.known_pacman = (pr, pc)
                if powered and not self.pacman_powered:
                    self.pacman_power_timer = 40
                self.pacman_powered = powered
                if not powered: self.pacman_power_timer = 0
                self.pacman_last_seen = self.frame
                pacman_diff = ("pacman", pr, pc, powered, self.frame)
            else:
                self.pacman_last_seen = self.frame
        else:
            if self.known_pacman is not None:
                kr, kc = self.known_pacman
                self.last_lost_pacman = (kr, kc)
                self.pacman_last_seen = self.frame 
                self.known_pacman = None
                pacman_diff = ("pacman_lost", kr, kc, self.frame)
        return lidar_hits, visible_prm, agent_diffs, pacman_diff, pellet_diffs

    def _update_lidar_memory(self, all_ghosts, player_pos, powered=False):
        lidar_hits, visible_prm, agent_diffs, pacman_diff, pellet_diffs = self._lidar_sweep(all_ghosts, player_pos, powered)
        diffs = []
        newly_discovered = 0
        stale_refreshed = 0.0
        self._personal_map_dirty = True  # invalidate cached personal_map
        
        diffs.extend(pellet_diffs)
        
        for pt in lidar_hits:
            if pt not in self.lidar_memory:
                newly_discovered += 1
                self.lidar_memory.add(pt)
                diffs.append(("wall_hit", pt))
                
        for n in visible_prm:
            last = self.prm_last_seen.get(n, -1)
            if last != -1:
                staleness = min(self.frame - last, 200) / 200.0
                if staleness > 0.25: stale_refreshed += staleness
            self.prm_last_seen[n] = self.frame
            diffs.append(("prm_refresh", n))
        

        diffs.extend(agent_diffs)
        if pacman_diff: diffs.append(pacman_diff)
        
        pr, pc = player_pos
        pacman_in_los = (self.known_pacman is not None)
        pacman_just_lost = pacman_diff is not None and pacman_diff[0] == "pacman_lost"
        
        if pacman_in_los:
            pac_dir = (0, 0)
            if self.prev_pac_row >= 0:
                pac_dir = (pr - self.prev_pac_row, pc - self.prev_pac_col)
            self.belief_map.observe((float(pr), float(pc)), pac_dir)
            self.prev_pac_row, self.prev_pac_col = pr, pc
        elif pacman_just_lost:
            _, kr, kc, _ = pacman_diff
            self.belief_map.observe_lost((float(kr), float(kc)))
            
        self.belief_map.diffuse((float(self.y), float(self.x)))
        pac_pos = (float(pr), float(pc)) if pacman_in_los else None
        self.belief_map.observe_clear((float(self.y), float(self.x)), pac_pos)
        
        # Cleanup eaten pellets from memory
        self.known_pellets = {p for p in self.known_pellets if p in getattr(self.world, 'pellets', [])}
        self.known_power_pellets = {p for p in self.known_power_pellets if p in getattr(self.world, 'power_pellets', [])}
        
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
        # sync missing lidar hits (very naive full sync, could be optimized)
        for pt in self.lidar_memory:
            if pt not in target_ghost.lidar_memory:
                sync_diffs.append(("wall_hit", pt))
        # sync prm timestamps
        for n, last_seen in self.prm_last_seen.items():
            if last_seen != -1:
                sync_diffs.append(("prm_refresh", n))
                
        # sync known pellets
        for p in self.known_pellets:
            if p not in getattr(target_ghost, 'known_pellets', set()):
                sync_diffs.append(("pellet", p))
        for p in self.known_power_pellets:
            if p not in getattr(target_ghost, 'known_power_pellets', set()):
                sync_diffs.append(("power", p))
                
        for gid, pos in self.known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            elif pos is not None:
                sync_diffs.append(("agent", gid, pos[0], pos[1]))
        for gid, hb_frame in self.last_heartbeat.items():
            frames_ago = self.frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))
        if self.known_pacman is not None:
            sync_diffs.append(("pacman", self.known_pacman[0], self.known_pacman[1], self.pacman_powered, self.pacman_last_seen))
        elif self.last_lost_pacman is not None and self.pacman_last_seen > -1:
            sync_diffs.append(("pacman_lost", self.last_lost_pacman[0], self.last_lost_pacman[1], self.pacman_last_seen))
        if sync_diffs:
            sync_id = ("sync", self.gid, target_ghost.gid, self.frame)
            self.seen_message_ids[sync_id] = True
            target_ghost.message_queue.append({"id": sync_id, "diffs": sync_diffs, "hop": 0})

    def _process_messages(self, all_ghosts):
        for msg in self.message_queue:
            if msg["id"] in self.seen_message_ids:
                continue
            self.seen_message_ids[msg["id"]] = True
            hop  = msg.get("hop", 0)
            relay_diffs = []
            for diff in msg["diffs"]:
                dtype = diff[0]
                if dtype == "wall_hit":
                    _, pt = diff
                    if pt not in self.lidar_memory:
                        self.lidar_memory.add(pt)
                        relay_diffs.append(diff)
                elif dtype == "prm_refresh":
                    _, n = diff
                    if self.prm_last_seen.get(n, -1) < self.frame - MEMORY_FRAMES:
                        self.prm_last_seen[n] = self.frame
                        relay_diffs.append(diff)
                elif dtype == "agent":
                    _, gid, r, c = diff
                    if gid == self.gid:
                        continue
                    old = self.known_agents.get(gid)
                    if old != (r, c):
                        self.known_agents[gid] = (r, c)
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
                elif dtype == "agent_lost":
                    _, gid = diff
                    if gid == self.gid:
                        continue
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
                    if r != 0 or c != 0:
                        old = self.known_agents.get(gid)
                        if old != (r, c):
                            self.known_agents[gid] = (r, c)
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
                    _, r, c, powered, obs_frame = diff
                    if obs_frame > self.pacman_last_seen:
                        self.known_pacman     = (r, c)
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
                    changed = self.cbba_agent.receive_consensus(sender_gid, payload["y"], payload["z"], payload["s"], self.frame)
                    if changed:
                        relay_diffs.append(diff)
                elif dtype == "belief":
                    _, sender_gid, payload = diff
                    if sender_gid == self.gid:
                        continue
                    self.belief_map.merge(sender_gid, payload, self.frame)
                    relay_diffs.append(diff)  #always relay — belief spreads like heartbeats
            if relay_diffs:
                MAX_RELAY_SIZE = 50
                for idx, i in enumerate(range(0, len(relay_diffs), MAX_RELAY_SIZE)):
                    chunk = relay_diffs[i : i + MAX_RELAY_SIZE]
                    if idx == 0:
                        chunk_msg_id = msg["id"]
                    else:
                        chunk_msg_id = tuple(list(msg["id"]) + [f"chunk_{idx}"])                        
                    self._broadcast(chunk, all_ghosts, msg_id=chunk_msg_id, hop=hop+1)
        self.message_queue.clear()
        self.belief_map._ensure_initialised()  #make sure belief map is ready before we try to prune messages
        #rolling prune - keep newest 250, discarding rest post 500 messages
        if len(self.seen_message_ids) > 500:
            to_remove = list(self.seen_message_ids)[:250]
            for item in to_remove:
                self.seen_message_ids.pop(item, None)

    def kill(self):
        self.dead = True

    def draw(self, surf, scale=None, offset_x=0, offset_y=0):
        if scale is None:
            scale = CELL
        if self.dead:
            return
        x = int(self.x * scale) + offset_x
        y = int(self.y * scale) + offset_y
        r = scale // 2 - 2
        #ghosts always keep their original color — only Pacman turns blue
        #(README rule #4: ghosts observe the colour change visually, they don't change themselves)
        color = self.color
        pygame.draw.circle(surf, color, (x, y - 2), r)
        pygame.draw.rect(surf, color, (x - r, y - 2, r * 2, r + 2))
        #wavy bottom edge
        wave_r = max(2, scale // 8)
        for i in range(3):
            wx = x - r + wave_r + i * (r * 2 - wave_r * 2) // 2
            wy = y + r - 1
            pygame.draw.circle(surf, BLACK, (wx, wy), wave_r)
        #direction-tracking eyes based on velocity
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
        #white sclera
        pygame.draw.circle(surf, WHITE, (x - eye_sep, y - eye_sep // 2), eye_r)
        pygame.draw.circle(surf, WHITE, (x + eye_sep, y - eye_sep // 2), eye_r)
        #black pupils — offset in movement direction
        px_off = int(dx_n * pupil_off)
        py_off = int(dy_n * pupil_off)
        pygame.draw.circle(surf, BLACK, (x - eye_sep + px_off, y - eye_sep // 2 + py_off), pupil_r)
        pygame.draw.circle(surf, BLACK, (x + eye_sep + px_off, y - eye_sep // 2 + py_off), pupil_r)