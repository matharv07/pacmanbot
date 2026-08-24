import sys
import os
if __name__ == "__main__":
    try:
        import setup_dependencies
        setup_dependencies.main()
    except Exception as e:
        print(f"Failed to check dependencies: {e}")
import pygame
import random
import math
import numpy as np
from collections import deque
from ghost import Ghost, UNKNOWN
import pathfinder
import torch
import argparse
from curriculum import STAGES
import glob
from net import GhostActor
from world import World

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, default=4, help="Curriculum stage index to visualize")
parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint to load")
args, _ = parser.parse_known_args()

check = args.checkpoint
STAGE = STAGES[args.stage] if args.stage != -1 and 0 <= args.stage < len(STAGES) else None
if STAGE:
    ROWS = STAGE.rows
    COLS = STAGE.cols
    N_POWER = STAGE.n_power
    N_GHOSTS = STAGE.n_ghosts
else:
    COLS = 28
    ROWS = 31
    N_POWER = 28
    N_GHOSTS = 4

CELL = 20
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + 48
FPS = 150
SIM_SPEEDUP = 1

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

AUTO_MODE = True
RL_MODE = True
TOGGLE_WIDTH, TOGGLE_HEIGHT = 160, 32
TOGGLE_RECT = pygame.Rect(WIDTH - TOGGLE_WIDTH * 2 - 20, ROWS * CELL + 8, TOGGLE_WIDTH, TOGGLE_HEIGHT)
RL_TOGGLE_RECT = pygame.Rect(WIDTH - TOGGLE_WIDTH - 10, ROWS * CELL + 8, TOGGLE_WIDTH, TOGGLE_HEIGHT)
RL_ACTOR = None
RL_DEVICE = None

def load_rl_model():
    global RL_ACTOR, RL_DEVICE, RL_MODE
    if RL_ACTOR is not None:
        return True
    print("Loading RL Model...")
    try:
        RL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpts = glob.glob("checkpoints/ckpt_*.pt")
        if not ckpts:
            print("No checkpoints found. RL Mode disabled.")
            RL_MODE = False
            return False            
        def ckpt_score(f):
            try:
                num = int(f.split('ckpt_')[-1].split('.pt')[0])
                return (num > 0 and num % 100 == 0, num)
            except ValueError:
                return (False, -1)
        latest = max(ckpts, key=ckpt_score)
        if check != -1 and os.path.exists(f"checkpoints/ckpt_{check}.pt"):
            latest = f"checkpoints/ckpt_{check}.pt"
        print(f"Loading checkpoint: {latest}")
        RL_ACTOR = GhostActor().to(RL_DEVICE)
        checkpoint = torch.load(latest, map_location=RL_DEVICE, weights_only=False)
        RL_ACTOR.load_state_dict(checkpoint["actor"])
        RL_ACTOR.eval()
        print("RL Model loaded successfully.")
        return True

    except Exception as e:
        print(f"Failed to load RL Model: {e}")
        RL_MODE = False
        return False

def generate_map(rows: int = ROWS, cols: int = COLS, n_power: int = N_POWER, random_spawn: bool = False):
    world = World(cols, rows, resolution=0.5)
    world.generate(n_obstacles=25)
    grid = np.full((rows, cols), WALL, dtype=np.int8)
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    px = (grid_x.ravel() * 1.0) + 0.5
    py = (grid_y.ravel() * 1.0) + 0.5
    dist_sq, r = world._points_to_segments_dist_sq(px, py)
    blocked = np.any(dist_sq <= (r + 0.35)**2, axis=1)
    grid_flat = np.where(blocked, WALL, EMPTY)
    grid = grid_flat.reshape((rows, cols))
    if world.pellets:
        pellets = np.array(world.pellets)
        pr = pellets[:, 1].astype(int)
        pc = pellets[:, 0].astype(int)
        valid = (pr >= 0) & (pr < rows) & (pc >= 0) & (pc < cols)
        pr, pc = pr[valid], pc[valid]
        empty_mask = grid[pr, pc] == EMPTY
        grid[pr[empty_mask], pc[empty_mask]] = PELLET
    if world.power_pellets:
        power_pellets = np.array(world.power_pellets)
        pr = power_pellets[:, 1].astype(int)
        pc = power_pellets[:, 0].astype(int)
        valid = (pr >= 0) & (pr < rows) & (pc >= 0) & (pc < cols)
        pr, pc = pr[valid], pc[valid]
        valid_mask = (grid[pr, pc] == EMPTY) | (grid[pr, pc] == PELLET)
        grid[pr[valid_mask], pc[valid_mask]] = POWER
        
    # Resync continuous world lists to the finalized snapped grid
    world.pellets = []
    world.power_pellets = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == PELLET: world.pellets.append((float(c) + 0.5, float(r) + 0.5))
            elif grid[r][c] == POWER: world.power_pellets.append((float(c) + 0.5, float(r) + 0.5))
    world._update_pellet_arrays()
    
    pr, pc = int(world.safe_area[0][1]), int(world.safe_area[0][0])
    return grid, (pr, pc), world

class Player:
    def __init__(self, grid, pos, world=None):
        self.grid = grid
        self.world = world
        self.y, self.x = float(pos[0]), float(pos[1])
        self.prev_y, self.prev_x = self.y, self.x
        self.start = pos
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = 1.0
        self.radius = 0.3
        self.dir = RIGHT
        self.next_dir = RIGHT
        self.score = 0
        self.powered = False
        self.power_timer = 0
        self.mouth_open = True
        self.mouth_tick = 0
        self.dead = False
        self.dead_timer = 0
        self._route = []                   #committed A* waypoints [(r,c), ...]
        self._route_target = None          #(r,c) of current route destination
        self._route_power_state = False    #power state when route was planned
        self._route_age = 0                #frames since last replan
        self.stationary = False            #if True, ghost skips movement logic

    def set_dir(self, d):
        self.next_dir = d

    def _get_ghost_maps(self, ghosts):
        """Geodesic distance maps from each living ghost — unused in current pipeline."""
        return []

    def _pick_target(self, ghosts):
        """PRM continuous scoring to pick the best pellet (or ghost when powered) target."""
        import pathfinder
        
        start = (self.y, self.x)
        if self.powered:
            best_ghost_dist = float('inf')
            best_ghost_target = None
            for g in ghosts.values():
                if not g.dead:
                    d = abs(self.y - g.y) + abs(self.x - g.x)
                    if d < best_ghost_dist and self.power_timer > (d * 2.0) + 15:
                        best_ghost_dist = d
                        best_ghost_target = (g.y, g.x)
            if best_ghost_target is not None:
                path = pathfinder.astar(self.world, start, best_ghost_target)
                if len(path) >= 2:
                    return best_ghost_target, list(path[1:])
                    
        targets = [(t[1], t[0]) for t in self.world.pellets + self.world.power_pellets]
        dists = pathfinder.dijkstra_multi(self.world, start, targets)
        
        best_score = float('inf')
        best_target = None
        best_path = []
        for tgt, (dist, path) in dists.items():
            if dist == math.inf: continue
            
            danger = 0.0
            for g in ghosts.values():
                if not g.dead:
                    gd = abs(tgt[0] - g.y) + abs(tgt[1] - g.x)
                    if gd < 4:
                        danger += (4 - gd) * 15.0
            
            orig_tgt = (tgt[1], tgt[0])
            weight = 0.5 if orig_tgt in self.world.power_pellets else 1.5
            score = dist * weight + danger
            if score < best_score:
                best_score = score
                best_target = orig_tgt
                best_path = path[1:]
                
        if not best_target and targets:
            best_target = (targets[0][1], targets[0][0])
            
        return best_target, best_path

    def update(self, ghosts):
        if self.dead:
            self.dead_timer -= 1
            if self.dead_timer <= 0:
                self.dead = False
                self.y, self.x = self.start
                self.dir = RIGHT
                self.next_dir = RIGHT
                self.powered = False
                self.power_timer = 0
            return
        if self.powered:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.powered = False
        rows = len(self.grid)
        cols = len(self.grid[0])
        def is_wall(r, c):
            if 0 <= r < rows and 0 <= c < cols:
                return self.grid[int(r)][int(c)] == WALL
            return True
        self.prev_y, self.prev_x = self.y, self.x
        if self.stationary:
            if not self.powered and random.random() < 0.0107:
                self.powered = True
                self.power_timer = 40
        elif AUTO_MODE:
            self._route_age += 1
            min_ghost_dist = float('inf')
            for g in ghosts.values():
                if not g.dead:
                    gd = math.hypot(self.y - g.y, self.x - g.x)
                    if gd < min_ghost_dist:
                        min_ghost_dist = gd
            #check if route needs replanning
            ghost_emergency = not self.powered and min_ghost_dist < 2.5 and self._route_age >= 3
            power_changed = self.powered != self._route_power_state
            target_eaten = False
            if self._route_target is not None and not self.powered:
                if self.world:
                    target_eaten = (self._route_target not in self.world.pellets and self._route_target not in self.world.power_pellets)
                else:
                    tr, tc = int(self._route_target[0]), int(self._route_target[1])
                    if 0 <= tr < len(self.grid) and 0 <= tc < len(self.grid[0]):
                        target_eaten = (self.grid[tr][tc] not in (PELLET, POWER))
                    else:
                        target_eaten = True
            path_exhausted = not self._route
            needs_replan = (path_exhausted or ghost_emergency or power_changed or target_eaten or self._route_age > 15)
            # Throttle expensive replanning — skip unless emergency or enough time passed
            if needs_replan and not ghost_emergency and not path_exhausted and self._route_age < 4:
                needs_replan = False
            if needs_replan:
                target, path = self._pick_target(ghosts)
                self._route = path
                self._route_target = target
                self._route_power_state = self.powered
                self._route_age = 0
            #pop waypoints we've reached
            while self._route and abs(self.y - self._route[0][0]) < 0.4 and abs(self.x - self._route[0][1]) < 0.4:
                self._route.pop(0)
            #get desired heading from next waypoint
            if self._route:
                wp_r, wp_c = self._route[0]
                dr = wp_r - self.y
                dc = wp_c - self.x
                heading_len = math.hypot(dr, dc)
                if heading_len > 0.01:
                    desired_vy = dr / heading_len
                    desired_vx = dc / heading_len
                else:
                    desired_vy = 0.0
                    desired_vx = 0.0
            else:
                desired_vy = 0.0
                desired_vx = 0.0
            if random.random() < 0.05 and not ghost_emergency:      
                wild_angle = random.uniform(0, 2 * math.pi)
                desired_vx = math.cos(wild_angle)
                desired_vy = math.sin(wild_angle)
            speed_mult = 1.0        #1.0 so that nominal motion is at max speed
            best_score = -float('inf')
            best_vx, best_vy = desired_vx, desired_vy
            num_rays = 16           #context steering for wall avoidance
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
            interests = ray_vx_arr * desired_vx + ray_vy_arr * desired_vy
            hysteresis = 0.2 * (ray_vx_arr * cur_vx_norm + ray_vy_arr * cur_vy_norm)
            scores = interests + hysteresis - ray_penalties
            best_idx = np.argmax(scores)
            best_vx, best_vy = ray_vx_arr[best_idx], ray_vy_arr[best_idx]
            #implementing momentum based low pass filter - while verifying if its safe to prevent wall clipping
            target_vy = best_vy * self.max_speed * speed_mult
            target_vx = best_vx * self.max_speed * speed_mult
            smooth_vy = self.vy * 0.7 + target_vy * 0.3
            smooth_vx = self.vx * 0.7 + target_vx * 0.3
            smooth_safe = True
            if self.world and hasattr(self.world, 'batch_is_passable'):
                #run checks for if smoothed momentum vector will hit a wall
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
                #if momentum pushes us into a wall, drop momentum and use safe steering output instantly
                self.vy = target_vy
                self.vx = target_vx
        if not self.stationary:
            #substepped continuous collision detection (CCD)
            if self.world and hasattr(self.world, 'resolve_collision'):
                self.path_this_frame = [(self.x, self.y)]
                steps = max(1, int(math.ceil(math.hypot(self.vx, self.vy) / 0.2)))
                step_vx = self.vx / steps
                step_vy = self.vy / steps
                for _ in range(steps):
                    self.x += step_vx
                    self.y += step_vy
                    self.x, self.y = self.world.resolve_collision(self.x, self.y, self.radius, max_iters=3)
                    self.path_this_frame.append((self.x, self.y))
            else:
                #simple collision: check corners of bounding box
                nr = self.y + self.vy
                nc = self.x + self.vx
                r_rad, c_rad = self.radius, self.radius
                if not (is_wall(self.y - r_rad, nc - c_rad) or is_wall(self.y - r_rad, nc + c_rad) or 
                        is_wall(self.y + r_rad, nc - c_rad) or is_wall(self.y + r_rad, nc + c_rad)):
                    self.x = nc
                if not (is_wall(nr - r_rad, self.x - c_rad) or is_wall(nr - r_rad, self.x + c_rad) or 
                        is_wall(nr + r_rad, self.x - c_rad) or is_wall(nr + r_rad, self.x + c_rad)):
                    self.y = nr
            if self.vx > 0: self.dir = RIGHT
            elif self.vx < 0: self.dir = LEFT
            elif self.vy > 0: self.dir = DOWN
            elif self.vy < 0: self.dir = UP
            self.x = max(self.radius, min(len(self.grid[0]) - self.radius, self.x))
            self.y = max(self.radius, min(len(self.grid) - self.radius, self.y))
        r_min = max(0, int(self.y - self.radius))
        r_max = min(len(self.grid) - 1, int(self.y + self.radius))
        c_min = max(0, int(self.x - self.radius))
        c_max = min(len(self.grid[0]) - 1, int(self.x + self.radius))
        collected_anything = False
        for cr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                cell = self.grid[cr][cc]
                if cell in (PELLET, POWER):
                    self.grid[cr][cc] = EMPTY
                    self.score += 10 if cell == PELLET else 50
                    pt = (float(cc) + 0.5, float(cr) + 0.5)
                    if self.world:
                        if cell == PELLET and pt in self.world.pellets:
                            self.world.pellets.remove(pt)
                            self.world._update_pellet_arrays()
                        elif cell == POWER and pt in self.world.power_pellets:
                            self.world.power_pellets.remove(pt)
                            self.world._update_pellet_arrays()
                    if cell == POWER:
                        self.powered = True
                        self.power_timer = 40
                    collected_anything = True
        if collected_anything:
            self._route = []
            self._route_target = None
        self.mouth_tick += 1
        if self.mouth_tick >= 3:
            self.mouth_tick = 0
            self.mouth_open = not self.mouth_open

    def die(self):
        if self.dead:
            return
        self.dead = True
        self.dead_timer = 20

    def draw(self, surf):
        x = int(self.x * CELL)
        y = int(self.y * CELL)
        r = CELL // 2 - 2
        angle = 0
        if self.vx > 0: angle = 0
        elif self.vx < 0: angle = 180
        elif self.vy > 0: angle = 270
        elif self.vy < 0: angle = 90
        else:
            angles = {(0, 1): 0, (0, -1): 180, (-1, 0): 90, (1, 0): 270}
            angle = angles.get(self.dir, 0)
        pac_color = POWERED_COLOR if self.powered else YELLOW
        if self.mouth_open and not self.dead:
            gap = 35
            start_a = math.radians(angle + gap)
            points = [(x, y)]
            steps = 20
            full = math.radians(360 - gap * 2)
            for i in range(steps + 1):
                a = start_a + full * i / steps
                points.append((x + r * math.cos(a), y - r * math.sin(a)))
            pygame.draw.polygon(surf, pac_color, points)
        else:
            pygame.draw.circle(surf, pac_color, (x, y), r)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
        pygame.display.set_caption("PACMAN")
        self.clock = pygame.time.Clock()
        try:
            pygame.font.init()
            self.font  = pygame.font.SysFont("monospace", 18, bold=True)
            self.small = pygame.font.SysFont("monospace", 14)
        except Exception:
            self.font  = pygame.font.Font(None, 22)
            self.small = pygame.font.Font(None, 16)
        pass
        self.new_game()

    def new_game(self):
        self.grid, self.player_start, self.world = generate_map()
        self.player = Player(self.grid, self.player_start, self.world)
        self.total_pellets = int(np.sum(np.isin(self.grid, (PELLET, POWER))))
        open_cells = np.argwhere(self.grid != WALL)
        pac_pos = np.array(self.player_start)
        dist_pac = np.sum(np.abs(open_cells - pac_pos), axis=1)
        min_dist_to_ghosts = np.full(len(open_cells), np.inf)
        available = np.ones(len(open_cells), dtype=bool)
        first_idx = np.argmax(dist_pac)
        ghost_starts = [tuple(open_cells[first_idx])]
        available[first_idx] = False
        for _ in range(N_GHOSTS - 1):
            last_placed = np.array(ghost_starts[-1])
            dist_to_last = np.sum(np.abs(open_cells - last_placed), axis=1)
            min_dist_to_ghosts = np.minimum(min_dist_to_ghosts, dist_to_last)
            scores = np.minimum(dist_pac, min_dist_to_ghosts)
            scores[~available] = -1
            best_idx = np.argmax(scores)
            ghost_starts.append(tuple(open_cells[best_idx]))
            available[best_idx] = False
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i % len(GHOST_COLORS)], self.player_start, self.world) for i, pos in enumerate(ghost_starts)}
        self.state = "playing"
        self.message_timer = 0
        self.debug_ghost_id = 0
        self.frame_counter = 0
        from obs import MAX_H, MAX_W
        self.recent_nom = { i: np.zeros((MAX_H, MAX_W), dtype=np.float32) for i in range(len(self.ghosts)) }
        if RL_MODE:
            load_rl_model()

    def pellets_left(self):
        return int(np.sum(np.isin(self.grid, (PELLET, POWER))))

    def handle_events(self):
        global AUTO_MODE, RL_MODE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_w, pygame.K_UP):
                    self.player.vy = -self.player.max_speed
                    self.player.vx = 0.0
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    self.player.vy = self.player.max_speed
                    self.player.vx = 0.0
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    self.player.vx = -self.player.max_speed
                    self.player.vy = 0.0
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    self.player.vx = self.player.max_speed
                    self.player.vy = 0.0
                elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    num = event.key - pygame.K_0
                    if num in self.ghosts:
                        self.debug_ghost_id = num
                elif event.key == pygame.K_r:
                    self.new_game()
            if event.type == pygame.KEYUP and not AUTO_MODE:
                if event.key in (pygame.K_w, pygame.K_UP):
                    if self.player.vy < 0: self.player.vy = 0
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    if self.player.vy > 0: self.player.vy = 0
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    if self.player.vx < 0: self.player.vx = 0
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    if self.player.vx > 0: self.player.vx = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if self.state == "playing":
                        if TOGGLE_RECT.collidepoint(event.pos):
                            AUTO_MODE = not AUTO_MODE
                            self.player.vx = 0.0
                            self.player.vy = 0.0
                            self.player._route = []
                            self.player._route_target = None
                        elif RL_TOGGLE_RECT.collidepoint(event.pos):
                            RL_MODE = not RL_MODE
                            if RL_MODE:
                                load_rl_model()

    def update(self):
        if self.state != "playing":
            self.message_timer -= 1
            if self.message_timer <= 0:
                if self.state == "win":
                    self.new_game()
                    self.state = "playing"
                elif self.state == "dead":
                    self.state = "playing"
                elif self.state == "gameover":
                    self.new_game()
            return
        self.frame_counter += 1
        if RL_MODE and self.frame_counter % 6 == 0:
            if RL_ACTOR is None:
                load_rl_model()
            if RL_ACTOR is not None:
                from obs import build_spatial, build_vector, build_valid_mask, actions_to_tasks
                alive = [gid for gid, g in self.ghosts.items() if not g.dead]
                cbba = {gid: g.cbba_agent.get_active_task() for gid, g in self.ghosts.items()}                
                sp, ve, vm = [], [], []
                for gid in alive:
                    g = self.ghosts[gid]
                    sp.append(build_spatial(g, self.recent_nom[gid]))
                    ve.append(build_vector(g))
                    vm.append(build_valid_mask(g))
                if alive:
                    t_sp = torch.tensor(np.stack(sp), device=RL_DEVICE, dtype=torch.float32)
                    t_ve = torch.tensor(np.stack(ve), device=RL_DEVICE, dtype=torch.float32)
                    t_vm = torch.tensor(np.stack(vm), device=RL_DEVICE, dtype=torch.bool)
                    with torch.inference_mode():
                        idx, _, scores, _, _ = RL_ACTOR(t_sp, t_ve, t_vm, K=3)
                    idx_np = idx.cpu().numpy()
                    sc_np  = scores.cpu().numpy()
                    n_cols = len(self.grid[0])
                    for i, gid in enumerate(alive):
                        g = self.ghosts[gid]
                        indices = [(int(x // n_cols), int(x % n_cols)) for x in idx_np[i]]
                        scores_map = sc_np[i]
                        self.recent_nom[gid] *= 0.8
                        for r, c in indices:
                            if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]):
                                self.recent_nom[gid][r, c] = 1.0
                        tasks = actions_to_tasks(g, scores_map, indices, self.frame_counter)
                        g.cbba_agent._last_auction = self.frame_counter
                        all_tasks = tasks
                        h_dists = {}
                        g.cbba_agent._task_map.clear()
                        if tasks:
                            all_targets = [t.target_pos for t in tasks]
                            from pathfinder import dijkstra_multi
                            dists = dijkstra_multi(g.world, (g.y, g.x), all_targets)
                            h_dists.update(dists)
                        g.cbba_agent._phase1(g, all_tasks, h_dists)
        self.player.update(self.ghosts)
        powered = self.player.powered
        for ghost in self.ghosts.values():
            ghost.update((self.player.y, self.player.x), powered, self.ghosts)
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                # add 0.15 tolerance to account for ghost's visual y-2 offset and continuous off-center steering
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
                        self.player.score += 200
                    else:
                        self.player.die()
                        self.state = "gameover"
                        self.message_timer = 90 if not AUTO_MODE else 0
                        break
        if self.pellets_left() == 0:
            self.state = "win"
            self.message_timer = 90 if not AUTO_MODE else 0
    def draw_grid(self):
        surf = self.screen
        for obs in self.world.obstacles:
            obs.draw(surf, CELL)
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL
                y = r * CELL
                cell = self.grid[r][c]
                if cell == PELLET:
                    pygame.draw.circle(surf, WHITE, (x + CELL // 2, y + CELL // 2), 2)
                elif cell == POWER:
                    pygame.draw.circle(surf, WHITE, (x + CELL // 2, y + CELL // 2), 5)

    def draw_hud(self):
        y = ROWS * CELL
        pygame.draw.rect(self.screen, BLACK, (0, y, WIDTH, 48))
        score_txt = self.font.render(f"SCORE  {self.player.score}", True, WHITE)
        self.screen.blit(score_txt, (10, y + 6))
        if self.player.powered:
            bar_w = int((self.player.power_timer / 40) * 100)
            pygame.draw.rect(self.screen, GREY, (WIDTH // 2 - 50, y + 28, 100, 8))
            pygame.draw.rect(self.screen, POWERED_COLOR, (WIDTH // 2 - 50, y + 28, bar_w, 8))
            txt = self.small.render("POWERED", True, POWERED_COLOR)
            self.screen.blit(txt, (WIDTH // 2 - 28, y + 10))
        bg_btn = (0, 200, 100) if AUTO_MODE else GREY
        pygame.draw.rect(self.screen, bg_btn, TOGGLE_RECT, border_radius=4)
        lbl_msg = "AUTO MODE" if AUTO_MODE else "MANUAL PLAY"
        text_btn = self.small.render(lbl_msg, True, WHITE)
        text_rect = text_btn.get_rect(center=TOGGLE_RECT.center)
        self.screen.blit(text_btn, text_rect)
        bg_btn_rl = (200, 0, 100) if RL_MODE else GREY
        pygame.draw.rect(self.screen, bg_btn_rl, RL_TOGGLE_RECT, border_radius=4)
        lbl_msg_rl = "RL MODE ON" if RL_MODE else "RL MODE OFF"
        text_btn_rl = self.small.render(lbl_msg_rl, True, WHITE)
        text_rect_rl = text_btn_rl.get_rect(center=RL_TOGGLE_RECT.center)
        self.screen.blit(text_btn_rl, text_rect_rl)

    def draw_personal_map(self):
        ghost = self.ghosts.get(self.debug_ghost_id)
        if not ghost:
            if self.ghosts:
                self.debug_ghost_id = next(iter(self.ghosts))
                ghost = self.ghosts[self.debug_ghost_id]
    def draw_personal_map(self):
        ghost = self.ghosts.get(self.debug_ghost_id)
        if not ghost:
            if self.ghosts:
                ghost = next(iter(self.ghosts.values()))
            else:
                return
        ox = WIDTH
        
        # Prevent continuous points from spilling over into the HUD
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(ox, 0, WIDTH, ROWS * CELL))
        
        pygame.draw.rect(self.screen, BLACK, (ox, 0, WIDTH, ROWS * CELL))
        

        # 1. draw Lidar Wall Hits (Point Cloud - Bright)
        if len(ghost.lidar_memory) > 0:
            if not hasattr(self, '_lidar_surf'):
                self._lidar_surf = pygame.Surface((WIDTH, ROWS * CELL), pygame.SRCALPHA)
                self._lidar_count = 0
                
            if len(ghost.lidar_memory) != self._lidar_count:
                self._lidar_surf.fill((0, 0, 0, 0))
                for py, px in ghost.lidar_memory:
                    hit_x = int(px * CELL)
                    hit_y = int(py * CELL)
                    pygame.draw.rect(self._lidar_surf, (150, 150, 255), (hit_x, hit_y, 2, 2))
                self._lidar_count = len(ghost.lidar_memory)
            self.screen.blit(self._lidar_surf, (ox, 0))
            
        # 2. draw Known Pellets
        if hasattr(ghost, 'known_pellets'):
            for px, py in ghost.known_pellets:
                pygame.draw.circle(self.screen, WHITE, (int(ox + px * CELL), int(py * CELL)), 2)
                
        # 3. draw Known Power Pellets
        if hasattr(ghost, 'known_power_pellets'):
            for px, py in ghost.known_power_pellets:
                pygame.draw.circle(self.screen, WHITE, (int(ox + px * CELL), int(py * CELL)), 5)
            
        # 4. fog-of-war: all belief topology nodes (including wall-area grid nodes)
        bm = ghost.belief_map
        if bm._open_cells:
            if not hasattr(self, '_prm_cache') or getattr(self, '_prm_cache_n', 0) != bm.n_nodes:
                self._prm_cache = pygame.Surface((WIDTH, ROWS * CELL), pygame.SRCALPHA)
                for n in bm._open_cells:
                    pygame.draw.rect(self._prm_cache, (30, 30, 30), (int(n[1] * CELL), int(n[0] * CELL), 4, 4))
                self._prm_cache_n = bm.n_nodes
            self.screen.blit(self._prm_cache, (ox, 0))
                
        # 5. continuous Belief Heatmap on PRM nodes
        bm = ghost.belief_map
        if bm._initialised and bm._open_cells:
            probs = bm._b_flat.tolist()
            max_p = max(probs) if probs else 0.0
            if max_p > 1e-9:
                if not hasattr(self, '_heat_cache'):
                    self._heat_cache = {}
                for (r, c), p in zip(bm._open_cells, probs):
                    if p < 0.001:
                        continue
                    t = min(1.0, p / max_p)
                    
                    # Discretize t into 32 levels to heavily hit the cache
                    t_idx = int(t * 31)
                    if t_idx not in self._heat_cache:
                        t_d = t_idx / 31.0
                        red = int(t_d * 255)
                        green = int((1.0 - t_d) * 40)
                        blue = int((1.0 - t_d) * 210)
                        alpha = int(40 + t_d * 140)
                        s_r = int(3 + t_d * 4)
                        surf = pygame.Surface((s_r*2, s_r*2), pygame.SRCALPHA)
                        pygame.draw.circle(surf, (red, green, blue, alpha), (s_r, s_r), s_r)
                        self._heat_cache[t_idx] = (surf, s_r)
                        
                    surf, s_r = self._heat_cache[t_idx]
                    self.screen.blit(surf, (ox + int(c * CELL) - s_r, int(r * CELL) - s_r))
                    
        # 6. CBBA task targets
        active_task = ghost.cbba_agent.get_active_task()
        if active_task is not None:
            tr, tc = active_task.target_pos
            tx = ox + int(tc * CELL)
            ty = int(tr * CELL)
            pygame.draw.circle(self.screen, (255, 255, 0), (tx, ty), CELL // 2 + 2, 2)
            pygame.draw.line(self.screen, (255, 255, 0), (tx - 4, ty - 4), (tx + 4, ty + 4), 2)
            pygame.draw.line(self.screen, (255, 255, 0), (tx - 4, ty + 4), (tx + 4, ty - 4), 2)
            
        # 7. communication radius circle
        from ghost import RADIUS as _COMM_RADIUS
        cx = ox + int(ghost.x * CELL)
        cy = int(ghost.y * CELL)
        pygame.draw.circle(self.screen, (60, 60, 120), (cx, cy), int(_COMM_RADIUS * CELL), 1)
        
        # 8. known agents
        for gid, pos in ghost.known_agents.items():
            if pos == "UNKNOWN": continue
            gr, gc = pos
            ax = ox + int(gc * CELL)
            ay = int(gr * CELL)
            c_col = GHOST_COLORS[gid % len(GHOST_COLORS)]
            pygame.draw.circle(self.screen, c_col, (ax, ay), CELL // 2 - 2)
            label = self.small.render(str(gid), True, WHITE)
            self.screen.blit(label, (ax - 4, ay - 6))
            
        # 9. ghost sprite
        ghost.draw(self.screen, scale=CELL, offset_x=ox)
        
        # 10. known pacman
        if ghost.known_pacman:
            pr, pc = ghost.known_pacman
            px = ox + int(pc * CELL)
            py = int(pr * CELL)
            pac_col = POWERED_COLOR if ghost.pacman_powered else YELLOW
            pygame.draw.circle(self.screen, pac_col, (px, py), CELL // 2 - 2)
            label = self.small.render("P", True, BLACK)
            self.screen.blit(label, (px - 4, py - 6))
            
        # 11. HUD label
        self.screen.set_clip(old_clip)  # Restore clip rect so HUD is not clipped!
        
        mode = "POWERED" if ghost.pacman_powered else ("HUNT" if ghost.known_pacman else "SEARCH")
        fb = " [FALLBACK]" if ghost.in_fallback_mode else ""
        txt = self.small.render(f"Ghost {self.debug_ghost_id} [{mode}{fb}]  [0-6 to switch]", True, WHITE)
        self.screen.blit(txt, (ox + 4, ROWS * CELL + 6))

    def draw_overlay(self, msg, color=WHITE):
        overlay = pygame.Surface((WIDTH, ROWS * CELL), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        txt = self.font.render(msg, True, color)
        rect = txt.get_rect(center=(WIDTH // 2, ROWS * CELL // 2))
        self.screen.blit(txt, rect)

    def run(self):
        while True:
            self.handle_events()
            for _ in range(1):
                self.update()
            self.screen.fill(BLACK)
            self.draw_grid()
            for ghost in self.ghosts.values():
                ghost.draw(self.screen)
            self.player.draw(self.screen)
            self.draw_hud()
            self.draw_personal_map()
            if self.state == "win":
                self.draw_overlay("CLEARED!  Next map loading...", YELLOW)
            elif self.state == "gameover":
                self.draw_overlay(f"GAME OVER   SCORE: {self.player.score}", RED)
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

if __name__ == "__main__":
    Game().run()