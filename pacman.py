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
FPS = 10

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
    for p in world.pellets:
        r_c, c_c = int(p[1]), int(p[0])
        if 0 <= r_c < rows and 0 <= c_c < cols and grid[r_c, c_c] == EMPTY:
            grid[r_c, c_c] = PELLET
    for p in world.power_pellets:
        r_c, c_c = int(p[1]), int(p[0])
        if 0 <= r_c < rows and 0 <= c_c < cols and grid[r_c, c_c] in (EMPTY, PELLET):
            grid[r_c, c_c] = POWER
    pr, pc = int(world.safe_area[0][1]), int(world.safe_area[0][0])
    return grid, (pr, pc), world

class Player:
    def __init__(self, grid, pos, world=None):
        self.grid = grid
        self.world = world
        self.row, self.col = float(pos[0]), float(pos[1])
        self.prev_row, self.prev_col = self.row, self.col
        self.start = pos
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = 1.0
        self.radius = 0.4
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
        """Geodesic distance maps from each living ghost via scipy dijkstra."""
        ghost_maps = []
        rows, cols = len(self.grid), len(self.grid[0])
        if pathfinder._SCIPY_AVAILABLE:
            cache = pathfinder.get_scipy_graph(self.grid)
            if cache is not None:
                graph, open_cells, cell_to_idx = cache
                g_indices = []
                for g in ghosts.values():
                    if not g.dead:
                        gr = max(0, min(rows - 1, int(round(g.row))))
                        gc = max(0, min(cols - 1, int(round(g.col))))
                        if (gr, gc) in cell_to_idx:
                            g_indices.append(cell_to_idx[(gr, gc)])
                if g_indices:
                    dist_matrix = pathfinder.scipy_dijkstra(csgraph=graph, directed=False, indices=g_indices)
                    if dist_matrix.ndim == 1:
                        dist_matrix = dist_matrix[np.newaxis, :]
                    r_coords = np.array([c[0] for c in open_cells])
                    c_coords = np.array([c[1] for c in open_cells])
                    for i in range(len(g_indices)):
                        g_map = np.full((rows, cols), np.inf)
                        g_map[r_coords, c_coords] = dist_matrix[i]
                        ghost_maps.append(g_map)
        return ghost_maps

    def _pick_target(self, ghosts):
        """BFS scoring to pick the best pellet (or ghost when powered) target."""
        rows, cols = len(self.grid), len(self.grid[0])
        ir = max(0, min(rows - 1, int(round(self.row))))
        ic = max(0, min(cols - 1, int(round(self.col))))
        #snap to nearest open cell if in a wall
        if self.grid[ir][ic] == WALL:
            best_d = float('inf')
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = ir + dr, ic + dc
                    if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] != WALL:
                        d = abs(dr) + abs(dc)
                        if d < best_d:
                            best_d = d
                            ir, ic = nr, nc
        if self.powered:
            best_ghost_dist = float('inf')
            best_ghost_target = None
            for g in ghosts.values():
                if not g.dead:
                    gr = max(0, min(rows - 1, int(round(g.row))))
                    gc = max(0, min(cols - 1, int(round(g.col))))
                    if self.grid[gr][gc] != WALL:
                        d = abs(ir - gr) + abs(ic - gc)
                        if d < best_ghost_dist:
                            best_ghost_dist = d
                            best_ghost_target = (gr, gc)
            if best_ghost_target is not None:
                path = pathfinder.astar(self.grid, (ir, ic), best_ghost_target)
                if len(path) >= 2:
                    return best_ghost_target, list(path[1:])
        ghost_cells = []      #ghost cell list for danger scoring
        for g in ghosts.values():
            if not g.dead:
                gr = max(0, min(rows - 1, int(round(g.row))))
                gc = max(0, min(cols - 1, int(round(g.col))))
                ghost_cells.append((gr, gc))
        best_score = float('inf')       #multi-source BFS from player position to find best pellet
        best_target = None
        graph_data = pathfinder.get_scipy_graph(self.grid)
        scipy_success = False
        if graph_data and pathfinder._SCIPY_AVAILABLE:
            graph, open_cells, cell_to_idx = graph_data
            if (ir, ic) in cell_to_idx:
                start_idx = cell_to_idx[(ir, ic)]
                distances, predecessors = pathfinder.scipy_dijkstra(graph, directed=False, indices=start_idx, return_predecessors=True)
                for idx, (r, c) in enumerate(open_cells):
                    d = distances[idx]
                    if d > 80 or math.isinf(d):
                        continue
                    cell = self.grid[r][c]
                    if cell in (PELLET, POWER):
                        ghost_safety = min(abs(r - gr) + abs(c - gc) for gr, gc in ghost_cells) if ghost_cells else 999
                        danger = max(0.0, 3.0 - ghost_safety) * 15.0 if ghost_safety < 3 else 0.0
                        weight = 0.5 if cell == POWER else 1.5
                        score = d * weight + danger
                        if score < best_score:
                            best_score = score
                            best_target = (r, c)
                scipy_success = True
        if not scipy_success:
            dist_map = np.full((rows, cols), -1, dtype=np.int32)
            dist_map[ir, ic] = 0
            queue = deque([(ir, ic)])
            while queue:
                r, c = queue.popleft()
                d = int(dist_map[r, c])
                cell = self.grid[r][c]
                if cell in (PELLET, POWER):
                    ghost_safety = min(abs(r - gr) + abs(c - gc) for gr, gc in ghost_cells) if ghost_cells else 999
                    danger = max(0.0, 3.0 - ghost_safety) * 15.0 if ghost_safety < 3 else 0.0
                    weight = 0.5 if cell == POWER else 1.5
                    score = d * weight + danger
                    if score < best_score:
                        best_score = score
                        best_target = (r, c)
                if d < 80:  #max bfs radius
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] != WALL and dist_map[nr, nc] == -1:
                            dist_map[nr, nc] = d + 1
                            queue.append((nr, nc))
        if best_target is None:
            return None, []
        if scipy_success and best_target in cell_to_idx:
            path = []
            curr_idx = cell_to_idx[best_target]
            while curr_idx >= 0 and curr_idx != -9999:
                path.append(open_cells[curr_idx])
                if curr_idx == start_idx:
                    break
                curr_idx = predecessors[curr_idx]
            path.reverse()
            if len(path) >= 2:
                return best_target, list(path[1:])
            return best_target, []
        path = pathfinder.astar(self.grid, (ir, ic), best_target)
        if len(path) >= 2:
            return best_target, list(path[1:])
        return best_target, []

    def update(self, ghosts):
        if self.dead:
            self.dead_timer -= 1
            if self.dead_timer <= 0:
                self.dead = False
                self.row, self.col = self.start
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

        self.prev_row, self.prev_col = self.row, self.col
        if self.stationary:
            if not self.powered and random.random() < 0.0107:
                self.powered = True
                self.power_timer = 40
        elif AUTO_MODE:
            self._route_age += 1
            min_ghost_dist = float('inf')
            for g in ghosts.values():
                if not g.dead:
                    gd = math.hypot(self.row - g.row, self.col - g.col)
                    if gd < min_ghost_dist:
                        min_ghost_dist = gd
            #check if route needs replanning
            ghost_emergency = not self.powered and min_ghost_dist < 2.5
            power_changed = self.powered != self._route_power_state
            target_eaten = (self._route_target is not None and self.grid[self._route_target[0]][self._route_target[1]] not in (PELLET, POWER) and not self.powered)
            path_exhausted = not self._route
            needs_replan = (path_exhausted or ghost_emergency or power_changed or target_eaten or self._route_age > 15)
            if needs_replan:
                target, path = self._pick_target(ghosts)
                self._route = path
                self._route_target = target
                self._route_power_state = self.powered
                self._route_age = 0
            #pop waypoints we've reached
            while self._route and abs(self.row - self._route[0][0]) < 0.4 and abs(self.col - self._route[0][1]) < 0.4:
                self._route.pop(0)
            #get desired heading from next waypoint
            if self._route:
                wp_r, wp_c = self._route[0]
                dr = wp_r - self.row
                dc = wp_c - self.col
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
            n_steps = max(2, int(math.ceil(check_dist_max / 0.05)))
            fracs = np.linspace(1/n_steps, 1.0, n_steps)
            cc_grid = self.col + np.outer(ray_vx_arr, fracs) * check_dist_max
            cr_grid = self.row + np.outer(ray_vy_arr, fracs) * check_dist_max
            if self.world and hasattr(self.world, 'batch_is_passable'):
                passable = self.world.batch_is_passable(cc_grid.flatten(), cr_grid.flatten(), self.radius).reshape((num_rays, n_steps))
            else:
                passable = np.ones((num_rays, n_steps), dtype=bool)
                for i in range(num_rays):
                    for s in range(n_steps):
                        if is_wall(cr_grid[i, s], cc_grid[i, s]):
                            passable[i, s] = False           
            for i in range(num_rays):
                ray_penalty = 0.0
                for s in range(n_steps):
                    if not passable[i, s]:
                        frac = (s + 1) / n_steps
                        ray_penalty = 1000.0 / frac
                        break
                interest = ray_vx_arr[i] * desired_vx + ray_vy_arr[i] * desired_vy
                hysteresis = 0.2 * (ray_vx_arr[i] * cur_vx_norm + ray_vy_arr[i] * cur_vy_norm)
                score = interest + hysteresis - ray_penalty
                if score > best_score:
                    best_score = score
                    best_vx, best_vy = ray_vx_arr[i], ray_vy_arr[i]
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
                    n_steps_s = max(2, int(math.ceil(check_dist / 0.05)))
                    s_vy_norm = smooth_vy / smooth_mag
                    s_vx_norm = smooth_vx / smooth_mag
                    fracs = np.linspace(1/n_steps_s, 1.0, n_steps_s)
                    cc_arr = self.col + s_vx_norm * check_dist * fracs
                    cr_arr = self.row + s_vy_norm * check_dist * fracs
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
                steps = max(1, int(math.ceil(math.hypot(self.vx, self.vy) / 0.2)))
                step_vx = self.vx / steps
                step_vy = self.vy / steps
                for _ in range(steps):
                    self.col += step_vx
                    self.row += step_vy
                    self.col, self.row = self.world.resolve_collision(self.col, self.row, self.radius, max_iters=10)
            else:
                #simple collision: check corners of bounding box
                nr = self.row + self.vy
                nc = self.col + self.vx
                r_rad, c_rad = self.radius, self.radius
                if not (is_wall(self.row - r_rad, nc - c_rad) or is_wall(self.row - r_rad, nc + c_rad) or 
                        is_wall(self.row + r_rad, nc - c_rad) or is_wall(self.row + r_rad, nc + c_rad)):
                    self.col = nc
                if not (is_wall(nr - r_rad, self.col - c_rad) or is_wall(nr - r_rad, self.col + c_rad) or 
                        is_wall(nr + r_rad, self.col - c_rad) or is_wall(nr + r_rad, self.col + c_rad)):
                    self.row = nr
            if self.vx > 0: self.dir = RIGHT
            elif self.vx < 0: self.dir = LEFT
            elif self.vy > 0: self.dir = DOWN
            elif self.vy < 0: self.dir = UP
            self.col = max(self.radius, min(len(self.grid[0]) - self.radius, self.col))
            self.row = max(self.radius, min(len(self.grid) - self.radius, self.row))
        r_min = max(0, int(self.row - self.radius))
        r_max = min(len(self.grid) - 1, int(self.row + self.radius))
        c_min = max(0, int(self.col - self.radius))
        c_max = min(len(self.grid[0]) - 1, int(self.col + self.radius))
        collected_anything = False
        for cr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                cell = self.grid[cr][cc]
                if cell in (PELLET, POWER):
                    self.grid[cr][cc] = EMPTY
                    self.score += 10 if cell == PELLET else 50
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
        x = int(self.col * CELL)
        y = int(self.row * CELL)
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
        import pathfinder
        pathfinder.build_scipy_graph(self.grid)
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
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i % len(GHOST_COLORS)], self.player_start) for i, pos in enumerate(ghost_starts)}
        self.state = "playing"
        self.message_timer = 0
        self.debug_ghost_id = 0
        self.frame_counter = 0
        from obs import MAX_H, MAX_W
        self.recent_nom = { i: np.zeros((MAX_H, MAX_W), dtype=np.float32) for i in range(len(self.ghosts)) }
        if RL_MODE:
            load_rl_model()

    def pellets_left(self):
        return sum(1 for r in self.grid for c in r if c in (PELLET, POWER))

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
                            dists = dijkstra_multi(g.grid, (g.row, g.col), all_targets)
                            h_dists.update(dists)
                        g.cbba_agent._phase1(g, all_tasks, h_dists)
        self.player.update(self.ghosts)
        powered = self.player.powered
        for ghost in self.ghosts.values():
            ghost.update((int(self.player.row), int(self.player.col)), powered, self.ghosts)
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                dist = math.hypot(ghost.y - self.player.row, ghost.x - self.player.col)
                collision_radius = self.player.radius + 0.4  # ghost radius ~0.4
                if dist < collision_radius:
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
            self.message_timer = 60 if not AUTO_MODE else 0

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
            else:
                return
        for r in range(ROWS):
            for c in range(COLS):
                x = WIDTH + c * CELL
                y = r * CELL
                val = ghost.personal_map[r][c]
                if val == UNKNOWN:
                    color = (30, 30, 30)
                elif val == WALL:
                    color = BLUE
                elif val == PELLET:
                    color = (180, 180, 180)
                elif val == POWER:
                    color = (255, 200, 0)
                elif val == EMPTY:
                    color = BLACK
                else:
                    color = (30, 30, 30)
                pygame.draw.rect(self.screen, color, (x, y, CELL, CELL))
        bm = ghost.belief_map
        if bm._initialised and bm._open_cells:
            probs = [bm._b[r][c] for r, c in bm._open_cells]
            max_p = max(probs) if probs else 0.0
            if max_p > 1e-9:
                cell_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                for (r, c), p in zip(bm._open_cells, probs):
                    if p < 0.001:
                        continue
                    t = min(1.0, p / max_p)
                    red = int(t * 255)
                    green = int((1.0 - t) * 40)
                    blue = int((1.0 - t) * 210)
                    alpha = int(60 + t * 180)
                    cell_surf.fill((red, green, blue, alpha))
                    self.screen.blit(cell_surf, (WIDTH + c * CELL, r * CELL))
        for gid, pos in ghost.known_agents.items():
            if pos == "UNKNOWN":
                continue
            gr, gc = pos
            x = WIDTH + gc * CELL + CELL // 2
            y = gr * CELL + CELL // 2
            pygame.draw.circle(self.screen, GHOST_COLORS[gid], (x, y), CELL // 2 - 2)
            label = self.small.render(str(gid), True, WHITE)
            self.screen.blit(label, (WIDTH + gc * CELL + 2, gr * CELL + 2))
        x = WIDTH + ghost.col * CELL + CELL // 2
        y = ghost.row * CELL + CELL // 2
        pygame.draw.circle(self.screen, GHOST_COLORS[self.debug_ghost_id], (x, y), CELL // 2 - 2)
        label = self.small.render(str(self.debug_ghost_id), True, WHITE)
        self.screen.blit(label, (WIDTH + ghost.col * CELL + 2, ghost.row * CELL + 2))
        if ghost.known_pacman:
            pr, pc = ghost.known_pacman
            x = WIDTH + pc * CELL + CELL // 2
            y = pr * CELL + CELL // 2
            pygame.draw.circle(self.screen, POWERED_COLOR if ghost.pacman_powered else YELLOW, (x, y), CELL // 2 - 2)
            label = self.small.render("P", True, BLACK)
            self.screen.blit(label, (WIDTH + pc * CELL + 2, pr * CELL + 2))
        txt = self.small.render(f"Ghost {self.debug_ghost_id} local map + belief heatmap  [0-6 to switch]", True, WHITE)
        self.screen.blit(txt, (WIDTH + 4, ROWS * CELL + 6))

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

if __name__ == "__main__":
    Game().run()