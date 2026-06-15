import pygame
import random
import math
import numpy as np
import sys
import os
from collections import deque
from ghost import Ghost, UNKNOWN

CELL = 20
COLS = 41
ROWS = 33
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

AUTO_MODE = False
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
        import torch
        import glob
        from net import GhostActor
        RL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ckpts = glob.glob("checkpoints/ckpt_*.pt")
        if not ckpts:
            print("No checkpoints found. RL Mode disabled.")
            RL_MODE = False
            return False
            
        latest = max(ckpts, key=os.path.getctime)
        print(f"Loading checkpoint: {latest}")
        
        RL_ACTOR = GhostActor().to(RL_DEVICE)
        checkpoint = torch.load(latest, map_location=RL_DEVICE)
        RL_ACTOR.load_state_dict(checkpoint["actor"])
        RL_ACTOR.eval()
        print("RL Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load RL Model: {e}")
        RL_MODE = False
        return False

WALL   = 1
EMPTY  = 0
PELLET = 2
POWER  = 3

UP    = (-1,  0)
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS  = [UP, DOWN, LEFT, RIGHT]

def generate_map(rows: int = ROWS, cols: int = COLS, n_power: int = 28):
    """Generate a random maze grid.

    Parameters
    ----------
    rows, cols : grid dimensions (curriculum-adjustable)
    n_power    : number of power pellets to place
    """
    grid = np.full((rows, cols), WALL, dtype=np.int8)

    def in_bounds(r, c):
        return 0 < r < rows - 1 and 0 < c < cols - 1

    def carve(r, c):
        grid[r, c] = EMPTY
        neighbours = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr, nc] == WALL:
                neighbours.append((nr, nc, r + dr // 2, c + dc // 2))
        random.shuffle(neighbours)
        return neighbours

    # Ensure start cell is on an odd row/col inside bounds
    max_sr = max(1, rows - 2)
    max_sc = max(1, cols - 2)
    sr = random.randrange(1, max_sr + 1, 2) if max_sr >= 1 else 1
    sc = random.randrange(1, max_sc + 1, 2) if max_sc >= 1 else 1
    frontier = carve(sr, sc)
    while frontier:
        idx = random.randrange(len(frontier))
        nr, nc, wr, wc = frontier.pop(idx)
        if grid[nr][nc] == WALL:
            grid[wr][wc] = EMPTY
            frontier.extend(carve(nr, nc))
    for _ in range(int(rows * cols * 0.1)):
        r = random.randrange(1, max(2, rows - 1))
        c = random.randrange(1, max(2, cols - 1))
        grid[r, c] = EMPTY
    grid[0, :] = WALL
    grid[rows - 1, :] = WALL
    grid[:, 0] = WALL
    grid[:, cols - 1] = WALL
    open_cells = list(map(tuple, np.argwhere(grid == EMPTY)))
    if not open_cells:
        # Fallback for tiny grids: force centre open
        cr, cc = rows // 2, cols // 2
        grid[cr, cc] = EMPTY
        open_cells = [(cr, cc)]
    centre = (rows // 2, cols // 2)
    open_cells.sort(key=lambda p: abs(p[0] - centre[0]) + abs(p[1] - centre[1]))
    player_start = open_cells[0]
    for r, c in open_cells:
        if abs(r - player_start[0]) + abs(c - player_start[1]) > 2:
            grid[r, c] = PELLET
    random.shuffle(open_cells)
    placed = 0
    for r, c in open_cells:
        if placed >= n_power:
            break
        if abs(r - player_start[0]) + abs(c - player_start[1]) > 4:
            grid[r][c] = POWER
            placed += 1
    return grid, player_start

class Player:
    def __init__(self, grid, pos):
        self.grid = grid
        self.row, self.col = pos
        self.prev_row, self.prev_col = pos
        self.start = pos
        self.dir = RIGHT
        self.next_dir = RIGHT
        self.score = 0
        self.powered = False
        self.power_timer = 0
        self.mouth_open = True
        self.mouth_tick = 0
        self.dead = False
        self.dead_timer = 0
        #Adam moment states for on-field processing 
        self.m_row, self.m_col = 0.0, 0.0
        self.v_row, self.v_col = 0.0, 0.0
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.macro_routing_active = False   #flag to indicate if we're currently in macro routing mode => following pellet gradients directly - adam gets confused and jittery otherwise
        self._bfs_cache = {}               #cache BFS maps per start position
        self.stationary = False            # if True, skips movement logic

    def set_dir(self, d):
        self.next_dir = d

    def _get_bfs_map(self, start_pos):
        grid_arr = np.array(self.grid)
        dist_map = np.full(grid_arr.shape, np.inf)
        r_st, c_st = start_pos
        dist_map[r_st, c_st] = 0
        active = np.zeros_like(grid_arr, dtype=bool)
        active[r_st, c_st] = True
        wall_mask = (grid_arr == WALL)
        d = 0
        while active.any():
            d += 1
            shifted_up = np.roll(active, -1, axis=0)
            shifted_down = np.roll(active, 1, axis=0)
            shifted_left = np.roll(active, -1, axis=1)
            shifted_right = np.roll(active, 1, axis=1)
            new_active = shifted_up | shifted_down | shifted_left | shifted_right
            new_active[wall_mask] = False
            #mask out already visited
            new_active = new_active & np.isinf(dist_map)
            dist_map[new_active] = d
            active = new_active
        return dist_map

    def _get_pellet_bfs_map(self):
        import numpy as np
        grid_arr = np.array(self.grid)
        dist_map = np.full(grid_arr.shape, np.inf)
        active = (grid_arr == PELLET) | (grid_arr == POWER)
        dist_map[active] = 0
        wall_mask = (grid_arr == WALL)
        d = 0
        while active.any():
            d += 1
            shifted_up = np.roll(active, -1, axis=0)
            shifted_down = np.roll(active, 1, axis=0)
            shifted_left = np.roll(active, -1, axis=1)
            shifted_right = np.roll(active, 1, axis=1)
            new_active = shifted_up | shifted_down | shifted_left | shifted_right
            new_active[wall_mask] = False
            new_active = new_active & np.isinf(dist_map)
            dist_map[new_active] = d
            active = new_active
        return dist_map

    def _evaluate_potential(self, r, c, ghost_maps, pellet_map):
        rows, cols = len(self.grid), len(self.grid[0])
        if not (0 <= r < rows and 0 <= c < cols) or self.grid[r][c] == WALL:
            return 9999.0
        g_dists = [g_map[r, c] for g_map in ghost_maps if not math.isinf(g_map[r, c]) and not math.isnan(g_map[r, c])]
        if self.powered:
            if g_dists:
                return min(g_dists) * 15.0  #hyper-focused pull to execution point
            else:
                p_dist = pellet_map[r, c]
                if math.isinf(p_dist) or math.isnan(p_dist):
                    return 999.0
                return p_dist * 1.0
        else:
            ghost_repulsion = 0.0
            for d in g_dists:
                if d <= 4:
                    ghost_repulsion += 200.0 / (d + 0.1)
                elif d <= 8:
                    ghost_repulsion += 40.0 / (d + 0.1)
            p_dist = pellet_map[r, c]
            cell_type = self.grid[r][c]
            weight = 5.0 if cell_type == POWER else 1.2
            pellet_attraction = p_dist * weight if not math.isinf(p_dist) and not math.isnan(p_dist) else 0.0
            return ghost_repulsion + pellet_attraction

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
        def can_move(r, c, d):
            nr, nc = r + d[0], c + d[1]
            return (0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] != WALL)
        self.prev_row, self.prev_col = self.row, self.col
        
        if self.stationary:
            if not self.powered and random.random() < 0.0107:
                self.powered = True
                self.power_timer = 40
        elif AUTO_MODE:
            self.t += 1
            self._bfs_cache.clear()    # invalidate BFS cache each frame (ghosts moved)
            ghost_maps = []
            min_ghost_dist = float('inf')
            for g in ghosts.values():
                if not g.dead:
                    cache_key = (g.row, g.col)
                    if cache_key in self._bfs_cache:
                        g_map = self._bfs_cache[cache_key]
                    else:
                        g_map = self._get_bfs_map(cache_key)
                        self._bfs_cache[cache_key] = g_map
                    ghost_maps.append(g_map)
                    if g_map[self.row, self.col] < min_ghost_dist:
                        min_ghost_dist = g_map[self.row, self.col]
            pellet_map = self._get_pellet_bfs_map()
            current_cell_pellet_dist = pellet_map[self.row, self.col]
            if self.macro_routing_active:           #break out of macro navigation only if we come across pellets or if a ghost intercepts us
                if current_cell_pellet_dist <= 1 or min_ghost_dist <= 4:
                    self.macro_routing_active = False
            else:                                   #enter macro navigation strategy if we are completely isolated
                if current_cell_pellet_dist > 3 and min_ghost_dist > 6:
                    self.macro_routing_active = True
            if self.macro_routing_active and not math.isinf(current_cell_pellet_dist) and not math.isnan(current_cell_pellet_dist):
                best_macro_dir = self.dir
                min_macro_dist = current_cell_pellet_dist
                for dr, dc in DIRS:
                    if can_move(self.row, self.col, (dr, dc)):
                        nr, nc = self.row + dr, self.col + dc
                        if pellet_map[nr, nc] < min_macro_dist:
                            min_macro_dist = pellet_map[nr, nc]
                            best_macro_dir = (dr, dc)
                self.dir = best_macro_dir
                self.row += self.dir[0]
                self.col += self.dir[1]
                #zero out old momentum info during manual override
                self.m_row, self.m_col = 0.0, 0.0
                self.v_row, self.v_col = 0.0, 0.0
                self.t = 0
            else:
                if not ghost_maps:
                    ghost_maps = [self._get_bfs_map((0, 0))]
                val_up = self._evaluate_potential(self.row - 1, self.col, ghost_maps, pellet_map)
                val_down = self._evaluate_potential(self.row + 1, self.col, ghost_maps, pellet_map)
                val_left = self._evaluate_potential(self.row, self.col - 1, ghost_maps, pellet_map)
                val_right = self._evaluate_potential(self.row, self.col + 1, ghost_maps, pellet_map)
                grad_row = val_up - val_down
                grad_col = val_left - val_right
                self.m_row = self.beta1 * self.m_row + (1.0 - self.beta1) * grad_row
                self.m_col = self.beta1 * self.m_col + (1.0 - self.beta1) * grad_col
                self.v_row = self.beta2 * self.v_row + (1.0 - self.beta2) * (grad_row**2)
                self.v_col = self.beta2 * self.v_col + (1.0 - self.beta2) * (grad_col**2)
                t_val = max(1, self.t)
                m_hat_r = self.m_row / (1.0 - self.beta1**t_val)
                m_hat_c = self.m_col / (1.0 - self.beta1**t_val)
                v_hat_r = self.v_row / (1.0 - self.beta2**t_val)
                v_hat_c = self.v_col / (1.0 - self.beta2**t_val)
                denom_r = math.sqrt(max(0.0, v_hat_r)) + self.eps
                denom_c = math.sqrt(max(0.0, v_hat_c)) + self.eps
                step_row = m_hat_r / denom_r
                step_col = m_hat_c / denom_c
                if math.isnan(step_row) or math.isinf(step_row):
                    step_row = 0.0
                if math.isnan(step_col) or math.isinf(step_col):
                    step_col = 0.0
                scored_moves = []
                fallback_moves = []
                for dr, dc in DIRS:
                    if can_move(self.row, self.col, (dr, dc)):
                        nr, nc = self.row + dr, self.col + dc
                        score = (dr * step_row) + (dc * step_col)
                        if (dr, dc) == self.dir:
                            score += 0.8  #Strong heading vector retention bonus - prevents taking up random paths
                        if (dr, dc) == (-self.dir[0], -self.dir[1]):
                            score -= 2.2  #penalizes uturns to pervent jitter
                        is_immediate_lethal_threat = False
                        if not self.powered:
                            for g_map in ghost_maps:
                                if g_map[nr, nc] <= 1:
                                    is_immediate_lethal_threat = True
                                    break
                        if is_immediate_lethal_threat:
                            fallback_moves.append((score, (dr, dc)))
                        else:
                            scored_moves.append((score, (dr, dc)))
                
                if scored_moves:
                    scored_moves.sort(key=lambda x: x[0], reverse=True)
                    rand_val = random.random()
                    if rand_val < 0.05 and len(scored_moves) > 2:
                        self.dir = scored_moves[2][1]    #occasional wild move
                    elif rand_val < 0.18 and len(scored_moves) > 1:
                        self.dir = scored_moves[1][1]    #suboptimal move
                    else:
                        self.dir = scored_moves[0][1]
                elif fallback_moves:
                    fallback_moves.sort(key=lambda x: x[0], reverse=True)
                    self.dir = fallback_moves[0][1]
                self.row += self.dir[0]
                self.col += self.dir[1]
        else:
            if can_move(self.row, self.col, self.next_dir):
                self.dir = self.next_dir
            if can_move(self.row, self.col, self.dir):
                self.row += self.dir[0]
                self.col += self.dir[1]

        cell = self.grid[self.row][self.col]
        if cell in (PELLET, POWER):
            self.grid[self.row][self.col] = EMPTY
            self.score += 10 if cell == PELLET else 50
            if cell == POWER:
                self.powered = True
                self.power_timer = 40
            #flush momentum info to prevent rubber-banding artifacts
            self.m_row, self.m_col = 0.0, 0.0
            self.v_row, self.v_col = 0.0, 0.0
            self.t = 0
            self._bfs_cache.clear()  # grid changed, invalidate BFS cache
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
        x = self.col * CELL + CELL // 2
        y = self.row * CELL + CELL // 2
        r = CELL // 2 - 2
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
        self.grid, self.player_start = generate_map()
        import pathfinder
        pathfinder.build_scipy_graph(self.grid)
        self.player = Player(self.grid, self.player_start)
        self.total_pellets = int(np.sum(np.isin(self.grid, (PELLET, POWER))))
        open_cells = np.argwhere(self.grid != WALL)
        pac_pos = np.array(self.player_start)
        dist_pac = np.sum(np.abs(open_cells - pac_pos), axis=1)
        min_dist_to_ghosts = np.full(len(open_cells), np.inf)
        available = np.ones(len(open_cells), dtype=bool)
        first_idx = np.argmax(dist_pac)
        ghost_starts = [tuple(open_cells[first_idx])]
        available[first_idx] = False
        for _ in range(6):
            last_placed = np.array(ghost_starts[-1])
            dist_to_last = np.sum(np.abs(open_cells - last_placed), axis=1)
            min_dist_to_ghosts = np.minimum(min_dist_to_ghosts, dist_to_last)
            scores = np.minimum(dist_pac, min_dist_to_ghosts)
            scores[~available] = -1
            best_idx = np.argmax(scores)
            ghost_starts.append(tuple(open_cells[best_idx]))
            available[best_idx] = False
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i], self.player_start) for i, pos in enumerate(ghost_starts)}
        self.state = "playing"
        self.message_timer = 0
        self.debug_ghost_id = 0
        self.frame_counter = 0
        
        from obs import MAX_H, MAX_W
        self.recent_nom = {
            i: np.zeros((MAX_H, MAX_W), dtype=np.float32) for i in range(len(self.ghosts))
        }

        if RL_MODE:
            load_rl_model()

    def pellets_left(self):
        return sum(1 for r in self.grid for c in r if c in (PELLET, POWER))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_w, pygame.K_UP):
                    self.player.set_dir(UP)
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    self.player.set_dir(DOWN)
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    self.player.set_dir(LEFT)
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    self.player.set_dir(RIGHT)
                elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    num = event.key - pygame.K_0
                    if num in self.ghosts:
                        self.debug_ghost_id = num
                elif event.key == pygame.K_r:
                    score = self.player.score
                    self.new_game()
                    self.player.score = score
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if self.state == "playing":
                        if TOGGLE_RECT.collidepoint(event.pos):
                            global AUTO_MODE
                            AUTO_MODE = not AUTO_MODE
                            self.player.m_row, self.player.m_col = 0.0, 0.0
                            self.player.v_row, self.player.v_col = 0.0, 0.0
                            self.player.t = 0
                        elif RL_TOGGLE_RECT.collidepoint(event.pos):
                            global RL_MODE
                            RL_MODE = not RL_MODE
                            if RL_MODE:
                                load_rl_model()

    def update(self):
        if self.state != "playing":
            self.message_timer -= 1
            if self.message_timer <= 0:
                if self.state == "win":
                    score = self.player.score
                    self.new_game()
                    self.player.score = score
                    self.state = "playing"
                elif self.state == "dead":
                    self.state = "playing"
                elif self.state == "gameover":
                    self.new_game()
            return

        self.frame_counter += 1
        if RL_MODE and self.frame_counter % 5 == 0:
            if RL_ACTOR is None:
                load_rl_model()
            if RL_ACTOR is not None:
                from obs import build_spatial, build_vector, build_valid_mask, actions_to_tasks
                import torch
                
                alive = [gid for gid, g in self.ghosts.items() if not g.dead]
                cbba = {gid: g.cbba_agent.get_active_task() for gid, g in self.ghosts.items()}
                
                sp, ve, vm = [], [], []
                for gid in alive:
                    g = self.ghosts[gid]
                    sp.append(build_spatial(g, self.recent_nom[gid]))
                    ve.append(build_vector(g, cbba))
                    vm.append(build_valid_mask(g))
                    
                if alive:
                    t_sp = torch.tensor(np.stack(sp), device=RL_DEVICE, dtype=torch.float32)
                    t_ve = torch.tensor(np.stack(ve), device=RL_DEVICE, dtype=torch.float32)
                    t_vm = torch.tensor(np.stack(vm), device=RL_DEVICE, dtype=torch.bool)
                    
                    with torch.no_grad():
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
                        if tasks:
                            g.cbba_agent._last_auction = self.frame_counter
                            g.cbba_agent._task_map = {(int(t.task_type), t.target_pos): t for t in tasks}
                            g.cbba_agent._phase1(g, tasks)

        self.player.update(self.ghosts)
        powered = self.player.powered
        for ghost in self.ghosts.values():
            ghost.update((self.player.row, self.player.col), powered, self.ghosts)
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                same_cell = (ghost.row == self.player.row and ghost.col == self.player.col)
                swapped = (ghost.row == self.player.prev_row and ghost.col == self.player.prev_col and self.player.row == ghost.prev_row and self.player.col == ghost.prev_col)
                if (same_cell or swapped):
                    if self.player.powered:

                        death_msg = {"id": ("death", gid, ghost.frame), "diffs": [("agent_lost", gid)], "hop": 0}
                        for g in self.ghosts.values():
                            if g.gid != gid:
                                g.message_queue.append(death_msg)
                        del self.ghosts[gid]
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
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL
                y = r * CELL
                cell = self.grid[r][c]
                if cell == WALL:
                    pygame.draw.rect(surf, DKBLUE, (x, y, CELL, CELL))
                    pygame.draw.rect(surf, BLUE, (x + 1, y + 1, CELL - 2, CELL - 2))
                else:
                    pygame.draw.rect(surf, BLACK, (x, y, CELL, CELL))
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