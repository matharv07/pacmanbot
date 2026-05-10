import pygame
import random
import math
import sys
from collections import deque

CELL = 20
COLS = 41
ROWS = 33
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + 48  #extra bar for score
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
GHOST_COLORS = [RED, PINK, CYAN, ORANGE, (180, 0, 180), (0, 180, 80), (220, 220, 0)]

WALL  = 1
EMPTY = 0
PELLET = 2
POWER  = 3

UP    = (-1,  0)
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS  = [UP, DOWN, LEFT, RIGHT]

def generate_map(): #using prims algorithm to generate a maze, then adding pellets and powerups
    rows, cols = ROWS, COLS
    grid = [[WALL] * cols for _ in range(rows)]

    def in_bounds(r, c):
        return 0 < r < rows - 1 and 0 < c < cols - 1

    def carve(r, c):
        grid[r][c] = EMPTY
        neighbours = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr][nc] == WALL:
                neighbours.append((nr, nc, r + dr // 2, c + dc // 2))
        random.shuffle(neighbours)
        return neighbours

    sr = random.randrange(1, rows - 1, 2)
    sc = random.randrange(1, cols - 1, 2)
    frontier = carve(sr, sc)

    while frontier:
        idx = random.randrange(len(frontier))
        nr, nc, wr, wc = frontier.pop(idx)
        if grid[nr][nc] == WALL:
            grid[wr][wc] = EMPTY
            frontier.extend(carve(nr, nc))

    #adding some extra random openings to reduce dead ends
    for _ in range(int(rows * cols * 0.1)):
        r = random.randrange(1, rows - 1)
        c = random.randrange(1, cols - 1)
        grid[r][c] = EMPTY

    #making border all wall
    for c in range(cols):
        grid[0][c] = WALL
        grid[rows - 1][c] = WALL
    for r in range(rows):
        grid[r][0] = WALL
        grid[r][cols - 1] = WALL

    open_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == EMPTY]

    centre = (rows // 2, cols // 2)
    open_cells.sort(key=lambda p: abs(p[0] - centre[0]) + abs(p[1] - centre[1]))
    player_start = open_cells[0]

    for r, c in open_cells:
        if abs(r - player_start[0]) + abs(c - player_start[1]) > 2:
            grid[r][c] = PELLET

    #convert 28 random pellets to powered pellets
    random.shuffle(open_cells)
    placed = 0
    for r, c in open_cells:
        if placed >= 28:
            break
        if abs(r - player_start[0]) + abs(c - player_start[1]) > 4:
            grid[r][c] = POWER
            placed += 1

    return grid, player_start


def bfs_path(grid, start, goal):
    """Return first step direction from start toward goal, or None."""
    rows = len(grid)
    cols = len(grid[0])
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path[0] if path else None
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr][nc] != WALL):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(dr, dc)]))
    return None


class Ghost:
    def __init__(self, gid, grid, pos, color):
        self.gid = gid          #permanent ID for hive implementation
        self.grid = grid
        self.row, self.col = pos
        self.color = color
        self.scared = False
        self.dead = False
        self.respawn = pos
        self.dead_timer = 0
        self.move_counter = 0
        self.move_every = 2   #ticks between moves
        self.last_dir = random.choice(DIRS)

    def update(self, player_pos, scared):
        self.move_counter += 1
        if self.move_counter < self.move_every:
            return
        self.move_counter = 0

        rows = len(self.grid)
        cols = len(self.grid[0])

        #available moves
        options = []
        for dr, dc in DIRS:
            nr, nc = self.row + dr, self.col + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and self.grid[nr][nc] != WALL):
                options.append((dr, dc))
        if not options:
            return

        #70% continue last dir, 30% random for now - RL later
        if self.last_dir in options and random.random() < 0.70:
            options = [self.last_dir]
        else:
            random.shuffle(options)

        dr, dc = options[0]
        self.row += dr
        self.col += dc
        self.last_dir = (dr, dc)

        #modifies game such that if a ghost captures a power pellet, it changes to a normal pellet
        if self.grid[self.row][self.col] == POWER:
            self.grid[self.row][self.col] = PELLET

    def kill(self):
        self.dead = True
        self.dead_timer = 30
        self.scared = False

    def draw(self, surf):
        if self.dead:
            return
        x = self.col * CELL + CELL // 2
        y = self.row * CELL + CELL // 2
        r = CELL // 2 - 2
        color = self.color
        pygame.draw.circle(surf, color, (x, y - 2), r)
        pygame.draw.rect(surf, color, (x - r, y - 2, r * 2, r + 2))
        pygame.draw.circle(surf, WHITE, (x - 4, y - 4), 3)
        pygame.draw.circle(surf, WHITE, (x + 4, y - 4), 3)
        pygame.draw.circle(surf, BLACK, (x - 3, y - 4), 2)
        pygame.draw.circle(surf, BLACK, (x + 5, y - 4), 2)

class Player:
    def __init__(self, grid, pos):
        self.grid = grid
        self.row, self.col = pos
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

    def set_dir(self, d):
        self.next_dir = d

    def update(self):
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

        if can_move(self.row, self.col, self.next_dir):
            self.dir = self.next_dir

        if can_move(self.row, self.col, self.dir):
            self.row += self.dir[0]
            self.col += self.dir[1]

        cell = self.grid[self.row][self.col]
        if cell == PELLET:
            self.grid[self.row][self.col] = EMPTY
            self.score += 10
        elif cell == POWER:
            self.grid[self.row][self.col] = EMPTY
            self.score += 50
            self.powered = True
            self.power_timer = 40

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
            #draw pie shape
            gap = 35
            start_a = math.radians(angle + gap)
            end_a = math.radians(angle - gap)
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
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PACMAN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.small = pygame.font.SysFont("monospace", 14)
        self.new_game()

    def new_game(self):
        self.grid, self.player_start = generate_map()
        self.player = Player(self.grid, self.player_start)

        #pellet counter
        self.total_pellets = sum(1 for r in self.grid for c in r if c in (PELLET, POWER))

        #randomise ghost starting positions
        open_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] != WALL]
        pr, pc = self.player_start
        open_cells.sort(key=lambda p: -abs(p[0] - pr) - abs(p[1] - pc))
        ghost_starts = open_cells[:7]
        self.ghosts = { i: Ghost(i, self.grid, pos, GHOST_COLORS[i]) for i, pos in enumerate(ghost_starts) }
        self.state = "playing"   #states = {playing, dead, win, gameover}
        self.message_timer = 0

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
                elif event.key == pygame.K_r:
                    score = self.player.score
                    self.new_game()
                    self.player.score = score  #keeps score on reset

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

        self.player.update()

        scared = self.player.powered
        for ghost in self.ghosts.values():
            ghost.update((self.player.row, self.player.col), scared)

        #collision condition
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if (not ghost.dead
                        and ghost.row == self.player.row
                        and ghost.col == self.player.col):
                    if self.player.powered:
                        del self.ghosts[gid]   #ID deleted, others unchanged to keep hive up
                        self.player.score += 200
                    else:
                        self.player.die()
                        self.state = "gameover"
                        self.message_timer = 90
                        break

        # win condition
        if self.pellets_left() == 0:
            self.state = "win"
            self.message_timer = 60

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
            if self.state == "win":
                self.draw_overlay("CLEARED!  Next map loading...", YELLOW)
            elif self.state == "dead":
                self.draw_overlay(f"LIVES LEFT: {self.player.lives}", RED)
            elif self.state == "gameover":
                self.draw_overlay(f"GAME OVER   SCORE: {self.player.score}", RED)
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    Game().run()