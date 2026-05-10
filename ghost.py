import pygame
import random

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


class Ghost:
    def __init__(self, gid, grid, pos, color):
        self.gid = gid  #permanent ID for hive implementation
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

