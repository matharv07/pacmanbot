import pygame
import random
import math
from collections import deque

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

RADIUS            = 12
RAY_COUNT         = 360
MAX_RAY_DIST      = 10
UNKNOWN           = -1
MEMORY_FRAMES     = 10
HEARTBEAT_EVERY   = 5
HEARTBEAT_TIMEOUT = 60  #if no heartbeat received for 60 frames (6s ), consider agent lost
RESYNC_EVERY      = 50


class Ghost:
    def __init__(self, gid, grid, pos, color):
        self.gid = gid
        self.grid = grid
        self.row, self.col = pos
        self.color = color
        self.scared = False
        self.dead = False
        self.respawn = pos
        self.dead_timer = 0
        self.move_counter = 0
        self.move_every = 2
        self.last_dir = random.choice(DIRS)
        rows = len(grid)
        cols = len(grid[0])
        self.personal_map = [[UNKNOWN] * cols for _ in range(rows)]
        self.last_seen = [[-999] * cols for _ in range(rows)]
        self.frame = 0
        self.message_queue = []
        self.seen_message_ids = set()
        self.seq = 0
        self.known_agents = {} #gid - (row, col) | UNKNOWN if ghost is lost or dead, None if not seen yet
        self.known_pacman = None #(row, col) | None
        self.last_heartbeat = {} #frame of last received heartbeat from every ghost
        self.last_sync_frame  = {} #frame of last full sync sent to every ghost

    def update(self, player_pos, scared, all_ghosts):
        self.frame += 1
        self.scared = scared
        self._check_liveness(all_ghosts)
        diffs = self._update_personal_map(all_ghosts, player_pos)
        #piggyback heartbeat onto broadcast every N=5 frames
        if self.frame % HEARTBEAT_EVERY == 0:
            diffs.append(("heartbeat", self.gid, self.row, self.col))
        self._broadcast(diffs, all_ghosts)
        self._process_messages(all_ghosts)
        self.move_counter += 1
        if self.move_counter < self.move_every:
            return
        self.move_counter = 0
        rows = len(self.grid)
        cols = len(self.grid[0])
        options = []
        for dr, dc in DIRS:
            nr, nc = self.row + dr, self.col + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and self.grid[nr][nc] != WALL):
                options.append((dr, dc))
        if not options:
            return
        if self.last_dir in options and random.random() < 0.70:
            options = [self.last_dir]
        else:
            random.shuffle(options)
        dr, dc = options[0]
        self.row += dr
        self.col += dc
        self.last_dir = (dr, dc)
        if self.grid[self.row][self.col] == POWER:
            self.grid[self.row][self.col] = PELLET

    def _check_liveness(self, all_ghosts):
        for gid in list(self.last_heartbeat.keys()):
            if self.frame - self.last_heartbeat[gid] > HEARTBEAT_TIMEOUT:
                if self.known_agents.get(gid) != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    self._broadcast([("agent_lost", gid)], all_ghosts)

    def _get_visible_cells(self, all_ghosts, player_pos):
        visible = {}
        rows = len(self.grid)
        cols = len(self.grid[0])
        visible[(self.row, self.col)] = self.grid[self.row][self.col]

        for i in range(RAY_COUNT):
            angle = math.radians(i)
            dx = math.cos(angle)
            dy = math.sin(angle)
            rx = self.col + 0.5
            ry = self.row + 0.5
            for _ in range(MAX_RAY_DIST * 2):
                rx += dx * 0.5
                ry += dy * 0.5
                c = int(rx)
                r = int(ry)
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                cell_val = self.grid[r][c]
                visible[(r, c)] = cell_val
                if cell_val == WALL:
                    break

        #check co-ghost visibility
        agent_diffs = []
        for gid, ghost in all_ghosts.items():
            if gid == self.gid:
                continue
            if (ghost.row, ghost.col) in visible:
                old = self.known_agents.get(gid)
                if old != (ghost.row, ghost.col):
                    self.known_agents[gid] = (ghost.row, ghost.col)
                    agent_diffs.append(("agent", gid, ghost.row, ghost.col))
            else:
                last_known = self.known_agents.get(gid)
                if last_known is not None and last_known != "UNKNOWN":
                    lr, lc = last_known
                    if (lr, lc) in visible:
                        self.known_agents[gid] = "UNKNOWN"
                        agent_diffs.append(("agent_lost", gid))
        #check pacman visibility
        pacman_diff = None
        pr, pc = player_pos
        if (pr, pc) in visible:
            if self.known_pacman != (pr, pc):
                self.known_pacman = (pr, pc)
                pacman_diff = ("pacman", pr, pc)

        return visible, agent_diffs, pacman_diff

    def _update_personal_map(self, all_ghosts, player_pos):
        visible, agent_diffs, pacman_diff = self._get_visible_cells(all_ghosts, player_pos)
        diffs = []
        for (r, c), val in visible.items():
            old = self.personal_map[r][c]
            self.last_seen[r][c] = self.frame
            if old != val:
                self.personal_map[r][c] = val
                diffs.append(("cell", r, c, val))
        diffs.extend(agent_diffs)
        if pacman_diff:
            diffs.append(pacman_diff)
        return diffs

    def _broadcast(self, diffs, all_ghosts, msg_id=None, hop=0):
        if not diffs:
            return
        if msg_id is None:
            msg_id = (self.gid, self.frame, self.seq)
            self.seq += 1
        self.seen_message_ids.add(msg_id)
        msg = {"id": msg_id, "diffs": diffs, "hop": hop}
        for ghost in all_ghosts.values():
            if ghost.gid == self.gid:
                continue
            dist = abs(ghost.row - self.row) + abs(ghost.col - self.col)
            if dist <= RADIUS:
                ghost.message_queue.append(msg)
                #full sync for when ghost interact for the first time/for complete resync every N=50 frames
                last = self.last_sync_frame.get(ghost.gid, -999)
                if self.frame - last >= RESYNC_EVERY:
                    self.last_sync_frame[ghost.gid] = self.frame
                    ghost.last_sync_frame[self.gid] = self.frame
                    self._send_full_sync(ghost)  #self to ghost
                    ghost._send_full_sync(self)  #ghost to self

    def _send_full_sync(self, target_ghost):
        sync_diffs = []
        rows = len(self.personal_map)
        cols = len(self.personal_map[0])
        #full map sync
        for r in range(rows):
            for c in range(cols):
                val = self.personal_map[r][c]
                if val != UNKNOWN:
                    sync_diffs.append(("cell", r, c, val))
        #add all known agent positions
        for gid, pos in self.known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            elif pos is not None:
                sync_diffs.append(("agent", gid, pos[0], pos[1]))
        #transfer prev heartbeat timing data
        for gid, hb_frame in self.last_heartbeat.items():
            frames_ago = self.frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))
        #transfer pacman last known position
        if self.known_pacman:
            sync_diffs.append(("pacman", self.known_pacman[0], self.known_pacman[1]))

        if sync_diffs:
            sync_id = ("sync", self.gid, target_ghost.gid, self.frame)
            target_ghost.message_queue.append({"id": sync_id, "diffs": sync_diffs, "hop": 0})

    def _process_messages(self, all_ghosts):
        for msg in self.message_queue:
            if msg["id"] in self.seen_message_ids:
                continue
            self.seen_message_ids.add(msg["id"])
            hop = msg.get("hop", 0)
            relay_diffs = []
            for diff in msg["diffs"]:
                dtype = diff[0]
                if dtype == "cell":
                    _, r, c, val = diff
                    old = self.personal_map[r][c]
                    if old != val:
                        if old != UNKNOWN and self.last_seen[r][c] >= self.frame - MEMORY_FRAMES:
                            continue
                        self.personal_map[r][c] = val
                        relay_diffs.append(diff)
                elif dtype == "agent":
                    _, gid, r, c = diff
                    if gid == self.gid:
                        continue
                    old = self.known_agents.get(gid)
                    if old != (r, c):
                        self.known_agents[gid] = (r, c)
                        relay_diffs.append(diff)
                elif dtype == "agent_lost":
                    _, gid = diff
                    if gid == self.gid:
                        continue
                    current = self.known_agents.get(gid)
                    if current != "UNKNOWN":
                        self.known_agents[gid] = "UNKNOWN"
                        relay_diffs.append(diff)
                elif dtype == "heartbeat":
                    _, gid, r, c = diff
                    if gid == self.gid:
                        continue
                    existing = self.last_heartbeat.get(gid, -999)
                    if self.frame > existing:
                        self.last_heartbeat[gid] = self.frame
                    if r != 0 or c != 0:
                        old = self.known_agents.get(gid)
                        if old != (r, c):
                            self.known_agents[gid] = (r, c)
                            relay_diffs.append(("agent", gid, r, c))
                    #relay heartbeat further so other ghosts know positions as well
                    relay_diffs.append(diff)
                elif dtype == "hb_sync":
                    _, gid, frames_ago = diff
                    if gid == self.gid:
                        continue
                    reconstructed = self.frame - frames_ago
                    existing = self.last_heartbeat.get(gid, -999)
                    if reconstructed > existing:
                        self.last_heartbeat[gid] = reconstructed
                        relay_diffs.append(diff)
                elif dtype == "pacman":
                    _, r, c = diff
                    if self.known_pacman != (r, c):
                        self.known_pacman = (r, c)
                        relay_diffs.append(diff)

            self._broadcast(relay_diffs, all_ghosts, msg_id=msg["id"], hop=hop + 1)
        self.message_queue.clear()
        
        #prune deque keeping only N=500 latest message ids
        if len(self.seen_message_ids) > 500:
            to_remove = list(self.seen_message_ids)[:250]
            for item in to_remove:
                self.seen_message_ids.discard(item)

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