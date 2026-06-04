import pygame
import random
import math
from collections import deque
from cbba import CBBA_Agent
from pathfinder import next_step
from beliefmap import BeliefMap
from allocator import TaskType
from rl_agent import RLAgent

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
RAY_COUNT         = 360
MAX_RAY_DIST      = 10
UNKNOWN           = -1
MEMORY_FRAMES     = 10
HEARTBEAT_EVERY   = 5
HEARTBEAT_TIMEOUT = 60
RESYNC_EVERY      = 50
OSCILLATION_WINDOW = 8   #position history length to prevent oscillations

class Ghost:
    def __init__(self, gid, grid, pos, color, player_start, rl_agent=None):
        self.gid = gid
        self.grid = grid
        self.row, self.col = pos
        self.prev_row, self.prev_col = pos
        self.color = color
        self.dead = False
        self.respawn = pos
        self.dead_timer = 0
        self.move_counter = 0
        self.move_every = 2
        self.last_dir = random.choice(DIRS)
        rows = len(grid)
        cols = len(grid[0])
        self.personal_map = [[UNKNOWN] * cols for _ in range(rows)]
        self.last_seen = [[-1] * cols for _ in range(rows)]
        self.frame = 0
        self.message_queue = []
        self.seen_message_ids = {}
        self.seq = 0
        self.known_agents = {}          #(row, col) | UNKNOWN for dead/out of reach agents
        self.last_heartbeat = {}        #frame of last received heartbeat from every ghost
        self.last_sync_frame = {}       #frame of last full sync sent to every ghost
        self.known_pacman = None        #(row, col) | None for not seen yet
        self.pacman_powered = False     #normal | powered | unknown
        self.pacman_last_seen = -1      #frame of when pacman was last seen for tiebreaks
        self.last_lost_pacman = None    #(row, col) of last invalidated pacman pos
        self.prev_pac_row: int = -1     #pacman's row on previous frame - belief map
        self.prev_pac_col: int = -1     #pacman's col on previous frame - belief map
        self.cbba_agent = CBBA_Agent(gid) #CBBA auction agent for this ghost
        self.pos_history: deque = deque(maxlen=OSCILLATION_WINDOW)  #rolling position window for oscillation detection
        self.belief_map = BeliefMap(gid, self.personal_map, pacman_start=player_start)
        self.rl_agent = rl_agent
        self.last_state = None
        self.last_action_idx = -1

    def update(self, player_pos, powered, all_ghosts):
        self.frame += 1
        newly_discovered = 0
        if self.dead:
            self.dead_timer -= 1
            if self.dead_timer <= 0:
                self.dead = False
                self.row, self.col = self.respawn
            return newly_discovered
        self._check_liveness(all_ghosts)
        diffs, newly_discovered = self._update_personal_map(all_ghosts, player_pos, powered)
        #piggyback heartbeat every N=5 frames
        if self.frame % HEARTBEAT_EVERY == 0:
            diffs.append(("heartbeat", self.gid, self.row, self.col, self.frame))
        self._broadcast(diffs, all_ghosts)
        self._process_messages(all_ghosts)
        self.belief_map.update_safety_map(self.known_agents, self.frame, powered=self.pacman_powered)
        self.move_counter += 1
        if self.move_counter < self.move_every:
            return newly_discovered
        self.move_counter = 0
        #CBBA: get active task and move toward target
        active_task = self.cbba_agent.step(self, self.frame)
        if self.rl_agent is not None:
            moved = False
            if active_task is not None:
                target_pos = active_task.target_pos if hasattr(active_task, 'target_pos') else None
                nxt, current_state, action_idx = self.rl_agent.get_next_step(self.personal_map, self.belief_map, (self.row, self.col), target_pos)
                
                self.last_state = current_state
                self.last_action_idx = action_idx
                
                if (nxt is not None and nxt != (self.row, self.col) and self.grid[nxt[0]][nxt[1]] != WALL):
                    if self.pacman_powered and self.known_pacman is not None and nxt == self.known_pacman:
                        pass
                    else:
                        self.prev_row, self.prev_col = self.row, self.col
                        self.row, self.col = nxt
                        self.last_dir = (self.row - self.prev_row, self.col - self.prev_col)
                        if self.grid[self.row][self.col] == POWER:
                            self.grid[self.row][self.col] = PELLET
                        moved = True
                        
            if not moved:
                rows = len(self.grid)
                cols = len(self.grid[0])
                pac_cell = self.known_pacman if (self.pacman_powered and self.known_pacman) else None
                options = []
                for dr, dc in DIRS:
                    nr, nc = self.row + dr, self.col + dc
                    if (0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] != WALL and (nr, nc) != pac_cell):
                        options.append((dr, dc))
                if options:
                    if self.last_dir in options and random.random() < 0.70:
                        options = [self.last_dir]
                    else:
                        random.shuffle(options)
                    dr, dc = options[0]
                    self.prev_row, self.prev_col = self.row, self.col
                    self.row += dr
                    self.col += dc
                    self.last_dir = (dr, dc)
                    if self.grid[self.row][self.col] == POWER:
                        self.grid[self.row][self.col] = PELLET

        self.pos_history.append((self.row, self.col))
        self._check_oscillation()
        return newly_discovered

    def _check_oscillation(self):
        if len(self.pos_history) < OSCILLATION_WINDOW:
            return
        cur = (self.row, self.col)
        if self.pos_history.count(cur) < 2:
            return
        if self.known_pacman is None and self.last_lost_pacman is not None:
            self.last_lost_pacman = None
            self.pos_history.clear()  

    def _check_liveness(self, all_ghosts):
        for gid in list(self.last_heartbeat.keys()):
            if self.frame - self.last_heartbeat[gid] > HEARTBEAT_TIMEOUT:
                if self.known_agents.get(gid) != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    self._broadcast([("agent_lost", gid)], all_ghosts)

    def _get_visible_cells(self, all_ghosts, player_pos, powered=False):
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
            if self.known_pacman != (pr, pc) or self.pacman_powered != powered:
                self.known_pacman    = (pr, pc)
                self.pacman_powered  = powered
                self.pacman_last_seen = self.frame
                pacman_diff = ("pacman", pr, pc, powered, self.frame)
            else:
                self.pacman_last_seen = self.frame
        else:
            #if pacman not visible, check if we can see where it was last seen
            if self.known_pacman is not None:
                kr, kc = self.known_pacman
                if (kr, kc) in visible:
                    self.last_lost_pacman = (kr, kc)
                    self.pacman_last_seen = self.frame 
                    self.known_pacman     = None
                    pacman_diff = ("pacman_lost", kr, kc, self.frame)
        return visible, agent_diffs, pacman_diff

    def _update_personal_map(self, all_ghosts, player_pos, powered=False):
        visible, agent_diffs, pacman_diff = self._get_visible_cells(all_ghosts, player_pos, powered)
        diffs = []
        newly_discovered = 0
        for (r, c), val in visible.items():
            old = self.personal_map[r][c]
            self.last_seen[r][c] = self.frame
            if old != val:
                if old == UNKNOWN:
                    newly_discovered += 1
                self.personal_map[r][c] = val
                if val == WALL:
                    self.belief_map.update_local_map_cell((r, c), WALL)
                diffs.append(("cell", r, c, val))
        diffs.extend(agent_diffs)
        if pacman_diff:
            diffs.append(pacman_diff)
        pr, pc = player_pos
        pacman_in_los  = (pr, pc) in visible      #true every frame Pacman is actually visible
        pacman_just_lost = pacman_diff is not None and pacman_diff[0] == "pacman_lost"
        if pacman_in_los:
            pac_dir = (0, 0)
            if self.prev_pac_row >= 0:
                pac_dir = (pr - self.prev_pac_row, pc - self.prev_pac_col)
            self.belief_map.observe((pr, pc), pac_dir)
            self.prev_pac_row, self.prev_pac_col = pr, pc  #update every LOS frame for accurate direction
        elif pacman_just_lost:
            _, kr, kc, _ = pacman_diff
            self.belief_map.observe_lost((kr, kc))
        self.belief_map.diffuse((self.row, self.col))
        pac_pos = (pr, pc) if pacman_in_los else None  #preserve Pacman's cell during clear
        self.belief_map.observe_clear(set(visible.keys()), pac_pos)
        return diffs, newly_discovered

    def _broadcast(self, diffs, all_ghosts, msg_id=None, hop=0):
        if not diffs:
            return
        if msg_id is None:
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
            dist = abs(ghost.row - self.row) + abs(ghost.col - self.col)
            if dist <= RADIUS:
                ghost.message_queue.append(msg)
                last = self.last_sync_frame.get(ghost.gid, -1)
                if self.frame - last >= RESYNC_EVERY:
                    self.last_sync_frame[ghost.gid] = self.frame
                    ghost.last_sync_frame[self.gid] = self.frame
                    self._send_full_sync(ghost)
                    ghost._send_full_sync(self)

    def _send_full_sync(self, target_ghost):
        sync_diffs = []
        rows = len(self.personal_map)
        cols = len(self.personal_map[0])

        for r in range(rows):
            for c in range(cols):
                val = self.personal_map[r][c]
                if val != UNKNOWN:
                    sync_diffs.append(("cell", r, c, val))

        #iterate through agent positions and liveness
        for gid, pos in self.known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            elif pos is not None:
                sync_diffs.append(("agent", gid, pos[0], pos[1]))

        for gid, hb_frame in self.last_heartbeat.items():
            frames_ago = self.frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))

        #check & relay pacman state
        if self.known_pacman is not None:
            sync_diffs.append(("pacman", self.known_pacman[0], self.known_pacman[1], self.pacman_powered, self.pacman_last_seen))
        elif self.last_lost_pacman is not None and self.pacman_last_seen > -1:
            sync_diffs.append(("pacman_lost", self.last_lost_pacman[0], self.last_lost_pacman[1], self.pacman_last_seen))

        if sync_diffs:
            sync_id = ("sync", self.gid, target_ghost.gid, self.frame)
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
                if dtype == "cell":
                    _, r, c, val = diff
                    old = self.personal_map[r][c]
                    if old != val:
                        if old != UNKNOWN and self.last_seen[r][c] >= self.frame - MEMORY_FRAMES:
                            continue    #reject stale cell update if we have seen it recently
                        self.personal_map[r][c] = val
                        if val == WALL:
                            self.belief_map.update_local_map_cell((r, c), WALL)
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
                    relay_diffs.append(diff)  #always relay heartbeats so all agents know where others are
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
                        self.pacman_powered   = powered
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
            self._broadcast(relay_diffs, all_ghosts, msg_id=msg["id"], hop=hop + 1)
        self.message_queue.clear()
        #rolling prune - keep newest 250, discarding rest post 500 messages
        if len(self.seen_message_ids) > 500:
            to_remove = list(self.seen_message_ids)[:250]
            for item in to_remove:
                self.seen_message_ids.pop(item, None)

    def kill(self):
        self.dead = True
        self.dead_timer = 30

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