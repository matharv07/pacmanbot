import numpy as np
import random
from pacman import generate_map, Player, GHOST_COLORS, ROWS, COLS
from ghost import Ghost, DIRS, WALL, PELLET, POWER, EMPTY, UNKNOWN

class PacmanMultiAgentEnv:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.step_count = 0
        self.grid = None
        self.player_start = None
        self.player = None
        self.ghosts = {}
        
    def reset(self):
        self.step_count = 0
        self.grid, self.player_start = generate_map()
        self.player = Player(self.grid, self.player_start)
        open_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] != WALL]
        pr, pc = self.player_start
        open_cells.sort(key=lambda p: -abs(p[0] - pr) - abs(p[1] - pc))
        ghost_starts = open_cells[:7]
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i], self.player_start) for i, pos in enumerate(ghost_starts)}
        return self._get_observations(), self._get_info()

    def step(self, actions):
        self.step_count += 1
        self.player.update(self.ghosts)
        powered = self.player.powered
        rewards = {i: -0.25 for i in self.ghosts.keys()}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                continue
            if gid in actions:
                action = actions[gid]
                if isinstance(action, tuple) and len(action) == 2:
                    dr, dc = action
                    nr, nc = ghost.row + dr, ghost.col + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] != WALL:
                        ghost.prev_row, ghost.prev_col = ghost.row, ghost.col
                        ghost.row, ghost.col = nr, nc
                        ghost.last_dir = (dr, dc)
                        if self.grid[ghost.row][ghost.col] == POWER:
                            self.grid[ghost.row][ghost.col] = PELLET
                            rewards[gid] += 7.0
            temp_agent = ghost.rl_agent
            ghost.rl_agent = None
            newly_discovered = ghost.update((self.player.row, self.player.col), powered, self.ghosts)
            ghost.rl_agent = temp_agent
            if newly_discovered:
                rewards[gid] += newly_discovered * 0.25
        terminated = False
        truncated = self.step_count >= self.max_steps
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                same_cell = (ghost.row == self.player.row and ghost.col == self.player.col)
                swapped = (ghost.row == self.player.prev_row and ghost.col == self.player.prev_col and self.player.row == ghost.prev_row and self.player.col == ghost.prev_col)
                if same_cell or swapped:
                    if self.player.powered:
                        rewards[gid] -= 100.0  #large penalty for being eaten
                        ghost.kill()
                        self.player.score += 200
                    else:
                        rewards[gid] += 300.0  #capturer reward
                        for g, pos in ghost.known_agents.items():
                            if pos != "UNKNOWN":
                                rewards[g] += 100.0  #hive strategy reward
                        self.player.die()
                        terminated = True
                        break
        if sum(1 for r in self.grid for c in r if c in (PELLET, POWER)) == 0:
            terminated = True
            for g in self.ghosts.keys():
                rewards[g] -= 200.0  #penalty for letting pacman win
        return self._get_observations(), rewards, terminated, truncated, self._get_info()

    def _get_observations(self):
        obs = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            #convert (personal+belief) map to numeric array for NN
            p_map = np.array(ghost.personal_map, dtype=np.int32)
            one_hot = np.zeros((4, ROWS, COLS), dtype=np.float32)
            for i in range(4):
                one_hot[i] = (p_map == i).astype(np.float32)
            b_map = np.zeros((ROWS, COLS), dtype=np.float32)
            if ghost.belief_map._initialised:
                for r in range(ROWS):
                    for c in range(COLS):
                        b_map[r][c] = ghost.belief_map._b[r][c]
            target_map = np.zeros((ROWS, COLS), dtype=np.float32)
            active_task = ghost.cbba_agent.get_active_task()
            if active_task and active_task.target_pos:
                tr, tc = active_task.target_pos
                target_map[tr][tc] = 1.0
            state = np.concatenate([one_hot, np.expand_dims(b_map, axis=0), np.expand_dims(target_map, axis=0)], axis=0)
            obs[gid] = state
        return obs

    def _get_info(self):
        return {"player_score": self.player.score, "step_count": self.step_count}