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
        
        # Ghost initialization similar to pacman.py
        open_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if self.grid[r][c] != WALL]
        pr, pc = self.player_start
        open_cells.sort(key=lambda p: -abs(p[0] - pr) - abs(p[1] - pc))
        ghost_starts = open_cells[:7]
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i], self.player_start) 
                       for i, pos in enumerate(ghost_starts)}
                       
        return self._get_observations(), self._get_info()

    def step(self, actions):
        """
        actions: dict of {ghost_id: (dr, dc)} or target tasks.
        For low-level pathfinding, action is a direction.
        For high-level allocation, action is a task target.
        """
        self.step_count += 1
        
        # Update pacman (Player) based on training mode logic
        self.player.update(self.ghosts)
        powered = self.player.powered
        
        # We process custom RL actions for ghosts here
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                continue
                
            # If an action was provided by the RL policy for this ghost
            if gid in actions:
                action = actions[gid]
                
                # Apply action (e.g., if action is a direction tuple (dr, dc))
                if isinstance(action, tuple) and len(action) == 2:
                    dr, dc = action
                    nr, nc = ghost.row + dr, ghost.col + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] != WALL:
                        ghost.prev_row, ghost.prev_col = ghost.row, ghost.col
                        ghost.row, ghost.col = nr, nc
                        ghost.last_dir = (dr, dc)
                        if self.grid[ghost.row][ghost.col] == POWER:
                            self.grid[ghost.row][ghost.col] = PELLET
            
            # Update ghost map / belief regardless of custom move
            ghost.update((self.player.row, self.player.col), powered, self.ghosts)

        # Check collisions and game over states
        terminated = False
        truncated = self.step_count >= self.max_steps
        rewards = {i: -0.1 for i in self.ghosts.keys()} # Default small time penalty
        
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                same_cell = (ghost.row == self.player.row and ghost.col == self.player.col)
                swapped = (ghost.row == self.player.prev_row and ghost.col == self.player.prev_col and 
                           self.player.row == ghost.prev_row and self.player.col == ghost.prev_col)
                if same_cell or swapped:
                    if self.player.powered:
                        rewards[gid] -= 100.0  # Huge penalty for being eaten
                        ghost.kill()
                        self.player.score += 200
                    else:
                        for g in self.ghosts.keys():
                            rewards[g] += 50.0  # Team reward for catching pacman
                        self.player.die()
                        terminated = True
                        break
        
        # Check if all pellets are eaten (Pacman wins)
        if sum(1 for r in self.grid for c in r if c in (PELLET, POWER)) == 0:
            terminated = True
            for g in self.ghosts.keys():
                rewards[g] -= 50.0  # Penalty for letting pacman win

        return self._get_observations(), rewards, terminated, truncated, self._get_info()

    def _get_observations(self):
        """
        Extract the state representation for each ghost.
        Currently returns the raw personal map and belief map.
        Can be adapted into tensors for PyTorch.
        """
        obs = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            
            # Convert personal map to numeric array for neural networks
            p_map = np.array(ghost.personal_map, dtype=np.float32)
            
            # Combine with belief map (which gives pacman probabilities)
            b_map = np.zeros_like(p_map)
            if ghost.belief_map._initialised:
                for r in range(ROWS):
                    for c in range(COLS):
                        b_map[r][c] = ghost.belief_map._b[r][c]
            
            # Stack features (Channels: [Personal Map, Belief Map])
            state = np.stack([p_map, b_map], axis=0)
            obs[gid] = state
            
        return obs

    def _get_info(self):
        return {"player_score": self.player.score, "step_count": self.step_count}
