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
        grid_array = np.array(self.grid)
        open_cells = np.argwhere(grid_array != WALL)
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
        return self._get_observations(), self._get_info()
    
    def step(self, actions):
        self.step_count += 1
        self.player.update(self.ghosts)
        powered = self.player.powered
        rewards = {i: -0.5 for i in self.ghosts.keys()}
        prev_dist = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            task = ghost.cbba_agent.get_active_task()
            if task and task.target_pos:
                tr, tc = task.target_pos
                prev_dist[gid] = abs(ghost.row - tr) + abs(ghost.col - tc)
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                continue
            if gid in actions and (ghost.move_counter+1) >= ghost.move_every:
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
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            if gid in prev_dist:
                task = ghost.cbba_agent.get_active_task()
                if task and task.target_pos:
                    tr, tc = task.target_pos
                    new_dist = abs(ghost.row - tr) + abs(ghost.col - tc)
                    delta = prev_dist[gid] - new_dist 
                    if delta > 0:
                        rewards[gid] += delta * 0.1         
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
                        rewards[gid] -= 100
                        ghost.kill()
                        self.player.score += 200
                    else:
                        rewards[gid] += 1000.0
                        for g, pos in ghost.known_agents.items():
                            if pos != "UNKNOWN":
                                rewards[g] += 400.0
                        self.player.die()
                        terminated = True
                        break
        if sum(1 for r in self.grid for c in r if c in (PELLET, POWER)) == 0:
            terminated = True
            for g in self.ghosts.keys():
                rewards[g] -= 1000.0
        if truncated and not terminated:
            for g in self.ghosts.keys():
                rewards[g] -= 300.0
        return self._get_observations(), rewards, terminated, truncated, self._get_info()

    def _get_observations(self):
        obs = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue 
            p_map = np.array(ghost.personal_map, dtype=np.int32)
            one_hot = np.zeros((4, ROWS, COLS), dtype=np.float32)
            for i in range(4):
                one_hot[i] = (p_map == i).astype(np.float32)
            b_map = np.zeros((ROWS, COLS), dtype=np.float32)
            if hasattr(ghost.belief_map, '_initialised') and ghost.belief_map._initialised:
                b_map = np.array(ghost.belief_map._b, dtype=np.float32)
            target_map = np.zeros((ROWS, COLS), dtype=np.float32)
            for idx, task_key in enumerate(ghost.cbba_agent.path):
                task = ghost.cbba_agent._task_map.get(task_key)
                if task and hasattr(task, 'target_pos') and task.target_pos:
                    tr, tc = task.target_pos
                    intensity = max(0.1, 1.0 * (0.7 ** idx))
                    target_map[tr][tc] = max(target_map[tr][tc], intensity)
            ghost_map = np.zeros((ROWS, COLS), dtype=np.float32)
            ghost_map[ghost.row][ghost.col] = 1.0
            ally_map = np.zeros((ROWS, COLS), dtype=np.float32)
            for ally_gid, pos in ghost.known_agents.items():
                if pos != "UNKNOWN" and pos is not None:
                    ally_map[pos[0]][pos[1]] = 1.0
            state = np.concatenate([one_hot, np.expand_dims(b_map, axis=0), np.expand_dims(target_map, axis=0), np.expand_dims(ghost_map, axis=0), np.expand_dims(ally_map, axis=0)], axis=0)
            obs[gid] = state
        return obs

    def _get_info(self):
        return {"player_score": self.player.score, "step_count": self.step_count}