import numpy as np
from collections import deque
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
        # list of open (non-wall) cells
        open_cells_arr = np.argwhere(np.array(self.grid) != WALL)
        pac_pos = np.array(self.player_start)
        open_cells = [tuple(x) for x in open_cells_arr]
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
        self.pos_history = {i: deque(maxlen=10) for i in self.ghosts.keys()}
        # Precompute SciPy CSR adjacency for the static maze to accelerate Dijkstra
        try:
            from scipy.sparse import csr_matrix
            import pathfinder
            cell_to_idx = {cell: i for i, cell in enumerate(open_cells)}
            n = len(open_cells)
            rows_idx = []
            cols_idx = []
            data = []
            for i, (r, c) in enumerate(open_cells):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cell_to_idx:
                        j = cell_to_idx[(nr, nc)]
                        rows_idx.append(i)
                        cols_idx.append(j)
                        data.append(1.0)
            if n > 0:
                graph = csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))
                pathfinder._SCIPY_GRAPH_CACHE[id(self.grid)] = (graph, open_cells, cell_to_idx)
        except Exception:
            pass

        return self._get_observations(), self._get_info()
    
    def step(self, actions):
        """actions: dict {gid: int} where int is direction index 0-3 (up/down/left/right)."""
        self.step_count += 1
        action_executed = {}
        self.player.update(self.ghosts)
        powered = self.player.powered
        rewards = {i: -0.1 for i in self.ghosts.keys()}
        prev_pac_dists = {gid: abs(g.row - self.player.row) + abs(g.col - self.player.col) for gid, g in self.ghosts.items()}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                action_executed[gid] = False
                continue
            on_move_frame = (ghost.move_counter + 1) >= ghost.move_every
            action_executed[gid] = on_move_frame
            if gid in actions and on_move_frame:
                dir_idx = int(actions[gid])
                if 0 <= dir_idx < 4:
                    dr, dc = DIRS[dir_idx]
                    nr, nc = ghost.row + dr, ghost.col + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] != WALL:
                        ghost.prev_row, ghost.prev_col = ghost.row, ghost.col
                        ghost.row, ghost.col = nr, nc
                        ghost.last_dir = (dr, dc)
                        if self.grid[ghost.row][ghost.col] == POWER:
                            self.grid[ghost.row][ghost.col] = PELLET
                            rewards[gid] += 3.0
            newly_discovered = ghost.update((self.player.row, self.player.col), powered, self.ghosts, skip_movement=True)
            if newly_discovered:
                rewards[gid] += newly_discovered * 0.15
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                rewards[gid] = 0.0
                continue
            if action_executed.get(gid, False):
                current_pos = (ghost.row, ghost.col)
                history = self.pos_history.get(gid, deque())
                if len(history) >= 6:
                    rows_h = [p[0] for p in history]
                    cols_h = [p[1] for p in history]
                    centroid_r = sum(rows_h) / len(rows_h)
                    centroid_c = sum(cols_h) / len(cols_h)
                    net_displacement = abs(current_pos[0] - centroid_r) + abs(current_pos[1] - centroid_c)
                    centroid_drift = abs(rows_h[-1] - rows_h[0]) + abs(cols_h[-1] - cols_h[0])
                    if net_displacement < 2.0 and centroid_drift < 3.0:
                        rewards[gid] -= 0.5
                if gid not in self.pos_history:
                    self.pos_history[gid] = deque(maxlen=10)
                self.pos_history[gid].append(current_pos)
            new_pac_dist = abs(ghost.row - self.player.row) + abs(ghost.col - self.player.col)
            pac_delta = prev_pac_dists[gid] - new_pac_dist  
            if self.player.powered:
                if new_pac_dist <= 5:
                    rewards[gid] -= pac_delta * 1.5
            else:
                if new_pac_dist <= 8:                       # Wider hunt gradient (was 3)
                    rewards[gid] += pac_delta * 2.0
                elif new_pac_dist <= 15:                    # Gentle long-range approach reward
                    rewards[gid] += pac_delta * 0.3
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
                        rewards[gid] -= 20.0
                        ghost.kill()
                        self.player.score += 200
                    else:
                        rewards[gid] += 50.0
                        for g_id, g_obj in self.ghosts.items():
                            if g_id == gid or g_obj.dead:
                                continue
                            if ghost.known_agents.get(g_id) != "UNKNOWN":
                                dist = abs(g_obj.row - self.player.row) + abs(g_obj.col - self.player.col)
                                if dist <= 8:  
                                    rewards[g_id] += 15.0
                        self.player.die()
                        terminated = True
                        break
        if sum(1 for r in self.grid for c in r if c in (PELLET, POWER)) == 0:
            terminated = True
            for g in self.ghosts.keys():
                rewards[g] -= 25.0
        if truncated and not terminated:
            for g in self.ghosts.keys():
                rewards[g] -= 10.0
        agent_dones = {}
        for gid, ghost in self.ghosts.items():
            agent_dones[gid] = ghost.dead or terminated or truncated
        info = self._get_info()
        info['action_executed'] = action_executed
        return self._get_observations(), rewards, agent_dones, terminated or truncated, info

    def _get_observations(self):
        """Return {gid: {'spatial': (7,33,41), 'scalars': (3,)}} for alive ghosts."""
        obs = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue 
            p_map = np.array(ghost.personal_map, dtype=np.int32)
            c1_wall    = (p_map == 1).astype(np.float32)
            c2_pellet  = (p_map == 2).astype(np.float32)
            c3_power   = (p_map == 3).astype(np.float32)
            c4_unknown = (p_map == -1).astype(np.float32)

            c5_belief = ghost.get_target_proximity_channel()

            c6_self = np.zeros((ROWS, COLS), dtype=np.float32)
            c6_self[ghost.row, ghost.col] = 1.0

            active_task = ghost.cbba_agent.get_active_task()
            if active_task is not None and active_task.target_pos is not None:
                target_row = float(active_task.target_pos[0]) / float(ROWS)
                target_col = float(active_task.target_pos[1]) / float(COLS)
            else:
                target_row = 0.0
                target_col = 0.0

            c7_allies = np.zeros((ROWS, COLS), dtype=np.float32)
            if ghost.known_agents:
                valid_allies = [pos for pos in ghost.known_agents.values() if pos != "UNKNOWN"]
                if valid_allies:
                    rs, cs = zip(*valid_allies)
                    c7_allies[np.array(rs), np.array(cs)] = 1.0

            spatial = np.stack([c1_wall, c2_pellet, c3_power, c4_unknown, c5_belief, c6_self, c7_allies], axis=0)
            powered = 1.0 if ghost.pacman_powered else 0.0
            known = 1.0 if ghost.known_pacman is not None else 0.0
            staleness = 0.0
            if ghost.known_pacman is not None:
                staleness = max(0.0, 1.0 - ((ghost.frame - ghost.pacman_last_seen) / 50.0))
            scalars_arr = np.array([powered, known, staleness, target_row, target_col], dtype=np.float32)
            obs[gid] = {'spatial': spatial, 'scalars': scalars_arr}
        return obs

    def _get_info(self):
        return {"player_score": self.player.score, "step_count": self.step_count}