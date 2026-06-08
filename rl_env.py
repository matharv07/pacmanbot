import math
import numpy as np
from collections import deque
from pacman import generate_map, Player, GHOST_COLORS, ROWS, COLS
from ghost import Ghost, DIRS, WALL, PELLET, POWER, EMPTY, UNKNOWN
from pathfinder import next_step, astar
from rl_agent import (NUM_ACTIONS, ACTION_HUNT, ACTION_HUNT_BELIEF,
                      ACTION_CONVERT, ACTION_EVADE, ACTION_EXPLORE)
from allocator import generate_tasks, TaskType

DECISION_EVERY = 5   # match train.py


class PacmanMultiAgentEnv:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.step_count = 0
        self.grid = None
        self.player_start = None
        self.player = None
        self.ghosts = {}
        self.current_targets = {}   # gid -> (row, col) current A* target
        self.current_actions = {}   # gid -> task_type_idx

    def reset(self):
        self.step_count = 0
        self.grid, self.player_start = generate_map()
        self.player = Player(self.grid, self.player_start)
        # spread ghosts far from Pacman and each other
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
        self.ghosts = {i: Ghost(i, self.grid, pos, GHOST_COLORS[i], self.player_start)
                       for i, pos in enumerate(ghost_starts)}
        self.pos_history = {i: deque(maxlen=10) for i in self.ghosts.keys()}
        self.current_targets = {}
        self.current_actions = {}
        # precompute SciPy CSR adjacency
        try:
            from scipy.sparse import csr_matrix
            import pathfinder
            cell_to_idx = {cell: i for i, cell in enumerate(open_cells)}
            n = len(open_cells)
            rows_idx, cols_idx, data = [], [], []
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
        """actions: dict {gid: task_type_idx} (0-4)."""
        self.step_count += 1
        action_executed = {}
        self.player.update(self.ghosts)
        powered = self.player.powered
        rewards = {i: 0.0 for i in self.ghosts.keys()}
        prev_pac_dists = {gid: abs(g.row - self.player.row) + abs(g.col - self.player.col)
                          for gid, g in self.ghosts.items()}

        on_decision = (self.step_count % DECISION_EVERY == 0)

        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                ghost.update((self.player.row, self.player.col), powered, self.ghosts)
                action_executed[gid] = False
                continue

            on_move_frame = (ghost.move_counter + 1) >= ghost.move_every
            action_executed[gid] = on_move_frame

            # update target if new action received on decision frame
            if gid in actions and actions[gid] is not None and on_decision:
                payload = actions[gid]
                action_idx = payload['action']
                q_values = payload['q_values']
                
                # compute valid mask
                valid_mask = np.ones(NUM_ACTIONS, dtype=bool)
                if ghost.known_pacman is None and ghost.last_lost_pacman is None:
                    if not (hasattr(ghost, 'belief_map') and getattr(ghost.belief_map, '_initialised', False)):
                        valid_mask[ACTION_HUNT] = False
                has_power = np.any(np.array(ghost.personal_map) == POWER)
                if not has_power:
                    valid_mask[ACTION_CONVERT] = False
                if not ghost.pacman_powered:
                    valid_mask[ACTION_EVADE] = False
                if not valid_mask.any():
                    valid_mask[:] = True
                    
                from rl_agent import RLAgent
                rl_tasks = RLAgent.generate_task_bids_from_q(
                    ghost, q_values, valid_mask, self.step_count, chosen_action_idx=action_idx)
                
                ghost.latest_rl_tasks = rl_tasks

            # Run CBBA step (this can be called every frame or just when needed, cbba_agent handles timing internally)
            active_task = ghost.cbba_agent.step(ghost, self.step_count)
            target = active_task.target_pos if active_task else None
            self.current_targets[gid] = target

            # A* navigation toward current target
            if on_move_frame:
                target = self.current_targets.get(gid)
                if target is not None:
                    nxt = next_step(ghost.personal_map,
                                    (ghost.row, ghost.col), target)
                    if nxt is not None and self.grid[nxt[0]][nxt[1]] != WALL:
                        # avoid stepping onto powered Pacman
                        if powered and ghost.known_pacman is not None:
                            task = self.current_actions.get(gid, ACTION_EXPLORE)
                            if task != ACTION_HUNT and nxt == ghost.known_pacman:
                                nxt = None
                        if nxt is not None:
                            ghost.prev_row, ghost.prev_col = ghost.row, ghost.col
                            ghost.row, ghost.col = nxt
                            ghost.last_dir = (ghost.row - ghost.prev_row,
                                              ghost.col - ghost.prev_col)
                            if self.grid[ghost.row][ghost.col] == POWER:
                                self.grid[ghost.row][ghost.col] = PELLET
                                rewards[gid] += 3.0
                    else:
                        self._random_move(ghost)
                else:
                    self._random_move(ghost)

            # perception update (belief map, CBBA, comms) — skip movement
            newly_discovered = ghost.update(
                (self.player.row, self.player.col), powered,
                self.ghosts, skip_movement=True)
            if newly_discovered:
                rewards[gid] += newly_discovered * 0.15

        # ── reward shaping ────────────────────────────────────────────
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                rewards[gid] = 0.0
                continue

            if on_decision:
                rewards[gid] -= 0.1    # step penalty at decision points only

            if action_executed.get(gid, False):
                current_pos = (ghost.row, ghost.col)
                history = self.pos_history.get(gid, deque())
                if len(history) >= 6:
                    rows_h = [p[0] for p in history]
                    cols_h = [p[1] for p in history]
                    centroid_r = sum(rows_h) / len(rows_h)
                    centroid_c = sum(cols_h) / len(cols_h)
                    net_disp = abs(current_pos[0] - centroid_r) + abs(current_pos[1] - centroid_c)
                    drift = abs(rows_h[-1] - rows_h[0]) + abs(cols_h[-1] - cols_h[0])
                    if net_disp < 2.0 and drift < 3.0:
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
                if new_pac_dist <= 8:
                    rewards[gid] += pac_delta * 2.0
                elif new_pac_dist <= 15:
                    rewards[gid] += pac_delta * 0.3

        # ── collisions ────────────────────────────────────────────────
        terminated = False
        truncated = self.step_count >= self.max_steps
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                same_cell = (ghost.row == self.player.row and
                             ghost.col == self.player.col)
                swapped = (ghost.row == self.player.prev_row and
                           ghost.col == self.player.prev_col and
                           self.player.row == ghost.prev_row and
                           self.player.col == ghost.prev_col)
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
                                dist = abs(g_obj.row - self.player.row) + \
                                       abs(g_obj.col - self.player.col)
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
        return self._get_observations(), rewards, agent_dones, \
               terminated or truncated, info

    # ── target resolution ─────────────────────────────────────────────
    def _resolve_target(self, ghost, task_type):
        """Convert a task-type index into a concrete (row, col) target."""
        if task_type == ACTION_HUNT:
            target = ghost.known_pacman or ghost.last_lost_pacman
            if target is None and hasattr(ghost, 'belief_map') and ghost.belief_map._initialised:
                top = ghost.belief_map.top_cells(n=1)
                if top:
                    target = top[0]
            return target

        elif task_type == ACTION_HUNT_BELIEF:
            if hasattr(ghost, 'belief_map') and ghost.belief_map._initialised:
                top = ghost.belief_map.top_cells(n=1)
                if top:
                    return top[0]
            return None

        elif task_type == ACTION_CONVERT:
            best, best_d = None, 1e9
            for r in range(ROWS):
                for c in range(COLS):
                    if ghost.personal_map[r][c] == POWER:
                        d = abs(r - ghost.row) + abs(c - ghost.col)
                        if d < best_d:
                            best_d = d
                            best = (r, c)
            return best

        elif task_type == ACTION_EVADE:
            pac = ghost.known_pacman
            if pac is None:
                return None
            best, best_d = None, -1
            for r in range(ROWS):
                for c in range(COLS):
                    if ghost.personal_map[r][c] in (WALL, UNKNOWN):
                        continue
                    d = abs(r - pac[0]) + abs(c - pac[1])
                    if d > best_d:
                        best_d = d
                        best = (r, c)
            return best

        elif task_type == ACTION_EXPLORE:
            tasks = generate_tasks(ghost, ghost.frame)
            explore = [t for t in tasks if t.task_type == TaskType.EXPLORE]
            if explore:
                return explore[0].target_pos
            if tasks:
                return tasks[0].target_pos
            return None

        return None

    def _random_move(self, ghost):
        """Fallback: take a random valid step."""
        options = []
        for dr, dc in DIRS:
            nr, nc = ghost.row + dr, ghost.col + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] != WALL:
                options.append((nr, nc, dr, dc))
        if options:
            import random
            nr, nc, dr, dc = random.choice(options)
            ghost.prev_row, ghost.prev_col = ghost.row, ghost.col
            ghost.row, ghost.col = nr, nc
            ghost.last_dir = (dr, dc)

    # ── observations ──────────────────────────────────────────────────
    def _get_observations(self):
        """Return {gid: {'spatial': (9,33,41), 'scalars': (8,), 'valid_mask': (5,)}}."""
        obs = {}
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            p_map = np.array(ghost.personal_map, dtype=np.int32)

            c0_wall    = (p_map == 1).astype(np.float32)
            c1_pellet  = (p_map == 2).astype(np.float32)
            c2_power   = (p_map == 3).astype(np.float32)
            c3_unknown = (p_map == -1).astype(np.float32)

            c4_belief = np.zeros((ROWS, COLS), dtype=np.float32)
            if hasattr(ghost, 'belief_map') and ghost.belief_map._initialised:
                c4_belief = ghost.belief_map._b.astype(np.float32)

            c5_safety = np.ones((ROWS, COLS), dtype=np.float32)
            if hasattr(ghost, 'belief_map'):
                c5_safety = ghost.belief_map._safety.astype(np.float32)

            c6_self = np.zeros((ROWS, COLS), dtype=np.float32)
            c6_self[ghost.row, ghost.col] = 1.0

            c7_allies = np.zeros((ROWS, COLS), dtype=np.float32)
            if ghost.known_agents:
                for pos in ghost.known_agents.values():
                    if pos != "UNKNOWN":
                        c7_allies[pos[0], pos[1]] = 1.0

            c8_pacman = np.zeros((ROWS, COLS), dtype=np.float32)
            if ghost.known_pacman is not None:
                c8_pacman[ghost.known_pacman[0], ghost.known_pacman[1]] = 1.0
            elif ghost.last_lost_pacman is not None:
                c8_pacman[ghost.last_lost_pacman[0], ghost.last_lost_pacman[1]] = 0.5
            elif hasattr(ghost, 'belief_map') and getattr(ghost.belief_map, '_initialised', False):
                top = ghost.belief_map.top_cells(n=1)
                if top:
                    c8_pacman[top[0][0], top[0][1]] = 0.3

            spatial = np.stack([c0_wall, c1_pellet, c2_power, c3_unknown,
                                c4_belief, c5_safety, c6_self, c7_allies,
                                c8_pacman], axis=0)

            powered   = 1.0 if ghost.pacman_powered else 0.0
            known     = 1.0 if ghost.known_pacman is not None else 0.0
            staleness = 0.0
            if ghost.known_pacman is not None:
                staleness = max(0.0, 1.0 - (
                    (ghost.frame - ghost.pacman_last_seen) / 50.0))

            active_task = ghost.cbba_agent.get_active_task()
            if active_task is not None and active_task.target_pos is not None:
                target_row = float(active_task.target_pos[0]) / float(ROWS)
                target_col = float(active_task.target_pos[1]) / float(COLS)
            else:
                target_row = 0.0
                target_col = 0.0

            num_power = float(np.sum(c2_power)) / 28.0
            alive = sum(1 for p in ghost.known_agents.values()
                        if p != "UNKNOWN")
            num_allies = float(alive) / 6.0
            fss = min(ghost.belief_map.frames_since_sighting, 200) / 200.0

            scalars = np.array([powered, known, staleness,
                                target_row, target_col,
                                num_power, num_allies, fss],
                               dtype=np.float32)

            # valid task mask
            valid_mask = np.ones(NUM_ACTIONS, dtype=bool)
            if ghost.known_pacman is None and ghost.last_lost_pacman is None:
                if not (hasattr(ghost, 'belief_map') and ghost.belief_map._initialised):
                    valid_mask[ACTION_HUNT] = False
            has_power = np.any(c2_power)
            if not has_power:
                valid_mask[ACTION_CONVERT] = False
            if not ghost.pacman_powered:
                valid_mask[ACTION_EVADE] = False
            if not valid_mask.any():
                valid_mask[:] = True

            obs[gid] = {'spatial': spatial, 'scalars': scalars,
                        'valid_mask': valid_mask}
        return obs

    def _get_info(self):
        return {"player_score": self.player.score,
                "step_count": self.step_count}