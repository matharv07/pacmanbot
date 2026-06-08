import os
import random
import numpy as np
from collections import deque
from allocator import Task, TaskType, WALL, POWER, UNKNOWN, generate_tasks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    device = "cpu"

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

# ── Task-type action indices ──────────────────────────────────────────
ACTION_HUNT        = 0   # chase known / last-lost Pacman position
ACTION_HUNT_BELIEF = 1   # go to belief-map hotspot
ACTION_CONVERT     = 2   # eat nearest POWER pellet
ACTION_EVADE       = 3   # flee from powered Pacman
ACTION_EXPLORE     = 4   # explore unknown / stale areas
NUM_ACTIONS        = 5

ROWS = 33
COLS = 41


class GhostDQN(nn.Module if TORCH_AVAILABLE else object):
    """
    Dueling DQN: 9-channel spatial + 8 scalar → 5 task-type Q-values.
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    def __init__(self, input_channels=9, rows=ROWS, cols=COLS,
                 num_scalars=8, num_actions=NUM_ACTIONS):
        if not TORCH_AVAILABLE:
            return
        super(GhostDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        def conv2d_size_out(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(cols))
        convh = conv2d_size_out(conv2d_size_out(rows))
        flat_size = convw * convh * 128 + num_scalars

        # value stream
        self.value_fc  = nn.Linear(flat_size, 256)
        self.value_out = nn.Linear(256, 1)

        # advantage stream
        self.adv_fc  = nn.Linear(flat_size, 256)
        self.adv_out = nn.Linear(256, num_actions)

    def forward(self, x, scalars=None):
        if not TORCH_AVAILABLE:
            return None
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if scalars is not None:
            x = torch.cat([x, scalars], dim=-1)

        value     = F.relu(self.value_fc(x))
        value     = self.value_out(value)                # (B, 1)

        advantage = F.relu(self.adv_fc(x))
        advantage = self.adv_out(advantage)              # (B, num_actions)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# ── Replay Buffer ─────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, scalars, action, reward, next_state, next_scalars,
             done, valid_mask):
        self.buffer.append((state, scalars, action, reward,
                            next_state, next_scalars, done, valid_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, scalars, actions, rewards, next_states, next_scalars, dones, masks = zip(*batch)
        return (np.array(states),  np.array(scalars),
                np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(next_scalars),
                np.array(dones, dtype=np.float32), np.array(masks))

    def __len__(self):
        return len(self.buffer)


# ── DQN Trainer ───────────────────────────────────────────────────────
class DQN_Trainer:
    def __init__(self, online_net, target_net, lr=1e-4, gamma=0.99,
                 batch_size=512, target_sync_every=2000):
        self.online_net = online_net
        self.target_net = target_net
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.train_steps = 0
        if TORCH_AVAILABLE:
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
            self.sync_target()          # start with identical weights

    def sync_target(self):
        if TORCH_AVAILABLE:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def update(self, replay_buffer):
        if not TORCH_AVAILABLE or len(replay_buffer) < self.batch_size:
            return 0.0
        states, scalars, actions, rewards, n_states, n_scalars, dones, masks = \
            replay_buffer.sample(self.batch_size)

        s   = torch.FloatTensor(states).to(device)
        sc  = torch.FloatTensor(scalars).to(device)
        a   = torch.LongTensor(actions).to(device)
        r   = torch.FloatTensor(rewards).to(device)
        ns  = torch.FloatTensor(n_states).to(device)
        nsc = torch.FloatTensor(n_scalars).to(device)
        d   = torch.FloatTensor(dones).to(device)
        m   = torch.BoolTensor(masks).to(device)

        # current Q for chosen action
        q_all = self.online_net(s, sc)                        # (B, 5)
        q_val = q_all.gather(1, a.unsqueeze(1)).squeeze(1)    # (B,)

        # Double DQN: online picks action, target evaluates
        with torch.no_grad():
            next_q_online = self.online_net(ns, nsc)
            next_q_online[~m] = -1e9                          # mask invalid
            next_actions = next_q_online.argmax(dim=1)        # (B,)
            next_q_target = self.target_net(ns, nsc)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q_val, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self.sync_target()

        return loss.item()


# ── RL Agent (per-ghost interface) ────────────────────────────────────
class RLAgent:
    def __init__(self, gid, model_path="ghostweights.pth",
                 shared_model=None, shared_target=None,
                 shared_trainer=None, shared_buffer=None):
        self.gid = gid
        self.model_path = model_path
        self.buffer = shared_buffer if shared_buffer is not None else ReplayBuffer()
        if TORCH_AVAILABLE:
            if shared_model is not None:
                self.model   = shared_model
                self.target  = shared_target
                self.trainer = shared_trainer
            else:
                self.model  = GhostDQN().to(device)
                self.target = GhostDQN().to(device)
                if os.path.exists(self.model_path):
                    try:
                        self.model.load_state_dict(
                            torch.load(self.model_path, map_location=device))
                        print(f"Ghost {self.gid} loaded weights from {self.model_path}")
                    except Exception as e:
                        print(f"Could not load weights: {e}")
                self.trainer = DQN_Trainer(self.model, self.target)
        else:
            self.model = self.target = self.trainer = None
        # per-agent state for training bookkeeping
        self.last_state   = None
        self.last_scalars = None
        self.last_action  = -1
        self.last_valid_mask = None

    # ── observation builder ───────────────────────────────────────────
    def _prepare_state(self, ghost):
        """Build 9-channel spatial + 8 scalar observation."""
        p_map = np.array(ghost.personal_map, dtype=np.int32)
        rows, cols = p_map.shape

        c0_wall    = (p_map == 1).astype(np.float32)
        c1_pellet  = (p_map == 2).astype(np.float32)
        c2_power   = (p_map == 3).astype(np.float32)
        c3_unknown = (p_map == -1).astype(np.float32)

        # belief probability map (raw normalised distribution)
        c4_belief = np.zeros((rows, cols), dtype=np.float32)
        if hasattr(ghost, 'belief_map') and ghost.belief_map._initialised:
            c4_belief = ghost.belief_map._b.astype(np.float32)

        # safety / danger map
        c5_safety = np.ones((rows, cols), dtype=np.float32)
        if hasattr(ghost, 'belief_map'):
            c5_safety = ghost.belief_map._safety.astype(np.float32)

        # self position
        c6_self = np.zeros((rows, cols), dtype=np.float32)
        c6_self[ghost.row, ghost.col] = 1.0

        # ally positions
        c7_allies = np.zeros((rows, cols), dtype=np.float32)
        if ghost.known_agents:
            for pos in ghost.known_agents.values():
                if pos != "UNKNOWN":
                    c7_allies[pos[0], pos[1]] = 1.0

        # pacman position (known or belief top-1)
        c8_pacman = np.zeros((rows, cols), dtype=np.float32)
        if ghost.known_pacman is not None:
            c8_pacman[ghost.known_pacman[0], ghost.known_pacman[1]] = 1.0
        elif ghost.last_lost_pacman is not None:
            c8_pacman[ghost.last_lost_pacman[0], ghost.last_lost_pacman[1]] = 0.5
        elif hasattr(ghost, 'belief_map') and ghost.belief_map._initialised:
            top = ghost.belief_map.top_cells(n=1)
            if top:
                c8_pacman[top[0][0], top[0][1]] = 0.3

        spatial = np.stack([c0_wall, c1_pellet, c2_power, c3_unknown,
                            c4_belief, c5_safety, c6_self, c7_allies,
                            c8_pacman], axis=0)

        # ── scalars ──
        powered   = 1.0 if ghost.pacman_powered else 0.0
        known     = 1.0 if ghost.known_pacman is not None else 0.0
        staleness = 0.0
        if ghost.known_pacman is not None:
            staleness = max(0.0, 1.0 - ((ghost.frame - ghost.pacman_last_seen) / 50.0))

        active_task = ghost.cbba_agent.get_active_task()
        if active_task is not None and active_task.target_pos is not None:
            target_row = float(active_task.target_pos[0]) / float(rows)
            target_col = float(active_task.target_pos[1]) / float(cols)
        else:
            target_row = 0.0
            target_col = 0.0

        # count power pellets remaining
        num_power = float(np.sum(c2_power)) / 28.0

        # count alive allies
        alive = sum(1 for p in ghost.known_agents.values() if p != "UNKNOWN")
        num_allies = float(alive) / 6.0

        # frames since Pacman sighting (normalised)
        fss = min(ghost.belief_map.frames_since_sighting, 200) / 200.0

        scalars = np.array([powered, known, staleness,
                            target_row, target_col,
                            num_power, num_allies, fss], dtype=np.float32)
        return {'spatial': spatial, 'scalars': scalars}

    # ── valid task-type mask ──────────────────────────────────────────
    def _get_valid_task_mask(self, ghost):
        """Return boolean mask of which task types are feasible."""
        mask = np.ones(NUM_ACTIONS, dtype=bool)

        # HUNT only if we have any idea where pacman is
        if ghost.known_pacman is None and ghost.last_lost_pacman is None:
            if not (hasattr(ghost, 'belief_map') and ghost.belief_map._initialised):
                mask[ACTION_HUNT] = False

        # CONVERT only if power pellets exist on the map
        has_power = any(ghost.personal_map[r][c] == POWER
                        for r in range(len(ghost.personal_map))
                        for c in range(len(ghost.personal_map[0])))
        if not has_power:
            mask[ACTION_CONVERT] = False

        # EVADE only if pacman is powered
        if not ghost.pacman_powered:
            mask[ACTION_EVADE] = False

        # at least one action must be valid
        if not mask.any():
            mask[:] = True
        return mask

    # ── epsilon-greedy action selection ────────────────────────────────
    def select_task(self, ghost, epsilon=0.0):
        """Return (task_type_idx, obs_dict) using epsilon-greedy."""
        obs = self._prepare_state(ghost)
        mask = self._get_valid_task_mask(ghost)

        if not TORCH_AVAILABLE or self.model is None:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions)), obs, mask

        if random.random() < epsilon:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions)), obs, mask

        state_t   = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
        mask_t    = torch.BoolTensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.model(state_t, scalars_t)           # (1, 5)
            q[~mask_t] = -1e9
            action = q.argmax(dim=-1).item()
        return int(action), obs, mask

    # ── generate CBBA task list from Q-values ─────────────────────────
    @staticmethod
    def generate_task_bids_from_q(ghost, q_values, mask, frame, chosen_action_idx=None):
        """Produce a list of Tasks scored by Q-values for CBBA."""
        # shift Q-values to be positive for bidding (CBBA expects positive scores)
        q_min = q_values.min()
        q_shifted = q_values - q_min + 0.1

        # if exploring, we can boost the chosen action heavily
        if chosen_action_idx is not None:
            q_shifted[chosen_action_idx] += 1000.0

        tasks = []
        # ACTION_HUNT
        if mask[ACTION_HUNT]:
            target = ghost.known_pacman or ghost.last_lost_pacman
            if target is None and hasattr(ghost, 'belief_map') and getattr(ghost.belief_map, '_initialised', False):
                top = ghost.belief_map.top_cells(n=1)
                target = top[0] if top else None
            if target is not None:
                tasks.append(Task(task_type=TaskType.HUNT, target_pos=target,
                                  score=float(q_shifted[ACTION_HUNT]),
                                  created_frame=frame))

        # ACTION_HUNT_BELIEF
        if mask[ACTION_HUNT_BELIEF] and hasattr(ghost, 'belief_map') and getattr(ghost.belief_map, '_initialised', False):
            top = ghost.belief_map.top_cells(n=1)
            if top:
                tasks.append(Task(task_type=TaskType.DYNAMIC, target_pos=top[0],
                                  score=float(q_shifted[ACTION_HUNT_BELIEF]),
                                  created_frame=frame))

        # ACTION_CONVERT
        if mask[ACTION_CONVERT]:
            rows_m = len(ghost.personal_map)
            cols_m = len(ghost.personal_map[0])
            best_power = None
            best_dist  = 1e9
            for r in range(rows_m):
                for c in range(cols_m):
                    if ghost.personal_map[r][c] == POWER:
                        d = abs(r - ghost.row) + abs(c - ghost.col)
                        if d < best_dist:
                            best_dist = d
                            best_power = (r, c)
            if best_power is not None:
                tasks.append(Task(task_type=TaskType.CONVERT, target_pos=best_power,
                                  score=float(q_shifted[ACTION_CONVERT]),
                                  created_frame=frame))

        # ACTION_EVADE
        if mask[ACTION_EVADE]:
            pac = ghost.known_pacman
            if pac is not None:
                # find farthest reachable cell from Pacman
                rows_m = len(ghost.personal_map)
                cols_m = len(ghost.personal_map[0])
                best_pos  = (ghost.row, ghost.col)
                best_dist = -1
                for r in range(rows_m):
                    for c in range(cols_m):
                        if ghost.personal_map[r][c] in (WALL, UNKNOWN):
                            continue
                        d = abs(r - pac[0]) + abs(c - pac[1])
                        if d > best_dist:
                            best_dist = d
                            best_pos  = (r, c)
                tasks.append(Task(task_type=TaskType.EVADE_TRACK, target_pos=best_pos,
                                  score=float(q_shifted[ACTION_EVADE]),
                                  created_frame=frame))

        # ACTION_EXPLORE
        if mask[ACTION_EXPLORE]:
            explore_tasks = generate_tasks(ghost, frame)
            explore_only = [t for t in explore_tasks if t.task_type == TaskType.EXPLORE]
            if explore_only:
                best_explore = explore_only[0]
                tasks.append(Task(task_type=TaskType.EXPLORE,
                                  target_pos=best_explore.target_pos,
                                  score=float(q_shifted[ACTION_EXPLORE]),
                                  created_frame=frame))

        if not tasks:
            # fallback: explore from allocator
            fallback = generate_tasks(ghost, frame)
            if fallback:
                tasks.append(fallback[0])

        tasks.sort(key=lambda t: t.score, reverse=True)
        return tasks

    def generate_task_bids(self, ghost, frame):
        """Produce a list of Tasks scored by DQN Q-values for CBBA."""
        obs = self._prepare_state(ghost)
        mask = self._get_valid_task_mask(ghost)

        if TORCH_AVAILABLE and self.model is not None:
            state_t   = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
            scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
            with torch.no_grad():
                q = self.model(state_t, scalars_t).squeeze(0).cpu().numpy()
        else:
            q = np.zeros(NUM_ACTIONS, dtype=np.float32)

        return self.generate_task_bids_from_q(ghost, q, mask, frame)

    def save_model(self):
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)