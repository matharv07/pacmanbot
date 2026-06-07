import os
import numpy as np
from allocator import Task, TaskType, WALL
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    device = "cpu"

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

class GhostActorCritic(None if not TORCH_AVAILABLE else nn.Module):
    """
    7-channel spatial input + 5 scalar inputs → 4-direction categorical actor + scalar critic.
    Channels: is_wall, is_pellet, is_power, is_unknown, belief_map, self_pos, ally_pos
    Scalars:  pacman_powered, pacman_known, pacman_staleness, target_row, target_col
    """
    def __init__(self, input_channels=7, rows=33, cols=41, num_scalars=5, num_actions=4):
        if not TORCH_AVAILABLE:
            return
        super(GhostActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        def conv2d_size_out(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(cols))
        convh = conv2d_size_out(conv2d_size_out(rows))
        linear_input_size = convw * convh * 128 + num_scalars
        self.shared_fc = nn.Linear(linear_input_size, 512)

        self.actor_fc = nn.Linear(512, 128)
        self.actor_out = nn.Linear(128, num_actions)

        self.critic_fc = nn.Linear(512, 128)
        self.critic_out = nn.Linear(128, 1)

    def forward(self, x, scalars=None):
        if not TORCH_AVAILABLE:
            return None, None
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if scalars is not None:
            x = torch.cat([x, scalars], dim=-1)
        shared_features = F.relu(self.shared_fc(x))

        actor_features = F.relu(self.actor_fc(shared_features))
        logits = self.actor_out(actor_features)

        critic_features = F.relu(self.critic_fc(shared_features))
        state_value = self.critic_out(critic_features)
        return logits, state_value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.scalars = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.returns = []
        self.valid_masks = []
    
    def clear(self):
        for lst in [self.states, self.scalars, self.actions, self.logprobs, self.rewards, self.values, self.dones, self.returns, self.valid_masks]:
            lst.clear()


class PPO_Trainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=6, mini_batch_size=2048):
        self.model = model
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mini_batch_size = mini_batch_size
        if TORCH_AVAILABLE:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.huber_loss = nn.HuberLoss(delta=1.0)
            
    def update(self, memory):
        if not TORCH_AVAILABLE or len(memory.states) < 2:
            memory.clear()
            return 0.0, 0.0, 0.0

        returns_t = torch.tensor(memory.returns, dtype=torch.float32).to(device)
        # Removed per-batch return normalization to preserve the absolute scale of the value function

        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_scalars = torch.FloatTensor(np.array(memory.scalars)).to(device)
        old_actions = torch.LongTensor(memory.actions).to(device)
        old_logprobs = torch.FloatTensor(memory.logprobs).to(device)
        old_values = torch.FloatTensor(memory.values).to(device)
        old_valid_masks = torch.BoolTensor(np.array(memory.valid_masks)).to(device)

        advantages = (returns_t - old_values).detach()
        adv_std = advantages.std()
        if adv_std > 1e-7:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-7)

        batch_size = len(memory.states)
        indices = np.arange(batch_size)
        last_loss, last_entropy, last_vloss = 0.0, 0.0, 0.0

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                mb_idx = indices[start:end]

                mb_states = old_states[mb_idx]
                mb_scalars = old_scalars[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_valid_masks = old_valid_masks[mb_idx]

                logits, state_values = self.model(mb_states, mb_scalars)
                # Mask invalid actions before computing distribution
                logits = logits.masked_fill(~mb_valid_masks, float('-inf'))
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy()

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                value_loss = self.huber_loss(state_values.squeeze(-1), mb_returns)
                loss = -torch.min(surr1, surr2) + 0.5 * value_loss - 0.03 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                last_loss = loss.mean().item()
                last_entropy = dist_entropy.mean().item()
                last_vloss = value_loss.item()

        memory.clear()
        return last_loss, last_entropy, last_vloss


class RLAgent:
    def __init__(self, gid, pathfinder_model_path="ghostweights.pth", shared_model=None, shared_trainer=None, shared_memory=None):
        self.gid = gid
        self.model_path = pathfinder_model_path
        self.memory = shared_memory if shared_memory is not None else PPOMemory()
        if TORCH_AVAILABLE:
            if shared_model is not None:
                self.model = shared_model
                self.trainer = shared_trainer
            else:
                self.model = GhostActorCritic().to(device)
                if os.path.exists(self.model_path):
                    try:
                        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                        print(f"Ghost {self.gid} loaded weights from {self.model_path}")
                    except Exception as e:
                        print(f"Could not load weights: {e}")
                self.trainer = PPO_Trainer(self.model)
        else:
            self.model = None
            self.trainer = None

    def _prepare_state(self, ghost):
        """Build 7-channel spatial + 5 scalar observation from ghost state."""
        p_map = np.array(ghost.personal_map, dtype=np.int32)
        rows, cols = p_map.shape
        c1_wall    = (p_map == 1).astype(np.float32)
        c2_pellet  = (p_map == 2).astype(np.float32)
        c3_power   = (p_map == 3).astype(np.float32)
        c4_unknown = (p_map == -1).astype(np.float32)

        c5_belief = ghost.get_target_proximity_channel()

        c6_self = np.zeros((rows, cols), dtype=np.float32)
        c6_self[ghost.row, ghost.col] = 1.0

        active_task = ghost.cbba_agent.get_active_task()
        if active_task is not None and active_task.target_pos is not None:
            target_row = float(active_task.target_pos[0]) / float(rows)
            target_col = float(active_task.target_pos[1]) / float(cols)
        else:
            target_row = 0.0
            target_col = 0.0

        c7_allies = np.zeros((rows, cols), dtype=np.float32)
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
        scalars = np.array([powered, known, staleness, target_row, target_col], dtype=np.float32)
        return {'spatial': spatial, 'scalars': scalars}

    def _get_valid_mask(self, ghost):
        """Return boolean mask of valid cardinal directions."""
        rows, cols = len(ghost.grid), len(ghost.grid[0])
        mask = np.zeros(4, dtype=bool)
        for i, (dr, dc) in enumerate(DIRS):
            nr, nc = ghost.row + dr, ghost.col + dc
            if 0 <= nr < rows and 0 <= nc < cols and ghost.grid[nr][nc] != WALL:
                mask[i] = True
        return mask

    def get_next_step(self, ghost):
        """Game-time inference: return (next_pos, state, action_idx)."""
        obs = self._prepare_state(ghost)
        if not TORCH_AVAILABLE or self.model is None:
            return None, obs['spatial'], -1
        valid_mask = self._get_valid_mask(ghost)
        if not valid_mask.any():
            return None, obs['spatial'], -1
        state_t = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
        mask_t = torch.BoolTensor(valid_mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = self.model(state_t, scalars_t)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            action_idx = torch.argmax(logits, dim=-1).item()
        dr, dc = DIRS[action_idx]
        return (ghost.row + dr, ghost.col + dc), obs['spatial'], action_idx

    def generate_dynamic_tasks(self, ghost, frame, num_tasks=5):
        """Use critic value + belief map top cells to generate CBBA tasks."""
        if not TORCH_AVAILABLE or self.model is None:
            return []
        obs = self._prepare_state(ghost)
        state_t = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
        with torch.no_grad():
            _, state_value = self.model(state_t, scalars_t)
        base_value = max(0.1, state_value.item())
        top_cells = ghost.belief_map.top_cells(n=num_tasks)
        tasks = []
        for pos in top_cells:
            prob = ghost.belief_map.probability_at(pos)
            bid = base_value * prob * 10.0
            tasks.append(Task(task_type=TaskType.DYNAMIC, target_pos=pos, score=bid, created_frame=frame))
        tasks.sort(key=lambda t: t.score, reverse=True)
        return tasks

    def get_bid(self, ghost):
        obs = self._prepare_state(ghost)
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0
        state_t = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.model(state_t, scalars_t)
        return value.item()

    def get_macro_waypoint(self, ghost):
        obs = self._prepare_state(ghost)
        if not TORCH_AVAILABLE or self.model is None:
            return (ghost.row, ghost.col), np.array([0, 0]), 0.0, 0.0, obs['spatial']
        state_t = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
        scalars_t = torch.FloatTensor(obs['scalars']).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = self.model(state_t, scalars_t)
        top = ghost.belief_map.top_cells(n=1)
        target = top[0] if top else (ghost.row, ghost.col)
        valid_mask = self._get_valid_mask(ghost)
        mask_t = torch.BoolTensor(valid_mask).unsqueeze(0).to(device)
        logits = logits.masked_fill(~mask_t, float('-inf'))
        action_idx = torch.argmax(logits, dim=-1).item() if valid_mask.any() else 0
        return target, np.array(DIRS[action_idx]), 0.0, value.item(), obs['spatial']

    def save_model(self):
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)