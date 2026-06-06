import math
import random
import os
from collections import deque
import numpy as np
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

class GhostRLNetwork(None if not TORCH_AVAILABLE else nn.Module):
    def __init__(self, input_channels=10, output_dim=4):
        if TORCH_AVAILABLE:
            super(GhostRLNetwork, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 10, 512)
            self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN_Trainer:
    def __init__(self, model, target_model, lr=1e-4, gamma=0.99):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        if TORCH_AVAILABLE:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.criterion = nn.SmoothL1Loss()
            
    def train_step(self, buffer, batch_size):
        if not TORCH_AVAILABLE or len(buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
    def sync_target(self):
        if TORCH_AVAILABLE:
            self.target_model.load_state_dict(self.model.state_dict())

class RLAgent:
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  #UP, DOWN, LEFT, RIGHT
    
    def __init__(self, gid, pathfinder_model_path="ghostweights.pth", shared_model=None, shared_target=None, shared_trainer=None, shared_buffer=None):
        self.gid = gid
        self.model_path = pathfinder_model_path
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.buffer = shared_buffer if shared_buffer is not None else ReplayBuffer(capacity=100000)
        self.batch_size = 512
        self.sync_target_frames = 1000
        self.frame_count = 0
        if TORCH_AVAILABLE:
            if shared_model is not None and shared_target is not None:
                self.model = shared_model
                self.target_model = shared_target
                self.trainer = shared_trainer
            else:
                self.model = GhostRLNetwork(input_channels=10, output_dim=4).to(device)
                self.target_model = GhostRLNetwork(input_channels=10, output_dim=4).to(device)
                if os.path.exists(self.model_path):
                    try:
                        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                        print(f"Ghost {self.gid} loaded weights from {self.model_path}")
                    except Exception as e:
                        print(f"Could not load weights: {e}")
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()
                self.trainer = DQN_Trainer(self.model, self.target_model)
        else:
            self.model = None
            self.target_model = None
            self.trainer = None

    def _prepare_state(self, ghost):
        rows, cols = len(ghost.personal_map), len(ghost.personal_map[0])
        p_map = np.array(ghost.personal_map, dtype=np.int32)
        one_hot = (p_map[None, :, :] == np.arange(4)[:, None, None]).astype(np.float32)
        b_map = np.zeros((rows, cols), dtype=np.float32)
        if hasattr(ghost.belief_map, '_initialised') and ghost.belief_map._initialised:
            b_map = np.array(ghost.belief_map._b, dtype=np.float32)
        target_map = np.zeros((rows, cols), dtype=np.float32)
        r_idxs = []
        c_idxs = []
        intensities = []
        for idx, task_key in enumerate(ghost.cbba_agent.path):
            task = ghost.cbba_agent._task_map.get(task_key)
            if task and hasattr(task, 'target_pos') and task.target_pos:
                tr, tc = task.target_pos
                intensity = max(0.1, 1.0 * (0.7 ** idx))
                r_idxs.append(tr); c_idxs.append(tc); intensities.append(intensity)
        if r_idxs:
            np.maximum.at(target_map, (np.array(r_idxs, dtype=int), np.array(c_idxs, dtype=int)), np.array(intensities, dtype=np.float32))
        ghost_map = np.zeros((rows, cols), dtype=np.float32)
        ghost_map[ghost.row][ghost.col] = 1.0
        ally_map = np.zeros((rows, cols), dtype=np.float32)
        if ghost.known_agents:
            positions = [pos for pos in ghost.known_agents.values() if pos is not None and pos != "UNKNOWN"]
            if positions:
                rs, cs = zip(*positions)
                ally_map[np.array(rs, dtype=int), np.array(cs, dtype=int)] = 1.0
        pacman_map = np.zeros((rows, cols), dtype=np.float32)
        if ghost.known_pacman is not None:
            pacman_map[ghost.known_pacman[0]][ghost.known_pacman[1]] = 1.0
        pacman_last_seen_map = np.zeros((rows, cols), dtype=np.float32)
        if ghost.pacman_last_seen is not None and ghost.pacman_last_seen >= 0:
            pacman_last_seen_map[:] = np.clip(ghost.pacman_last_seen / 1000.0, 0.0, 1.0)
        state = np.concatenate([one_hot, np.expand_dims(b_map, axis=0), np.expand_dims(target_map, axis=0), np.expand_dims(ghost_map, axis=0), np.expand_dims(ally_map, axis=0), np.expand_dims(pacman_map, axis=0), np.expand_dims(pacman_last_seen_map, axis=0)], axis=0)
        return state
                
    def save_model(self):
        if TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), self.model_path)

    def score_tasks(self, tasks, personal_map, belief_map, ghost_pos):          #for RL based task scoring -- setup only currently we use djikstra
        if not tasks:
            return []
        for t in tasks:
            t.score = random.uniform(0.1, 10.0)
        return tasks

    def get_next_step(self, ghost, training_mode=False):
        state = self._prepare_state(ghost)
        rows, cols = len(ghost.grid), len(ghost.grid[0])
        valid_actions = []
        for i, (dr, dc) in enumerate(self.DIRS):
            nr, nc = ghost.row + dr, ghost.col + dc
            if 0 <= nr < rows and 0 <= nc < cols and ghost.grid[nr][nc] != 1:
                valid_actions.append(i)
        reverse_idx = -1
        if hasattr(ghost, 'last_dir') and ghost.last_dir:
            for i, (dr, dc) in enumerate(self.DIRS):
                if dr == -ghost.last_dir[0] and dc == -ghost.last_dir[1]:
                    reverse_idx = i
                    break
        if len(valid_actions) > 1 and reverse_idx in valid_actions:
            valid_actions.remove(reverse_idx)
        if not valid_actions:
            return (ghost.row, ghost.col), state, -1
        action_idx = self.act(state, valid_actions, training_mode)
        if action_idx == -1:
            return (ghost.row, ghost.col), state, -1
        dr, dc = self.DIRS[action_idx]
        nr, nc = ghost.row + dr, ghost.col + dc
        return (nr, nc), state, action_idx
        
    def act(self, state, valid_actions, training_mode=False):
        if not valid_actions:
            return -1
        action_idx = -1
        if training_mode and random.random() < self.epsilon:
            action_idx = random.choice(valid_actions)
        elif TORCH_AVAILABLE and self.model is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.model(state_tensor).cpu().numpy()[0]
                for i in range(4):
                    if i not in valid_actions:
                        q_values[i] = -float('inf')
                action_idx = int(np.argmax(q_values))
        else:
            action_idx = random.choice(valid_actions)
        return action_idx
        
    def train(self):
        if not TORCH_AVAILABLE or self.trainer is None:
            return
        self.trainer.train_step(self.buffer, self.batch_size)
        self.frame_count += 1
        if self.frame_count % self.sync_target_frames == 0:
            self.trainer.sync_target()
            self.save_model()