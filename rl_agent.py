import math
import random
import os
from collections import deque
import numpy as np

# Placeholders for PyTorch if available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    device = torch.device("cpu") # Forced to CPU to avoid OOM with 7 agents
except ImportError:
    TORCH_AVAILABLE = False
    device = "cpu"

class GhostRLNetwork(None if not TORCH_AVAILABLE else nn.Module):
    """
    A unified Convolutional Neural Network that can be used for Low-Level Pathfinding.
    Inputs: [Batch, Channels, Rows, Cols]
    Channels: 3 (Personal Map, Belief Map, Target Location Map)
    Output: Q-values for 4 possible directions (UP, DOWN, LEFT, RIGHT)
    """
    def __init__(self, input_channels=3, output_dim=4):
        if TORCH_AVAILABLE:
            super(GhostRLNetwork, self).__init__()
            # Board size is 33 rows by 41 cols
            self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Flatten size: 64 channels * 33 * 41 = 86592
            self.fc1 = nn.Linear(64 * 33 * 41, 512)
            self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
            self.criterion = nn.MSELoss()
            
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
        self.optimizer.step()
        
    def sync_target(self):
        if TORCH_AVAILABLE:
            self.target_model.load_state_dict(self.model.state_dict())

class RLAgent:
    """
    Wrapper class to interface the RL models with the Pacman game loop.
    """
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
    
    def __init__(self, gid, pathfinder_model_path="ghost_pathfinder.pth"):
        self.gid = gid
        self.model_path = pathfinder_model_path
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.sync_target_frames = 1000
        self.frame_count = 0
        
        if TORCH_AVAILABLE:
            self.model = GhostRLNetwork(input_channels=3, output_dim=4).to(device)
            self.target_model = GhostRLNetwork(input_channels=3, output_dim=4).to(device)
            
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

    def _prepare_state(self, personal_map, belief_map, target_pos):
        """
        Converts the raw game state into a 3-channel tensor:
        [Personal Map, Belief Map, Target Map]
        """
        rows, cols = len(personal_map), len(personal_map[0])
        p_map = np.array(personal_map, dtype=np.float32)
        b_map = np.zeros_like(p_map)
        if hasattr(belief_map, '_b'):
            for r in range(rows):
                for c in range(cols):
                    b_map[r][c] = belief_map._b[r][c]
                    
        t_map = np.zeros_like(p_map)
        if target_pos:
            tr, tc = target_pos
            if 0 <= tr < rows and 0 <= tc < cols:
                t_map[tr][tc] = 1.0
                
        state = np.stack([p_map, b_map, t_map], axis=0)
        return state
        
    def save_model(self):
        if TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), self.model_path)

    def score_tasks(self, tasks, personal_map, belief_map, ghost_pos):
        """
        We still use randomized fallback for high-level tasks to focus on training the pathfinder first.
        """
        if not tasks:
            return []
        for t in tasks:
            t.score = random.uniform(0.1, 10.0)
        return tasks

    def get_next_step(self, personal_map, belief_map, current_pos, target_pos, training_mode=False):
        """
        Predicts the optimal immediate next step (dr, dc) to reach target_pos.
        If training_mode is True, uses epsilon-greedy.
        Returns: next_pos (tuple), state (numpy array), action_idx (int)
        """
        state = self._prepare_state(personal_map, belief_map, target_pos)
        rows, cols = len(personal_map), len(personal_map[0])
        
        valid_actions = []
        for i, (dr, dc) in enumerate(self.DIRS):
            nr, nc = current_pos[0] + dr, current_pos[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and personal_map[nr][nc] != 1: # 1 is WALL
                valid_actions.append(i)
                
        if not valid_actions:
            return current_pos, state, -1

        action_idx = -1
        if training_mode and random.random() < self.epsilon:
            # Explore
            action_idx = random.choice(valid_actions)
        elif TORCH_AVAILABLE and self.model is not None:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.model(state_tensor).cpu().numpy()[0]
                # Filter out invalid actions
                for i in range(4):
                    if i not in valid_actions:
                        q_values[i] = -float('inf')
                action_idx = int(np.argmax(q_values))
        else:
            # Fallback
            action_idx = random.choice(valid_actions)
            
        dr, dc = self.DIRS[action_idx]
        nr, nc = current_pos[0] + dr, current_pos[1] + dc
        
        return (nr, nc), state, action_idx
        
    def train(self):
        if not TORCH_AVAILABLE or self.trainer is None:
            return
        self.trainer.train_step(self.buffer, self.batch_size)
        self.frame_count += 1
        
        if self.frame_count % self.sync_target_frames == 0:
            self.trainer.sync_target()
            self.save_model()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
