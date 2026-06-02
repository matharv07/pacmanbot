import os
import time
from rl_env import PacmanMultiAgentEnv
from rl_agent import RLAgent, GhostRLNetwork, DQN_Trainer, ReplayBuffer, TORCH_AVAILABLE, device
import torch
import pacman

def train(episodes=1000):
    pacman.AUTO_MODE = True
    print("Starting headless MARL training!!!")
    env = PacmanMultiAgentEnv(max_steps=500)
    if TORCH_AVAILABLE:
        shared_model = GhostRLNetwork(input_channels=6, output_dim=4).to(device)
        shared_target = GhostRLNetwork(input_channels=6, output_dim=4).to(device)
        model_path = "ghostweights.pth"
        if os.path.exists(model_path):
            try:
                shared_model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded weights from {model_path}")
            except Exception as e:
                print(f"Could not load weights: {e}")
        shared_target.load_state_dict(shared_model.state_dict())
        shared_target.eval()
        shared_trainer = DQN_Trainer(shared_model, shared_target)
        shared_buffer = ReplayBuffer(100000)
    else:
        shared_model = shared_target = shared_trainer = shared_buffer = None
    agents = {i: RLAgent(i, shared_model=shared_model, shared_target=shared_target, shared_trainer=shared_trainer, shared_buffer=shared_buffer) for i in range(7)}
    for ep in range(episodes):
        obs, info = env.reset()
        for gid, ghost in env.ghosts.items():
            ghost.rl_agent = agents[gid]
        terminated = False
        truncated = False
        total_reward = 0
        while not (terminated or truncated):
            actions = {}
            action_indices = {}
            current_states = {}
            for gid, state in obs.items():
                ghost = env.ghosts[gid]
                valid_actions = []
                for i, (dr, dc) in enumerate(RLAgent.DIRS):
                    nr, nc = ghost.row + dr, ghost.col + dc
                    if 0 <= nr < len(ghost.grid) and 0 <= nc < len(ghost.grid[0]) and ghost.grid[nr][nc] != 1:
                        valid_actions.append(i)
                if not valid_actions:
                    continue            
                action_idx = agents[gid].act(state, valid_actions, training_mode=True)
                actions[gid] = RLAgent.DIRS[action_idx]
                action_indices[gid] = action_idx
                current_states[gid] = state
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            for gid, state in current_states.items():
                reward = rewards.get(gid, 0)
                next_state = next_obs.get(gid, state)
                done = terminated or truncated
                agents[gid].buffer.push(state, action_indices[gid], reward, next_state, done)
            if current_states:
                agents[0].train()
                for gid in agents:
                    agents[gid].epsilon = agents[0].epsilon
            obs = next_obs
            total_reward += sum(rewards.values())
        print(f"Episode {ep+1}/{episodes} | Steps: {info['step_count']} | Pacman Score: {info['player_score']} | Total Reward: {total_reward:.2f}")
    print("Training finished.")

if __name__ == "__main__":
    train()