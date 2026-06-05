import os
import time
import datetime
import numpy as np
import random
import torch
from vector_env import SubprocVecEnv 
from rl_agent import RLAgent, GhostRLNetwork, DQN_Trainer, ReplayBuffer, TORCH_AVAILABLE, device

def train(episodes=1000, start_episode=0):
    NUM_ENVS = 10
    print(f"Starting headless MARL training -- {NUM_ENVS} parallel environments!!!")
    vec_env = SubprocVecEnv(num_envs=NUM_ENVS)
    if TORCH_AVAILABLE:
        shared_model = GhostRLNetwork(input_channels=10, output_dim=4).to(device)
        shared_target = GhostRLNetwork(input_channels=10, output_dim=4).to(device)
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
    for gid in agents:
        agents[gid].batch_size = 2048  
        agents[gid].epsilon = max(agents[gid].epsilon_min, (agents[gid].epsilon_decay ** start_episode))
    ep_counts = [0] * NUM_ENVS
    total_rewards = [0] * NUM_ENVS
    obs_list, info_list = vec_env.reset()
    start_time = time.time()
    try:
        while sum(ep_counts) < episodes:
            all_states = []
            valid_actions_list = []
            active_gids = []
            env_indices = []
            for env_idx in range(NUM_ENVS):
                obs = obs_list[env_idx]
                info = info_list[env_idx]
                for gid, state in obs.items():
                    valid_actions = info.get('valid_actions', {}).get(gid, [0, 1, 2, 3])
                    if not valid_actions:
                        continue            
                    all_states.append(state)
                    valid_actions_list.append(valid_actions)
                    active_gids.append(gid)
                    env_indices.append(env_idx)
            q_values_batch = None
            if all_states and TORCH_AVAILABLE and shared_model is not None:
                states_tensor = torch.FloatTensor(np.array(all_states)).to(device)
                with torch.no_grad():
                    q_values_batch = shared_model(states_tensor).cpu().numpy()
            env_actions = [{} for _ in range(NUM_ENVS)]
            flat_action_indices = []
            for i in range(len(all_states)):
                gid = active_gids[i]
                env_idx = env_indices[i]
                epsilon = agents[gid].epsilon
                valid_actions = valid_actions_list[i]
                if random.random() < epsilon or q_values_batch is None:
                    action_idx = random.choice(valid_actions)
                else:
                    q_vals = q_values_batch[i].copy()
                    for a_idx in range(4):
                        if a_idx not in valid_actions:
                            q_vals[a_idx] = -float('inf')
                    action_idx = int(np.argmax(q_vals))
                env_actions[env_idx][gid] = RLAgent.DIRS[action_idx]
                flat_action_indices.append(action_idx)
            next_obs_list, rewards_list, agent_dones_list, env_dones_list, next_info_list = vec_env.step(env_actions)
            for i in range(len(all_states)):
                gid = active_gids[i]
                env_idx = env_indices[i]
                state = all_states[i]
                action = flat_action_indices[i]
                reward = rewards_list[env_idx].get(gid, 0)
                next_state = next_obs_list[env_idx].get(gid, state)
                done = agent_dones_list[env_idx].get(gid, False)
                executed = next_info_list[env_idx].get('action_executed', {}).get(gid, False)
                if executed or done:
                    agents[gid].buffer.push(state, action, reward, next_state, done)
            for env_idx in range(NUM_ENVS):
                total_rewards[env_idx] += sum(rewards_list[env_idx].values())
            if all_states:
                agents[0].train()
            for env_idx in range(NUM_ENVS):
                if env_dones_list[env_idx]:
                    ep_counts[env_idx] += 1
                    completed_eps = sum(ep_counts)
                    elapsed_seconds = int(time.time() - start_time)
                    formatted_time = str(datetime.timedelta(seconds=elapsed_seconds))
                    final_info = next_info_list[env_idx].get('terminal_info', {})
                    steps = final_info.get('step_count', 0)
                    p_score = final_info.get('player_score', 0)
                    print(f"[{formatted_time}] Total Eps: {completed_eps}/{episodes} | Env {env_idx} | Steps: {steps} | Pacman Score: {p_score} | Ghost Score: {total_rewards[env_idx]:.2f} | Epsilon: {agents[0].epsilon:.3f}")
                    total_rewards[env_idx] = 0
                    if agents[0].epsilon > agents[0].epsilon_min:
                        new_epsilon = agents[0].epsilon * agents[0].epsilon_decay
                        for gid in agents:
                            agents[gid].epsilon = max(agents[0].epsilon_min, new_epsilon)
            obs_list = next_obs_list
            info_list = next_info_list
    finally:
        vec_env.close()

if __name__ == "__main__":
    train(episodes=1000, start_episode=0)