import os
import time
import datetime
import numpy as np
import torch
from torch.distributions import Categorical
from vector_env import SubprocVecEnv 
from rl_agent import RLAgent, GhostActorCritic, PPO_Trainer, PPOMemory, TORCH_AVAILABLE, device

def compute_discounted_returns(rewards, dones, next_value=0.0, gamma=0.99):
    """Compute discounted returns for a single trajectory (one agent in one env)."""
    returns = []
    discounted = next_value
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted = 0.0
        discounted = reward + gamma * discounted
        returns.append(discounted)
    returns.reverse()
    return returns

def train(max_training_steps=10000000):
    NUM_ENVS = 10
    UPDATE_TIMESTEP = 2048
    GAMMA = 0.99
    print(f"Starting PPO Headless MARL Training -- {NUM_ENVS} parallel environments!!!")
    vec_env = SubprocVecEnv(num_envs=NUM_ENVS)
    if not TORCH_AVAILABLE:
        print("Torch not found. Exiting.")
        return
    shared_model = GhostActorCritic(input_channels=7, rows=33, cols=41, num_scalars=5, num_actions=4).to(device)
    model_path = "ghostweights.pth"
    if os.path.exists(model_path):
        try:
            shared_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    shared_trainer = PPO_Trainer(shared_model, k_epochs=6, mini_batch_size=2048)
    shared_memory = PPOMemory()
    agents = {i: RLAgent(i, shared_model=shared_model, shared_trainer=shared_trainer, shared_memory=shared_memory) for i in range(7)}  
    obs_list, info_list = vec_env.reset()
    start_time = time.time()
    time_step = 0
    episodes_completed = 0
    total_transitions = 0

    accumulated_rewards = {env_idx: {gid: 0.0 for gid in range(7)} for env_idx in range(NUM_ENVS)}
    trajectory_buffers = {}
    
    def get_traj_buf(env_idx, gid):
        key = (env_idx, gid)
        if key not in trajectory_buffers:
            trajectory_buffers[key] = {'states': [], 'scalars': [], 'actions': [], 'logprobs': [], 'values': [], 'rewards': [], 'dones': [], 'valid_masks': []}
        return trajectory_buffers[key]

    def flush_trajectories_to_memory(next_obs_list):
        for key, buf in trajectory_buffers.items():
            if len(buf['rewards']) == 0:
                continue
            
            env_idx, gid = key
            last_done = buf['dones'][-1]
            next_value = 0.0
            
            if not last_done and gid in next_obs_list[env_idx]:
                obs_dict = next_obs_list[env_idx][gid]
                state_t = torch.FloatTensor(obs_dict['spatial']).unsqueeze(0).to(device)
                scalars_t = torch.FloatTensor(obs_dict['scalars']).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, state_value = shared_model(state_t, scalars_t)
                next_value = state_value.item()
                
            returns = compute_discounted_returns(buf['rewards'], buf['dones'], next_value, GAMMA)
            shared_memory.states.extend(buf['states'])
            shared_memory.scalars.extend(buf['scalars'])
            shared_memory.actions.extend(buf['actions'])
            shared_memory.logprobs.extend(buf['logprobs'])
            shared_memory.values.extend(buf['values'])
            shared_memory.rewards.extend(buf['rewards'])
            shared_memory.dones.extend(buf['dones'])
            shared_memory.returns.extend(returns)
            shared_memory.valid_masks.extend(buf['valid_masks'])
        for buf in trajectory_buffers.values():
            for v in buf.values():
                v.clear()

    try:
        while time_step < max_training_steps:
            env_actions = [{} for _ in range(NUM_ENVS)]
            action_data_store = {} 
            for env_idx in range(NUM_ENVS):
                obs = obs_list[env_idx]
                valid_actions_dict = info_list[env_idx].get('valid_actions', {})
                action_data_store[env_idx] = {}
                for gid, obs_dict in obs.items():
                    spatial = obs_dict['spatial']
                    scalars_arr = obs_dict['scalars']
                    # Build valid action mask
                    valid_dirs = valid_actions_dict.get(gid, [0, 1, 2, 3])
                    valid_mask = np.zeros(4, dtype=bool)
                    for idx in valid_dirs:
                        valid_mask[idx] = True
                    if not valid_mask.any():
                        valid_mask[:] = True  # fallback: allow all

                    state_t = torch.FloatTensor(spatial).unsqueeze(0).to(device)
                    scalars_t = torch.FloatTensor(scalars_arr).unsqueeze(0).to(device)
                    mask_t = torch.BoolTensor(valid_mask).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, state_value = shared_model(state_t, scalars_t)
                        logits = logits.masked_fill(~mask_t, float('-inf'))
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    action_int = action.item()
                    env_actions[env_idx][gid] = action_int
                    action_data_store[env_idx][gid] = (spatial, scalars_arr, action_int, log_prob.item(), state_value.item(), valid_mask)

            next_obs_list, rewards_list, agent_dones_list, env_dones_list, next_info_list = vec_env.step(env_actions)
            
            for env_idx in range(NUM_ENVS):
                action_executed = next_info_list[env_idx].get('action_executed', {})
                env_done = env_dones_list[env_idx]
                if env_done:
                    episodes_completed += 1
                    
                for gid in range(7):
                    if gid in rewards_list[env_idx]:
                        accumulated_rewards[env_idx][gid] += rewards_list[env_idx][gid]

                for gid in action_data_store[env_idx]:
                    done = agent_dones_list[env_idx].get(gid, False)
                    executed = action_executed.get(gid, False)
                    
                    if executed or done:
                        spatial, scalars_arr, action_int, log_prob, value, valid_mask = action_data_store[env_idx][gid]
                        reward = accumulated_rewards[env_idx][gid]
                        
                        buf = get_traj_buf(env_idx, gid)
                        buf['states'].append(spatial)
                        buf['scalars'].append(scalars_arr)
                        buf['actions'].append(action_int)
                        buf['logprobs'].append(log_prob)
                        buf['values'].append(value)
                        buf['rewards'].append(reward)
                        buf['dones'].append(done)
                        buf['valid_masks'].append(valid_mask)
                        total_transitions += 1
                        
                        accumulated_rewards[env_idx][gid] = 0.0

                if env_done:
                    for gid in range(7):
                        accumulated_rewards[env_idx][gid] = 0.0

            time_step += 1
            if time_step % UPDATE_TIMESTEP == 0:
                elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                flush_trajectories_to_memory(next_obs_list)
                if len(shared_memory.states) > 0:
                    total_loss, entropy, value_loss = shared_trainer.update(shared_memory)
                    agents[0].save_model()                
                    print(f"[{elapsed}] Eps: {episodes_completed} | Trans: {total_transitions} | Loss: {total_loss:.3f} | Value Loss: {value_loss:.3f} | Entropy: {entropy:.3f}")
                else:
                    print(f"[{elapsed}] Eps: {episodes_completed} | No transitions to train on")
            obs_list = next_obs_list
            info_list = next_info_list
            
    finally:
        vec_env.close()

if __name__ == "__main__":
    train()