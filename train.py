import os
import time
import datetime
import numpy as np
import torch
from vector_env import SubprocVecEnv
from rl_agent import (RLAgent, GhostDQN, DQN_Trainer, ReplayBuffer,
                      TORCH_AVAILABLE, device, NUM_ACTIONS)

# ── Hyperparameters ───────────────────────────────────────────────────
NUM_ENVS           = 10
WARMUP_STEPS       = 5000        # fill buffer before training
BATCH_SIZE         = 512
GAMMA              = 0.99
LR                 = 1e-4
TARGET_SYNC        = 2000        # hard-sync target net every N train steps
BUFFER_CAPACITY    = 200_000
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY_STEPS = 500_000    # linear decay over this many env steps
DECISION_EVERY     = 5           # ghost makes a new task decision every N env frames
SAVE_EVERY         = 5000        # save model every N env steps
LOG_EVERY          = 500         # print stats every N env steps
MODEL_PATH         = "ghostweights.pth"


def get_epsilon(step):
    """Linear epsilon decay."""
    frac = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def train(max_training_steps=10_000_000):
    print(f"Starting DQN Headless MARL Training -- {NUM_ENVS} parallel environments")
    print(f"  Buffer: {BUFFER_CAPACITY}  |  Batch: {BATCH_SIZE}  |  "
          f"Target sync: {TARGET_SYNC}  |  Decision every: {DECISION_EVERY} frames")

    if not TORCH_AVAILABLE:
        print("Torch not found. Exiting.")
        return

    # shared model, target, trainer, replay buffer
    online_net = GhostDQN().to(device)
    target_net = GhostDQN().to(device)
    target_net.eval()

    if os.path.exists(MODEL_PATH):
        try:
            online_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load weights (architecture mismatch?): {e}")

    shared_buffer  = ReplayBuffer(capacity=BUFFER_CAPACITY)
    shared_trainer = DQN_Trainer(online_net, target_net, lr=LR, gamma=GAMMA,
                                  batch_size=BATCH_SIZE,
                                  target_sync_every=TARGET_SYNC)

    agents = {i: RLAgent(i, model_path=MODEL_PATH,
                          shared_model=online_net, shared_target=target_net,
                          shared_trainer=shared_trainer,
                          shared_buffer=shared_buffer)
              for i in range(7)}

    vec_env = SubprocVecEnv(num_envs=NUM_ENVS)
    obs_list, info_list = vec_env.reset()

    start_time = time.time()
    env_step = 0
    episodes_completed = 0
    total_transitions = 0
    running_loss = 0.0
    loss_count = 0

    # per-(env, ghost) bookkeeping for multi-step reward accumulation
    pending = {}   # (env_idx, gid) -> {state, scalars, action, mask, acc_reward}

    try:
        while env_step < max_training_steps:
            epsilon = get_epsilon(env_step)
            env_actions = [{} for _ in range(NUM_ENVS)]

            for env_idx in range(NUM_ENVS):
                obs = obs_list[env_idx]
                for gid, obs_dict in obs.items():
                    on_decision = (env_step % DECISION_EVERY == 0)
                    key = (env_idx, gid)

                    if on_decision:
                        # build a temporary ghost-like object from obs for select_task
                        action, q_np = _select_action_from_obs(
                            agents[gid], obs_dict, epsilon)
                        env_actions[env_idx][gid] = {'action': action, 'q_values': q_np}

                        # if there is a pending transition, close it out now
                        if key in pending:
                            p = pending[key]
                            shared_buffer.push(
                                p['state'], p['scalars'], p['action'],
                                p['acc_reward'],
                                obs_dict['spatial'], obs_dict['scalars'],
                                False, p['mask'])
                            total_transitions += 1

                        # open a new pending transition
                        valid_mask = obs_dict.get('valid_mask',
                                                   np.ones(NUM_ACTIONS, dtype=bool))
                        pending[key] = {
                            'state':   obs_dict['spatial'],
                            'scalars': obs_dict['scalars'],
                            'action':  action,
                            'mask':    valid_mask,
                            'acc_reward': 0.0,
                        }
                    else:
                        # keep previous action
                        if key in pending:
                            # if no decision, we don't need to send q_values again, 
                            # environment already has latest_rl_tasks
                            env_actions[env_idx][gid] = None
                        else:
                            env_actions[env_idx][gid] = None

            # step all envs
            next_obs_list, rewards_list, agent_dones_list, env_dones_list, next_info_list = \
                vec_env.step(env_actions)

            # accumulate rewards and handle terminal transitions
            for env_idx in range(NUM_ENVS):
                env_done = env_dones_list[env_idx]
                if env_done:
                    episodes_completed += 1

                for gid in range(7):
                    key = (env_idx, gid)
                    reward = rewards_list[env_idx].get(gid, 0.0)
                    done = agent_dones_list[env_idx].get(gid, False)

                    if key in pending:
                        pending[key]['acc_reward'] += reward

                    if done and key in pending:
                        p = pending[key]
                        # use current obs as terminal next_state
                        next_obs = next_obs_list[env_idx].get(gid)
                        if next_obs is not None:
                            ns = next_obs['spatial']
                            nsc = next_obs['scalars']
                        else:
                            ns = p['state']       # terminal placeholder
                            nsc = p['scalars']
                        shared_buffer.push(
                            p['state'], p['scalars'], p['action'],
                            p['acc_reward'], ns, nsc,
                            True, p['mask'])
                        total_transitions += 1
                        del pending[key]

            # train on a batch from replay buffer
            if len(shared_buffer) >= WARMUP_STEPS:
                loss = shared_trainer.update(shared_buffer)
                running_loss += loss
                loss_count += 1

            env_step += 1
            obs_list = next_obs_list
            info_list = next_info_list

            # logging
            if env_step % LOG_EVERY == 0:
                elapsed = str(datetime.timedelta(
                    seconds=int(time.time() - start_time)))
                avg_loss = running_loss / max(1, loss_count)
                buf_fill = len(shared_buffer)
                print(f"[{elapsed}] Step {env_step:,} | Eps: {episodes_completed} | "
                      f"Trans: {total_transitions:,} | Buf: {buf_fill:,} | "
                      f"Loss: {avg_loss:.4f} | ε: {epsilon:.3f} | "
                      f"TrainSteps: {shared_trainer.train_steps}")
                running_loss = 0.0
                loss_count = 0

            # save
            if env_step % SAVE_EVERY == 0 and len(shared_buffer) >= WARMUP_STEPS:
                agents[0].save_model()

    finally:
        vec_env.close()
        if len(shared_buffer) >= WARMUP_STEPS:
            agents[0].save_model()
            print("Final model saved.")


def _select_action_from_obs(agent, obs_dict, epsilon):
    """Pick an action using DQN given pre-built observation dict. Returns (action_idx, q_values_np)."""
    valid_mask = obs_dict.get('valid_mask', np.ones(NUM_ACTIONS, dtype=bool))
    
    if not TORCH_AVAILABLE or agent.model is None:
        q_zeros = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action = int(np.random.choice(np.where(valid_mask)[0]))
        return action, q_zeros

    state_t   = torch.FloatTensor(obs_dict['spatial']).unsqueeze(0).to(device)
    scalars_t = torch.FloatTensor(obs_dict['scalars']).unsqueeze(0).to(device)
    mask_t    = torch.BoolTensor(valid_mask).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q = agent.model(state_t, scalars_t)
        q_np = q.squeeze(0).cpu().numpy()
        q[~mask_t] = -1e9
        best_action = q.argmax(dim=-1).item()

    if np.random.random() < epsilon:
        action = int(np.random.choice(np.where(valid_mask)[0]))
    else:
        action = int(best_action)
        
    return action, q_np


if __name__ == "__main__":
    train()