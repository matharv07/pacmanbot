import os
import time
import random
import numpy as np
import multiprocessing as mp
import pacman
from rl_env import PacmanMultiAgentEnv

def _get_valid_actions(env):
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    WALL = 1
    ROWS = len(env.grid)
    COLS = len(env.grid[0])
    valid = {}
    for gid, ghost in env.ghosts.items():
        if ghost.dead:
            valid[gid] = []
            continue            
        v = []
        reverse_idx = -1
        for i, (dr, dc) in enumerate(DIRS): 
            nr, nc = ghost.row + dr, ghost.col + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and env.grid[nr][nc] != WALL:
                v.append(i)
            if hasattr(ghost, 'last_dir') and ghost.last_dir:
                if dr == -ghost.last_dir[0] and dc == -ghost.last_dir[1]:
                    reverse_idx = i
        if len(v) > 1 and reverse_idx in v:
            v.remove(reverse_idx)            
        valid[gid] = v
    return valid
    
def worker(remote, parent_remote):
    parent_remote.close()
    seed = (os.getpid() + int(time.time() * 1000)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    pacman.AUTO_MODE = True     
    env = PacmanMultiAgentEnv(max_steps=500)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, env_rewards, agent_dones, env_done, info = env.step(data)
            action_executed = info.get('action_executed', {})
            final_rewards = env_rewards
            terminal_info = info.copy()
            if env_done:
                obs, info = env.reset()
                info['terminal_info'] = terminal_info
                info['action_executed'] = action_executed
            info['valid_actions'] = _get_valid_actions(env)    
            remote.send((obs, final_rewards, agent_dones, env_done, info))
            
        elif cmd == 'reset':
            obs, info = env.reset()
            info['valid_actions'] = _get_valid_actions(env)
            remote.send((obs, info))
            
        elif cmd == 'close':
            remote.close()
            break

class SubprocVecEnv:
    def __init__(self, num_envs=6):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [mp.Process(target=worker, args=(work_remote, remote)) for work_remote, remote in zip(self.work_remotes, self.remotes)]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
            
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return obs, infos

    def step(self, actions_list):
        for remote, action in zip(self.remotes, actions_list):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, agent_dones, env_dones, infos = zip(*results)
        return obs, rewards, agent_dones, env_dones, infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()