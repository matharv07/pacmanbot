import os
import time
import random
import numpy as np
import multiprocessing as mp
import pacman
from rl_env import PacmanMultiAgentEnv

def _get_valid_actions(env):
    va = {}
    for gid, ghost in env.ghosts.items():
        if ghost.dead:
            continue
        opts = []
        for i, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nr, nc = ghost.row + dr, ghost.col + dc
            if 0 <= nr < len(ghost.grid) and 0 <= nc < len(ghost.grid[0]) and ghost.grid[nr][nc] != 1:
                opts.append(i)
        va[gid] = opts if opts else [0]
    return va

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
            active_gids = list(env.ghosts.keys())
            prev_dead = {gid: env.ghosts[gid].dead for gid in active_gids}
            action_executed = {}
            for gid, ghost in env.ghosts.items():
                if not ghost.dead:
                    action_executed[gid] = ((ghost.move_counter + 1) >= ghost.move_every)
            obs, rewards, terminated, truncated, info = env.step(data)
            agent_dones = {}
            for gid in active_gids:
                just_died = env.ghosts[gid].dead and not prev_dead.get(gid, False)
                agent_dones[gid] = terminated or truncated or just_died
            terminal_info = info.copy()
            env_done = terminated or truncated
            if env_done:
                obs, info = env.reset()
            info['valid_actions'] = _get_valid_actions(env)
            info['action_executed'] = action_executed
            if env_done:
                info['terminal_info'] = terminal_info
            remote.send((obs, rewards, agent_dones, env_done, info))
            
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