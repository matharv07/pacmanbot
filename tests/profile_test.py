import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cProfile
import pstats
import random
import numpy as np
import torch
from worker import Env

if __name__ == '__main__':
    env = Env(env_id=0, num_ghosts=4, world_height=33.0, world_width=41.0, n_power=28)
    env.reset()
    rows, cols = int(env.world.height), int(env.world.width)
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(10):
        action_dict = {}
        for gid in range(4):
            r, c = random.randint(1, 31), random.randint(1, 39)
            scores = np.zeros((rows, cols), dtype=np.float32)
            scores[r, c] = 1.0
            action_dict[gid] = ([(r, c)], scores, 1.0)
        env.step(action_dict, bc_prob=0.0)
    profiler.disable()
    with open('profile.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats(40)
    print("✓ Profile test completed successfully!")