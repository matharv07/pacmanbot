import cProfile
import pstats
from worker import Env
import random

env = Env(env_id=0, num_ghosts=4, grid_rows=33, grid_cols=41, n_power=28)
env.reset()

profiler = cProfile.Profile()
profiler.enable()

for i in range(100):
    action_dict = {}
    for gid in range(4):
        r, c = random.randint(1, 31), random.randint(1, 39)
        action_dict[gid] = ([(r, c)], { (r,c): 1.0 })
    env.step(action_dict, bc_prob=0.0)

profiler.disable()
with open('profile.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
    stats.print_stats(40)