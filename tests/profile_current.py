import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='hide'; os.environ['SDL_VIDEODRIVER']='dummy'
import cProfile
import pstats
import time
import sys
import os

# Append the parent directory to sys.path so we can import pacman
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pacman import Game

def run_prof():
    g = Game()
    start_time = time.time()
    frames = 0
    while g.state != 'quit' and time.time() - start_time < 3.0:
        g.handle_events()
        g.update()
        frames += 1
    print(f'Processed {frames} frames in {time.time() - start_time:.2f} seconds', file=sys.stderr)

cProfile.run('run_prof()', '/home/atharv/xTerra/pacmanbot/tests/profile_stats.prof')
p = pstats.Stats('/home/atharv/xTerra/pacmanbot/tests/profile_stats.prof')
with open('/home/atharv/xTerra/pacmanbot/tests/profile_current.txt', 'w') as f:
    p.strip_dirs().sort_stats('cumtime').print_stats(30, file=f)
