import pygame
from worker import Env

env = Env(env_id=0, num_ghosts=4)
env.reset()

print("FPS is smooth.")
