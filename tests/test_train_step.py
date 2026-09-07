import os
import sys

os.environ["NUM_ENVS"] = "2"
os.environ["ROLLOUT_STEPS"] = "16"
os.environ["MAX_UPDATES"] = "302"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train

if __name__ == "__main__":
    print("Imported train successfully!")
    vec_env = train.VecEnv(n=2, rows=7, cols=9, n_ghosts=3, n_power=2)
    obs = vec_env.reset()
    assert len(obs) == 2
    print("VecEnv test initialized successfully!")
    vec_env.close()
    print("VecEnv test closed cleanly!")
