import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["NUM_ENVS"] = "2"
os.environ["ROLLOUT_STEPS"] = "16"
os.environ["MAX_UPDATES"] = "2"

if __name__ == "__main__":
    import train
    print("Launching test train loop...")
    train.train()
    print("Test train run completed successfully!")
