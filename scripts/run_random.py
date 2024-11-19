import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole.random_policy import random_policy
from cartpole.config import create_environment

if __name__ == "__main__":
    env, _, _ = create_environment()
    avg_reward = random_policy(env)
    print(f"Average reward of random policy: {avg_reward}")