import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole.q_learning import Q_Learning
from cartpole.config import create_environment, CONFIG
from cartpole.utils import plot_rewards

if __name__ == "__main__":
    env, hbounds, lbounds = create_environment()
    Q = Q_Learning(env, CONFIG["alpha"], CONFIG["gamma"], CONFIG["epsilon"], CONFIG["nepisodes"], CONFIG["nbins"], lbounds, hbounds)
    Q.train()
    plot_rewards(Q.reward_per_episode, log_scale=True)