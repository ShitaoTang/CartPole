import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole.q_learning import Q_Learning
from cartpole.config import create_environment, CONFIG
from cartpole.utils import plot_rewards

if __name__ == "__main__":
    env, hbounds, lbounds = create_environment()
    Q = Q_Learning(env, CONFIG["alpha"], CONFIG["gamma"], CONFIG["epsilon"], CONFIG["nepisodes"], CONFIG["nbins"], lbounds, hbounds)

    # run nepisodes(15000) episodes to update Q-Matrix
    Q.train()

    # This would interfere performance of training, weird...
    # # store Q-matrix
    # with open('q_matrix.pkl', 'wb') as f:
    #     pickle.dump(Q.qmatrix, f)
    # print("Q-matrix saved to 'q_matrix.pkl'")

    # plot rewards while training
    plot_rewards(Q.reward_per_episode, log_scale=True)

    # test 10 episodes
    totalrewards = Q.run()

    # This would also interfere...
    plot_rewards(totalrewards, log_scale=True, draw_line=True)
    
    # print to console
    # print("Test rewards for 10 episodes:")
    # for i, reward in enumerate(totalrewards, start=1):
    #     print(f"Episode {i}: Reward = {reward}")