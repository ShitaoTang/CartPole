import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from cartpole.replay_buffer import ReplayBuffer
from cartpole.dqn import DQN
from cartpole.utils import plot_rewards
import torch

# Hyperparameters
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize environment, replay buffer, and agent
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

totalrewards = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            reward_per_episode = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                reward_per_episode += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

            totalrewards.append(reward_per_episode)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                    'reward': f'{np.mean(totalrewards[-10:]):.3f}'
                })
            pbar.update(1)

plot_rewards(totalrewards, title=f"DQN on {env_name}", log_scale=False, draw_line=True)