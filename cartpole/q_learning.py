import gymnasium as gym
import numpy as np
import time
from tqdm import tqdm

class Q_Learning:
    def __init__(self, env, alpha, gamma, epsilon, nepisodes, nbins, lbounds, hbounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nepisodes = nepisodes
        self.nactions = env.action_space.n
        self.nbins = nbins
        self.lbounds = lbounds
        self.hbounds = hbounds
        self.reward_per_episode = []
        # qmatrix store accumulated rewards for each state-action pair
        self.qmatrix = np.random.uniform(low=0, high=1, size=(nbins[0], nbins[1], nbins[2], nbins[3], self.nactions))

    def idx_state(self, state):
        pos = state[0]
        vel = state[1]
        ang = state[2]
        ang_vel = state[3]
        
        pos_bins = np.linspace(self.lbounds[0], self.hbounds[0], self.nbins[0])
        vel_bins = np.linspace(self.lbounds[1], self.hbounds[1], self.nbins[1])
        ang_bins = np.linspace(self.lbounds[2], self.hbounds[2], self.nbins[2])
        ang_vel_bins = np.linspace(self.lbounds[3], self.hbounds[3], self.nbins[3])

        pos_idx = max(np.digitize(state[0], pos_bins) - 1, 0)
        vel_idx = max(np.digitize(state[1], vel_bins) - 1, 0)
        ang_idx = max(np.digitize(state[2], ang_bins) - 1, 0)
        ang_vel_idx = max(np.digitize(state[3], ang_vel_bins) - 1, 0)

        return tuple([pos_idx, vel_idx, ang_idx, ang_vel_idx])

    def select(self, state, nepisode):      
        if nepisode < 500:
            return self.env.action_space.sample()
        if nepisode > 7000:
            self.epsilon *= 0.999
        
        _ = np.random.random()
        if _ < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.random.choice(np.where(self.qmatrix[self.idx_state(state)] == np.max(self.qmatrix[self.idx_state(state)]))[0])
    
    def train(self):
        for nepisode in tqdm(range(self.nepisodes), desc="Training Progress"):
            rewards = []
            state, _ = self.env.reset()
            state = list(state)

            # print(f"\033[31m[TRAINING]\033[0m episode {nepisode}")
            done = False
            while not done:
                idx_state = self.idx_state(state)
                action = self.select(state, nepisode)

                next_state, reward, done, _, _ = self.env.step(action)
                next_state = list(next_state)
                idx_next_state = self.idx_state(next_state)
                rewards.append(reward)

                maxQprime = np.max(self.qmatrix[idx_next_state])

                if not done:
                    self.qmatrix[idx_state+(action,)] += self.alpha * (reward + self.gamma * maxQprime - self.qmatrix[idx_state+(action,)])
                else:
                    self.qmatrix[idx_state+(action,)] += self.alpha * (reward - self.qmatrix[idx_state+(action,)])
                
                state = next_state
            
            # print(f"Sum of rewards: {np.sum(rewards)}")
            self.reward_per_episode.append(np.sum(rewards))

    def run(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        state, _ = env1.reset()
        env1.render()
        total_rewards = []
        nepisodes = 10
        max_steps = 1000

        for episode in range(nepisodes):
            done = False
            reward_per_episode = 0
            state, _ = env1.reset()
            
            for step in range(max_steps):
                action = np.random.choice(np.where(self.qmatrix[self.idx_state(state)] == np.max(self.qmatrix[self.idx_state(state)]))[0])
                state, reward, done, truncated, info = env1.step(action)
                reward_per_episode += reward
                time.sleep(0.05)
                if step == max_steps or done:
                    total_rewards.append(reward_per_episode)
                if done:
                    time.sleep(1)
                    break
        return total_rewards