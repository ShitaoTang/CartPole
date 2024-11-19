import numpy as np

def random_policy(env, nepisodes=100):
    total_rewards = []
    for episode in range(nepisodes):
        reward_per_episode = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            reward_per_episode += reward
        total_rewards.append(reward_per_episode)
    return np.mean(total_rewards)