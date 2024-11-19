import gymnasium as gym

def create_environment():
    env = gym.make('CartPole-v1')
    hbounds = env.observation_space.high
    lbounds = env.observation_space.low
    hbounds[1], hbounds[3] = 3, 10
    lbounds[1], lbounds[3] = -3, -10
    return env, hbounds, lbounds

# 配置参数
CONFIG = {
    "alpha": 0.1,
    "gamma": 1,
    "epsilon": 0.2,
    "nepisodes": 15000,
    "nbins": [30, 30, 30, 30]
}