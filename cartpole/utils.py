import matplotlib.pyplot as plt

def plot_rewards(rewards, title="Rewards per Episode", log_scale=False):
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, color='blue', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if log_scale:
        plt.yscale('log')
    plt.title(title)
    plt.show()