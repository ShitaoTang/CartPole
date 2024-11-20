import matplotlib.pyplot as plt

def plot_rewards(rewards, title="Rewards per Episode", log_scale=False, draw_line=False):
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, color='blue', linewidth=1, label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    if log_scale:
        plt.yscale('log')
    
    if draw_line:
        plt.axhline(y=200, color='red', linestyle='--', linewidth=1.5, label='y = 200')
    
    plt.title(title)
    plt.legend()
    plt.show()