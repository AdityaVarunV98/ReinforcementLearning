import numpy as np
import matplotlib.pyplot as plt

epsilon_values = [0.999, 0.9, 0.8]
file_format_avg = "Data/dqn_avg_rewards_epsilon_{}.npy"
file_format_best = "Data/dqn_best_rewards_epsilon_{}.npy"

plt.figure(figsize=(10, 6))
for epsilon in epsilon_values:
    mean_rewards = np.load(file_format_avg.format(epsilon), allow_pickle=True)
    best_rewards = np.load(file_format_best.format(epsilon), allow_pickle=True)
    
    plt.plot(mean_rewards, label=f'{10}-Episode Mean Reward')
    plt.plot(best_rewards, label=f'{10}-Episode Best Reward', linestyle='--')

plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("DQN Rewards for Mountain Car")
plt.legend()
plt.grid()
plt.show()