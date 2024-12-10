import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
rewards_rtg_an = np.load("Data/pg_rewards_LunarLander-v3_True_True.npy")
rewards_rtg_no_an = np.load("Data/pg_rewards_LunarLander-v3_True_False.npy")
rewards_no_rtg_an = np.load("Data/pg_rewards_LunarLander-v3_False_True.npy")
rewards_no_rtg_no_an = np.load("Data/pg_rewards_LunarLander-v3_False_False.npy")

# Define the epochs based on the length of the least number of epochs
epochs = range(100)

plt.figure(figsize=(10, 6))
plt.plot(epochs, rewards_rtg_an[:len(epochs)], label="Reward-to-Go: True, Advantage Norm: True")
plt.plot(epochs, rewards_rtg_no_an[:len(epochs)], label="Reward-to-Go: True, Advantage Norm: False")
plt.plot(epochs, rewards_no_rtg_an[:len(epochs)], label="Reward-to-Go: False, Advantage Norm: True")
plt.plot(epochs, rewards_no_rtg_no_an[:len(epochs)], label="Reward-to-Go: False, Advantage Norm: False")

plt.xlabel("Epochs")
plt.ylabel("Average Reward")
plt.title("Average Reward vs Epoch for Different Configurations")
plt.legend()
plt.grid(True)

plt.show()