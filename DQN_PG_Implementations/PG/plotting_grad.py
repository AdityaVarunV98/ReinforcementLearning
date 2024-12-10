import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files containing gradient magnitudes
gradients_batch_8 = np.load("Data/grad_mag_LunarLander-v3_True_True_8.npy")
gradients_batch_16 = np.load("Data/grad_mag_LunarLander-v3_True_True_16.npy")
gradients_batch_50 = np.load("Data/grad_mag_LunarLander-v3_True_True_50.npy")

# Define the epochs based on the length of the gradient magnitude lists
epochs = range(len(gradients_batch_8))

plt.figure(figsize=(10, 6))

# Plot gradient magnitudes, min/max lines for the different batch sizes
line_8, = plt.plot(epochs, gradients_batch_8, label="Batch Size: 8")
plt.axhline(y=np.min(gradients_batch_8), color=line_8.get_color(), linestyle='--', linewidth=1.5, label="Min Batch Size 8")
plt.axhline(y=np.max(gradients_batch_8), color=line_8.get_color(), linestyle=':', linewidth=1.5, label="Max Batch Size 8")

line_16, = plt.plot(epochs, gradients_batch_16, label="Batch Size: 16")
plt.axhline(y=np.min(gradients_batch_16), color=line_16.get_color(), linestyle='--', linewidth=1.5, label="Min Batch Size 16")
plt.axhline(y=np.max(gradients_batch_16), color=line_16.get_color(), linestyle=':', linewidth=1.5, label="Max Batch Size 16")

line_50, = plt.plot(epochs, gradients_batch_50, label="Batch Size: 50")
plt.axhline(y=np.min(gradients_batch_50), color=line_50.get_color(), linestyle='--', linewidth=1.5, label="Min Batch Size 50")
plt.axhline(y=np.max(gradients_batch_50), color=line_50.get_color(), linestyle=':', linewidth=1.5, label="Max Batch Size 50")

plt.xlabel("Epochs")
plt.ylabel("Gradient Magnitude")
plt.title("Gradient Magnitude vs Epoch for Different Batch Sizes with Min/Max Lines")
plt.legend()
plt.grid(True)

plt.show()