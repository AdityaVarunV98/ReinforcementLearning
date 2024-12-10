import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from Q_Models import DQN


env = gym.make('MountainCar-v0')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim).to(device)

model.load_state_dict(torch.load("SavedModels/dqn_model.pth", weights_only = True))

model.eval()

position_min, position_max = -1.2, 0.6
velocity_min, velocity_max = -10, 10

position_values = np.linspace(position_min, position_max, 100)
velocity_values = np.linspace(velocity_min, velocity_max, 100)

pos_grid, vel_grid = np.meshgrid(position_values, velocity_values)
action_grid = np.zeros_like(pos_grid, dtype=int)

for i in range(pos_grid.shape[0]):
    for j in range(pos_grid.shape[1]):
        state = np.array([pos_grid[i, j], vel_grid[i, j]], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, 2]

        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

        action_grid[i, j] = action

plt.figure(figsize=(12, 8))
contour = plt.contourf(pos_grid, vel_grid, action_grid, cmap='viridis', levels=3)

# Custom colorbar with action labels --- for labelling the actions
cbar = plt.colorbar(contour, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Accelerate Left', 'Do Nothing', 'Accelerate Right'])
cbar.set_label("Action")

plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title("Action Choices of Trained Agent in MountainCar-v0")

plt.show()