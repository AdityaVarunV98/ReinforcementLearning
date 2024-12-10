import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from Q_Models import DQN
from modifiedReward import observation, reward_function

env = gym.make('MountainCar-v0', render_mode = "human")

device = "cpu"

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize model
policy_net = DQN(state_dim, action_dim).to(device)

# Load saved parameters (Specify the correct file location)
file_name = "SavedModels/dqn_model.pth"

policy_net.load_state_dict(torch.load(file_name))

# Set model to evaluation mode
policy_net.eval()

test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    state = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
    total_reward = 0
    done = False

    while not done:
        # Choosing the action with the highest Q value
        with torch.no_grad():
            action = policy_net(state).argmax().item()
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = observation(next_state, env)
        reward = reward_function(next_state)
        
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        total_reward += reward

    print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
