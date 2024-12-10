import gym
import os
import gc
import torch
import pygame
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Modified reward function referenced from: https://github.com/MehdiShahbazi/DQN-Mountain-Car-Gymnasium/blob/master/DQN.py

def observation(state, env):
    # Min-Max Normalizing the state to pass into the reward_function
    
    min_value = env.observation_space.low
    max_value = env.observation_space.high
    
    normalized_state = (state - min_value) / (max_value - min_value)
    
    return normalized_state

def reward_function(state):
    # Modifies the reward based on the state
    
    current_position, current_velocity = state
    
    # Interpolate the value to the desired range (because the velocity normalized value would be in range of 0 to 1 and now it would be in range of -0.5 to 0.5)
    current_velocity = np.interp(current_velocity, np.array([0, 1]), np.array([-0.5, 0.5]))
    
    # (1) Calculate the modified reward based on the current position and velocity of the car.
    degree = current_position * 360
    degree2radian = np.deg2rad(degree)
    modified_reward =  0.2 * (np.cos(degree2radian) + 2 * np.abs(current_velocity))
    
    # (2) Step limitation
    modified_reward -= 0.5 # Subtract 0.5 to adjust the base reward (to limit useless steps).
    
    # (3) Check if the car has surpassed a threshold of the path and is closer to the goal
    if current_position > 0.98:
        modified_reward += 20  # Add a bonus reward (Reached the goal)
    elif current_position > 0.92: 
        modified_reward += 10 # So close to the goal
    elif current_position > 0.82:
        modified_reward += 6 # car is closer to the goal
    elif current_position > 0.65:
        modified_reward += 1 - np.exp(-2 * current_position) # car is getting close. Thus, giving reward based on the position and the further it reached
        
    
    # (4) Check if the car is coming down with velocity from left and goes with full velocity to right
    initial_position = 0.40842572 # Normalized value of initial position of the car which is extracted manually
    
    if current_velocity > 0.3 and current_position > initial_position + 0.1:
        modified_reward += 1 + 2 * current_position  # Add a bonus reward for this desired behavior
    
    modified_reward = np.float32(modified_reward)
    return modified_reward    
