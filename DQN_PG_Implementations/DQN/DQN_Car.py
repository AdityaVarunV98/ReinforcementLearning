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

if __name__ == "__main__":

    # CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters
    gamma = 0.99
    epsilon = 0.999
    epsilon_min = 0.01
    epsilon_decay = 0.997
    learning_rate = 0.001
    memory_size = 10000
    batch_size = 64
    target_update = 10
    num_episodes = 1000
    epsilon_max = epsilon
    
    # Replay memory as a deque. Automatically removes the older elements once the max_len is crossed
    replay_memory = deque(maxlen=memory_size)
    
    # Initialize environment
    env = gym.make('MountainCar-v0')
    
    # State and Action Spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Action-value and target networks
    # When using forward, need to pass both an extra dimension of batch_size as the first element
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Since we are only updating the parameters of the policy_net at each time step
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Initialize lists to store rewards for plotting
    episode_rewards = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    best_rewards = []
    n = 10
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()
                    
            # Take action and observe next state and reward
            next_state, reward, done, truncated, info = env.step(action)
            
            next_state = observation(next_state, env)
            reward = reward_function(next_state)
            
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            total_reward += reward
            
            # print(next_state)
            
            # Store transition in replay memory
            replay_memory.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            
            # If the number of elements in the replay_memory is greater than the batch size, we can random sample, then perform the required weight updates
            if len(replay_memory) > batch_size:
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).to(device)
                dones = torch.tensor(dones).to(device)
                
                # print(states.shape)
                
                # Q-values for current states
                q_values = policy_net(states).gather(1, actions)
                
                # Compute target Q-values, using the Bellman Equation
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + (1 - dones.float()) * gamma * next_q_values

                # Compute loss and update network
                loss = loss_fn(q_values, targets.unsqueeze(1))
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
            
            
            # Update target network every C steps
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            # Stopping the episode after reaching a final state or truncating the episode
            done = done or truncated
                
        episode_rewards.append(total_reward)
        
        # mean reward for the last n episodes
        if len(episode_rewards) >= n:
            mean_reward = np.mean(episode_rewards[-n:])
            mean_rewards.append(mean_reward)
            
            # Store best_mean_reward
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
            best_rewards.append(best_mean_reward)
        else:
            mean_rewards.append(None)
            best_rewards.append(best_mean_reward)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Mean Reward: {mean_reward if len(episode_rewards) >= n else 'N/A'}")
        
            
    env.close()

    torch.save(policy_net.state_dict(), f"dqn_model_epsilon_{epsilon_max}.pth")

    # Plotting learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label=f'{n}-Episode Mean Reward')
    plt.plot(best_rewards, label='Best Mean Reward', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title(f'Learning Curve of DQN on MountainCar-v0')
    plt.legend()
    plt.show()
    
    mean_rewards = np.array(mean_rewards)
    best_rewards = np.array(best_rewards)
    
    # Saving for future plots
    np.save(f"dqn_avg_rewards_epsilon_{epsilon_max}.npy", mean_rewards)
    np.save(f"dqn_best_rewards_epsilon_{epsilon_max}.npy", best_rewards)