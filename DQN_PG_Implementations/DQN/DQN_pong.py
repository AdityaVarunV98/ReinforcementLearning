# import gym
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import cv2
# import matplotlib.pyplot as plt
# from Q_Models import DQN_pong

# # Frame preprocessing 
# def preprocess_frame(frame, prev_frame=None):
#     # Convert to numpy if required
#     if not isinstance(frame, np.ndarray):
#         frame = np.array(frame)

#     # Convert to grayscale
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = cv2.resize(frame, (84, 84))
#     frame = frame.astype(np.float32) / 255.0

#     if prev_frame is None:
#         return frame
#     else:
#         frame_diff = frame - prev_frame
#         return frame_diff

# if __name__ == "__main__":
#     # CUDA device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(device)

#     # Hyperparameters
#     gamma = 0.99
#     epsilon = 0.999
#     epsilon_min = 0.01
#     epsilon_decay = 0.997
#     learning_rate = 0.001
#     memory_size = 10000
#     batch_size = 32
#     target_update = 10
#     num_episodes = 2000
#     frame_stack = 4

#     # Replay memory as a deque
#     replay_memory = deque(maxlen=memory_size)

#     # Initialize environment
#     env = gym.make('Pong-v4')

#     # State and action space
#     state_dim = (frame_stack, 84, 84)
#     action_dim = env.action_space.n

#     # Action-value and target networks
#     policy_net = DQN_pong(state_dim, action_dim).to(device)
#     target_net = DQN_pong(state_dim, action_dim).to(device)
#     target_net.load_state_dict(policy_net.state_dict())
#     target_net.eval()

#     optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()

#     # Initialize lists to store rewards for plotting
#     episode_rewards = []
#     mean_rewards = []
#     best_mean_reward = -float('inf')
#     best_rewards = []
#     n = 10

#     for episode in range(num_episodes):
#         # Reset environment and preprocess the initial state
#         state = env.reset()[0]
#         prev_frame = None
#         frames = deque(maxlen=frame_stack)

#         # Stack initial frames
#         for _ in range(frame_stack):
#             frame = preprocess_frame(state, prev_frame)
#             frames.append(frame)
#             prev_frame = frame

#         stacked_state = np.stack(frames, axis=0)
#         stacked_state = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)

#         total_reward = 0
#         done = False

#         while not done:
#             # Epsilon-greedy action selection
#             if random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     action = policy_net(stacked_state).argmax().item()

#             # Take action and observe next state and reward
#             next_frame, reward, done, truncated, info = env.step(action)
#             next_frame = preprocess_frame(next_frame, prev_frame)
#             frames.append(next_frame)
#             prev_frame = next_frame

#             # Stack frames again for next state
#             next_stacked_state = np.stack(frames, axis=0)
#             next_stacked_state = torch.FloatTensor(next_stacked_state).unsqueeze(0).to(device)
#             total_reward += reward

#             # Store transition in replay memory
#             replay_memory.append((stacked_state, action, reward, next_stacked_state, done))

#             # Update state
#             stacked_state = next_stacked_state

#             # Update network using replay memory
#             if len(replay_memory) > batch_size:
#                 batch = random.sample(replay_memory, batch_size)
#                 states, actions, rewards, next_states, dones = zip(*batch)

#                 states = torch.cat(states).to(device)
#                 next_states = torch.cat(next_states).to(device)
#                 actions = torch.tensor(actions).unsqueeze(1).to(device)
#                 rewards = torch.tensor(rewards).to(device)
#                 dones = torch.tensor(dones).to(device)

#                 # Q-values for current states
#                 q_values = policy_net(states).gather(1, actions)

#                 # Compute target Q-values, using Bellman Equation
#                 with torch.no_grad():
#                     next_q_values = target_net(next_states).max(1)[0]
#                     targets = rewards + (1 - dones.float()) * gamma * next_q_values

#                 # Compute loss and update network
#                 loss = loss_fn(q_values, targets.unsqueeze(1))
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # Update target network every C steps
#             if episode % target_update == 0:
#                 target_net.load_state_dict(policy_net.state_dict())

#             done = done or truncated

#         # Store reward for this episode
#         episode_rewards.append(total_reward)

#         # Calculate mean reward for last n episodes
#         if len(episode_rewards) >= n:
#             mean_reward = np.mean(episode_rewards[-n:])
#             mean_rewards.append(mean_reward)
#             if mean_reward > best_mean_reward:
#                 best_mean_reward = mean_reward
#             best_rewards.append(best_mean_reward)
#         else:
#             mean_rewards.append(None)
#             best_rewards.append(best_mean_reward)

#         # Decay epsilon
#         if epsilon > epsilon_min:
#             epsilon *= epsilon_decay

#         print(f"Episode {episode+1}, Total Reward: {total_reward}, Mean Reward: {mean_reward if len(episode_rewards) >= n else 'N/A'}")

#     env.close()

#     torch.save(policy_net.state_dict(), "pong_dqn_model.pth")

#     plt.figure(figsize=(10, 5))
#     plt.plot(mean_rewards, label=f'{n}-Episode Mean Reward')
#     plt.plot(best_rewards, label='Best Mean Reward', linestyle='--')
#     plt.xlabel('Episodes')
#     plt.ylabel('Mean Reward')
#     plt.title('Learning Curve of DQN on Pong-v0')
#     plt.legend()
#     plt.show()

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import matplotlib.pyplot as plt
from Q_Models import DQN_pong

# Convert to gray, frame difference
def preprocess_frame(frame, prev_frame=None):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    if prev_frame is None:
        return frame
    else:
        return frame - prev_frame

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters
    gamma = 0.99
    epsilon = 0.999
    epsilon_min = 0.01
    epsilon_decay = 0.997
    learning_rate = 0.001
    memory_size = 10000
    batch_size = 32
    target_update = 10
    num_episodes = 1000
    frame_stack = 4

    # Defining the memory buffer
    replay_memory = deque(maxlen=memory_size)

    env = gym.make('Pong-v4')
    state_dim = (frame_stack, 84, 84)
    action_dim = env.action_space.n

    # Defining the neural networks
    policy_net = DQN_pong(state_dim, action_dim).to(device)
    target_net = DQN_pong(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Only updating the policy net through backprop
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    episode_rewards = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    best_rewards = []
    n = 10

    for episode in range(num_episodes):
        state = env.reset()[0]
        
        prev_frame = None
        frames = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            frame = preprocess_frame(state, prev_frame)
            frames.append(frame)
            prev_frame = frame

        stacked_state = np.stack(frames, axis=0)
        stacked_state = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)

        total_reward = 0
        done = False

        while not done:
            # Epsilon Greedy    
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(stacked_state).argmax().item()

            # Take action
            next_frame, reward, done, truncated, info = env.step(action)
            next_frame = preprocess_frame(next_frame, prev_frame)
            frames.append(next_frame)
            prev_frame = next_frame

            next_stacked_state = np.stack(frames, axis=0)
            next_stacked_state = torch.FloatTensor(next_stacked_state).unsqueeze(0).to(device)
            total_reward += reward

            # Add to buffer
            replay_memory.append((stacked_state, action, reward, next_stacked_state, done))
            stacked_state = next_stacked_state

            # Training loop during the each step
            if len(replay_memory) > batch_size:
                
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).to(device)
                dones = torch.tensor(dones).to(device)

                q_values = policy_net(states).gather(1, actions)

                # Calculating targets through bellman equation
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + (1 - dones.float()) * gamma * next_q_values

                loss = loss_fn(q_values, targets.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            done = done or truncated

        episode_rewards.append(total_reward)

        # Storing the mean and the best mean rewards
        if len(episode_rewards) >= n:
            mean_reward = np.mean(episode_rewards[-n:])
            mean_rewards.append(mean_reward)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
            best_rewards.append(best_mean_reward)
        else:
            mean_rewards.append(None)
            best_rewards.append(best_mean_reward)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Mean Reward: {mean_reward if len(episode_rewards) >= n else 'N/A'}")

    env.close()

    torch.save(policy_net.state_dict(), "pong_dqn_model.pth")

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label=f'{n}-Episode Mean Reward')
    plt.plot(best_rewards, label='Best Mean Reward', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Learning Curve of DQN on Pong-v0')
    plt.legend()
    plt.show()
