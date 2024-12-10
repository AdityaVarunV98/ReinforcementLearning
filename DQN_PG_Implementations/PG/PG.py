# import argparse
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical

# # architecture for Policy Gradient
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, action_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         return torch.softmax(self.fc2(x), dim=-1)

# # Function to compute policy gradient
# def policy_gradient(env_name, num_epochs, batch_size, reward_to_go, advantage_norm, lr, test_interval):
#     discount_factor = 0.99
    
#     # Initialize environment
#     env = gym.make(env_name)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(device)
    
#     # State and Action Space
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     policy_net = PolicyNetwork(state_dim, action_dim).to(device)
#     optimizer = optim.Adam(policy_net.parameters(), lr=lr)

#     # function to compute the baseline (mean reward)
#     def compute_baseline(returns):
#         return torch.mean(returns).item()

#     # function to calculate the advantage (reward - baseline)
#     def calculate_advantages(returns, baseline_value):
#         advantages = returns - baseline_value if advantage_norm else returns
#         return (advantages - advantages.mean()) / (advantages.std() + 1e-10) if advantage_norm else advantages

#     avg_rewards = []
#     gradient_magnitudes = []
    
#     # Training loop
#     for epoch in range(num_epochs):
        
#         total_reward = 0
        
#         loss = 0
#         # Generate episodes using the policy network
#         for episode in range(batch_size):
#             states, actions, rewards = [], [], []
            
            
#             state = env.reset()[0]
#             episode_rewards = []
#             done = False
            
#             c = 0
#             while not done:
#                 c+=1
                
#                 state_tensor = torch.FloatTensor(state).to(device)
#                 action_probs = policy_net(state_tensor)
#                 dist = Categorical(action_probs)
#                 action = dist.sample()
                
#                 next_state, reward, done, truncated, info = env.step(action.item())
#                 states.append(state)
#                 actions.append(action)
#                 rewards.append(reward)
                
#                 state = next_state
#                 total_reward += reward
#                 episode_rewards.append(reward)
                
#                 done = done or truncated

#             # Calculate cumulative reward for each step for each episode
#             returns = []
            
#             G = 0
#             for r in reversed(episode_rewards):
#                 G = r + G * discount_factor
#                 returns.insert(0, G)
            
#             if reward_to_go:
#                 returns = torch.tensor(returns).float().to(device)
#             else:
#                 returns = torch.full((len(episode_rewards),), G).float().to(device)
                
#             baseline_value = compute_baseline(returns) if advantage_norm else 0
#             advantages = calculate_advantages(returns, baseline_value)
            
#             states_tensor = torch.FloatTensor(np.array(states)).to(device)
#             actions_tensor = torch.tensor(actions).to(device)
            
#             log_probs = Categorical(policy_net(states_tensor)).log_prob(actions_tensor)
        
#             loss -= (log_probs * advantages).sum()
            
#         loss /= batch_size
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         gradient_magnitude = 0
#         for param in policy_net.parameters():
#             gradient_magnitude += param.grad.norm().item()
#         gradient_magnitudes.append(gradient_magnitude)

#         avg_reward = total_reward / batch_size
#         avg_rewards.append(avg_reward)
#         if epoch % test_interval == 0:
#             print(f"Epoch {epoch}/{num_epochs}, Average Reward: {avg_reward:.2f}")

#     env.close()
    
#     print("Saving the model as:", f"PG_model_{env_name}_{reward_to_go}_{advantage_norm}.pth")
#     torch.save(policy_net.state_dict(), f"PG_model_{env_name}_{reward_to_go}_{advantage_norm}.pth")
    
#     np.save(f"pg_rewards_{env_name}_{reward_to_go}_{advantage_norm}.npy", avg_rewards)
#     np.save(f"grad_mag_{env_name}_{reward_to_go}_{advantage_norm}_{batch_size}.npy", gradient_magnitudes)
    
#     return avg_rewards

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Policy Gradient Training")
#     parser.add_argument("--env_name", type=str, default="CartPole-v0", help="Environment name")
#     parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument("--reward_to_go", type=bool, default=False, help="Use reward-to-go (True/False)")
#     parser.add_argument("--advantage_norm", type=bool, default=False, help="Use advantage normalization (True/False)")
#     parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
#     parser.add_argument("--test_interval", type=int, default=100, help="Interval for displaying progress")
#     args = parser.parse_args()

#     print(args.reward_to_go)
#     print(args.advantage_norm)
    
    
#     # Run policy gradient training
#     rewards = policy_gradient(
#         env_name=args.env_name,
#         num_epochs=args.num_epochs,
#         batch_size=args.batch_size,
#         reward_to_go=args.reward_to_go,
#         advantage_norm=args.advantage_norm,
#         lr=args.lr,
#         test_interval=args.test_interval,
#     )


import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Defining the neural network for the policy: outputs the probabilities for the actions
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def policy_gradient(env_name, num_epochs, batch_size, reward_to_go, advantage_norm, lr, test_interval):
    discount_factor = 0.99
    
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initializing the neural networks
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Function to compute the baselines
    def compute_baseline(returns):
        return torch.mean(returns).item()

    # Function to compute the normalized advantages
    def calculate_advantages(returns, baseline_value):
        advantages = returns - baseline_value if advantage_norm else returns
        return (advantages - advantages.mean()) / (advantages.std() + 1e-10) if advantage_norm else advantages

    avg_rewards = []
    gradient_magnitudes = []
    
    for epoch in range(num_epochs):
        total_reward = 0
        loss = 0
        
        # Generate batch_size number of episodes
        for episode in range(batch_size):
            states, actions, rewards = [], [], []
            state = env.reset()[0]
            episode_rewards = []
            done = False
            
            c = 0
            
            # Storing the states, actions and rewards of the episode
            while not done:
                c += 1
                state_tensor = torch.FloatTensor(state).to(device)
                action_probs = policy_net(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                
                next_state, reward, done, truncated, info = env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                total_reward += reward
                episode_rewards.append(reward)
                
                done = done or truncated

            
            # Calculating the loss based on the specified conditions
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + G * discount_factor
                returns.insert(0, G)
            
            if reward_to_go:
                returns = torch.tensor(returns).float().to(device)
            else:
                returns = torch.full((len(episode_rewards),), G).float().to(device)
                
            baseline_value = compute_baseline(returns) if advantage_norm else 0
            advantages = calculate_advantages(returns, baseline_value)
            
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            actions_tensor = torch.tensor(actions).to(device)
            
            log_probs = Categorical(policy_net(states_tensor)).log_prob(actions_tensor)
            loss -= (log_probs * advantages).sum()
            
        # Backprop
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Storing the magnitude of the gradient
        gradient_magnitude = 0
        for param in policy_net.parameters():
            gradient_magnitude += param.grad.norm().item()
        gradient_magnitudes.append(gradient_magnitude)

        avg_reward = total_reward / batch_size
        avg_rewards.append(avg_reward)
        if epoch % test_interval == 0:
            print(f"Epoch {epoch}/{num_epochs}, Average Reward: {avg_reward:.2f}")

    env.close()
    
    # Saving the model, rewards and the grad_mag
    print("Saving the model as:", f"PG_model_{env_name}_{reward_to_go}_{advantage_norm}.pth")
    torch.save(policy_net.state_dict(), f"PG_model_{env_name}_{reward_to_go}_{advantage_norm}.pth")
    
    np.save(f"pg_rewards_{env_name}_{reward_to_go}_{advantage_norm}.npy", avg_rewards)
    np.save(f"grad_mag_{env_name}_{reward_to_go}_{advantage_norm}_{batch_size}.npy", gradient_magnitudes)
    
    return avg_rewards

if __name__ == "__main__":
    # Parser to allow for command line input
    
    parser = argparse.ArgumentParser(description="Policy Gradient Training")
    parser.add_argument("--env_name", type=str, default="CartPole-v0", help="Environment name")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--reward_to_go", type=bool, default=False, help="Use reward-to-go (True/False)")
    parser.add_argument("--advantage_norm", type=bool, default=False, help="Use advantage normalization (True/False)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--test_interval", type=int, default=100, help="Interval for displaying progress")
    args = parser.parse_args()
    
    
    rewards = policy_gradient(env_name=args.env_name, num_epochs=args.num_epochs, batch_size=args.batch_size, reward_to_go=args.reward_to_go, advantage_norm=args.advantage_norm, lr=args.lr, test_interval=args.test_interval)
