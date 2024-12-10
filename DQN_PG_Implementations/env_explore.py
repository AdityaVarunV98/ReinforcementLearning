import matplotlib.pyplot as plt
import gym

def explore_env(env_name):
    env = gym.make(env_name)
    print(f"Exploring {env_name}:")
    
    state = env.reset()
    done = False
    total_reward = 0

    print("Random agent actions and rewards:")
    # while not done:
    #     action = env.action_space.sample()  # Random action
    #     next_state, reward, done, truncated, info  = env.step(action)
    #     total_reward += reward
    #     print(f"Action: {action}, Reward: {reward}, New State: [Image Frame Omitted]")

    print("Total Reward:", total_reward)
    print(f"State space (observation space): {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("Sample state:", env.observation_space.sample())
    print("Sample action:", env.action_space.sample())
    env.close()

if __name__ == "__main__":
    # Run explorations
    explore_env('MountainCar-v0')
    explore_env("Pong-v0")
    explore_env('CartPole-v0')
    explore_env('LunarLander-v2')