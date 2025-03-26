import gym
import numpy as np
from stable_baselines3 import PPO  # Change this if you used a different algorithm
from custom_env import QuadrupedEnv  # Import your custom environment

# Load the trained model
model_path = "/home/vinoth/rl_project/quadruped_rl/models/spot.zip"  # Update with your actual model path
model = PPO.load(model_path)

# Create the environment
env = QuadrupedEnv("quadruped_rl/boston_dynamics_spot/scene.xml")  # Update with your model XML
obs = env.reset()

# Run the trained model in the environment
num_episodes = 75
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Predict action
        obs, reward, done, info = env.step(action)  # Apply action
        episode_reward += reward
        env.render()  # Render simulation

    print(f"Episode {episode + 1}: Reward = {episode_reward}")

env.close()
