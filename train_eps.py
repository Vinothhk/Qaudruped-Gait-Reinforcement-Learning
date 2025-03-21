from huggingface_hub import EvalResult
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback,EvalCallback,CallbackList
from custom_env import QuadrupedEnv

# Custom callback for rendering and episode tracking
class RenderAndEpisodeCallback(BaseCallback):
    def __init__(self, env, total_episodes, render_freq=1):
        super().__init__()
        self.env = env
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.render_freq = render_freq

    def _on_step(self):
        if self.locals.get("dones"):  # Check if the episode ended
            self.episode_count += 1
            print(f"Episode {self.episode_count} completed.")
            if self.episode_count >= self.total_episodes:
                return False  # Stop training

        if self.n_calls % self.render_freq == 0:
            self.env.render()  # Render the environment
        return True

MODEL_DIR = "models"
LOG_DIR = "logs"
# Create a single environment (not vectorized)
env = QuadrupedEnv('/home/vinoth/rl_project/boston_dynamics_spot/scene.xml')

eval_callback = EvalCallback(
    env,
    best_model_save_path="models",
    log_path=LOG_DIR,
    eval_freq=10000, #timesteps
    n_eval_episodes=4,
    deterministic=True,
    render=False,
)

# Initialize the PPO agent
# model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

# Train the agent for a specific number of episodes
total_episodes = 1000  # Declare the number of episodes
average_episode_length = 500  # Adjust based on your environment
total_timesteps = total_episodes * average_episode_length * 2.5

# Train the agent with rendering and episode tracking
callback = RenderAndEpisodeCallback(env, total_episodes, render_freq=100)

cb_list = ([eval_callback,callback])
model.learn(total_timesteps=total_timesteps, progress_bar=True,callback=cb_list)

# Save the model
model.save("models/final_quadruped_ppo")