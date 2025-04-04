# # Best Working for spot_env import QuadrupedEnv
# import os
# import torch
# from multiprocessing import freeze_support
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.callbacks import EvalCallback, CallbackList
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed
# from spot_env import QuadrupedEnv  # Your custom environment

# def make_env(env_id, rank=0, seed=0):
#     """
#     Utility function for multiprocessed env.
#     :param env_id: (str) the environment ID
#     :param rank: (int) index of the subprocess
#     :param seed: (int) the initial seed for RNG
#     """
#     def _init():
#         env = QuadrupedEnv('quadruped_rl/boston_dynamics_spot/scene.xml')
#         env.seed(seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

# def main():
#     # Configuration
#     MODEL_DIR = "models"
#     LOG_DIR = "logs"
#     NUM_ENVS = 4  # Number of parallel environments
#     TOTAL_TIMESTEPS = 2_000_000
#     EVAL_FREQ = 10000  # Evaluate every 10k timesteps
#     SEED = 42

#     # Create vectorized environments
#     if NUM_ENVS > 1:
#         # When using SubprocVecEnv, the environment must be pickleable
#         env = SubprocVecEnv([make_env(i, rank=i, seed=SEED) for i in range(NUM_ENVS)])
#     else:
#         env = DummyVecEnv([make_env(0, seed=SEED)])

#     # Create evaluation environment
#     eval_env = DummyVecEnv([make_env(0, seed=SEED)])

#     # Callbacks
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=MODEL_DIR,
#         log_path=LOG_DIR,
#         eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
#         n_eval_episodes=5,
#         deterministic=True,
#         render=False,
#         verbose=1
#     )

#     # Set up PPO parameters
#     policy_kwargs = dict(
#         activation_fn=torch.nn.ReLU,
#         net_arch=dict(pi=[256, 256], vf=[256, 256])
#     )

#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         policy_kwargs=policy_kwargs,
#         learning_rate=3e-4,
#         n_steps=2048 // NUM_ENVS,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         clip_range_vf=0.2,
#         ent_coef=0.01,
#         tensorboard_log=LOG_DIR,
#         seed=SEED
#     )

#     # Training
#     try:
#         model.learn(
#             total_timesteps=TOTAL_TIMESTEPS,
#             callback=eval_callback,
#             progress_bar=True,
#             tb_log_name="ppo_spot"
#         )
#     finally:
#         # Cleanup
#         model.save(os.path.join(MODEL_DIR, "spot_ppo_final"))
#         env.close()
#         eval_env.close()

# if __name__ == '__main__':
#     freeze_support()  # Required for Windows support
#     main()

import os
import torch
from datetime import datetime
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
# from spot_env import QuadrupedEnv
from env_v2 import QuadrupedEnv
def make_env(rank=0, seed=0):
    """Environment creation function for parallel envs"""
    def _init():
        env = QuadrupedEnv('quadruped_rl/boston_dynamics_spot/scene.xml')
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # Configuration
    NUM_ENVS = 12
    TOTAL_TIMESTEPS = 3_000_000
    EVAL_FREQ = 10000
    SEED = 42
    
    # Create timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/spot_ppo_{timestamp}"
    log_dir = f"logs/spot_ppo_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environments
    env = make_vec_env(
        lambda: QuadrupedEnv('quadruped_rl/boston_dynamics_spot/scene.xml'),
        n_envs=NUM_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=log_dir,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    # Create evaluation environment (same type as training env)
    eval_env = make_vec_env(
        lambda: QuadrupedEnv('quadruped_rl/boston_dynamics_spot/scene.xml'),
        n_envs=1,
        seed=SEED,
        monitor_dir=log_dir
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Model configuration
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048 // NUM_ENVS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        max_grad_norm=0.5, #added
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        seed=SEED,
        device="auto"
    )

    # Training
    print(f"Starting training - models will be saved to {model_dir}")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
            tb_log_name="ppo_spot"
        )
    finally:
        # Save final model
        final_model_path = os.path.join(model_dir, f"spot_ppo_final_{timestamp}")
        model.save(final_model_path)
        print(f"Training complete. Final model saved to {final_model_path}")
        
        # Cleanup
        env.close()
        eval_env.close()

if __name__ == '__main__':
    freeze_support()
    main()