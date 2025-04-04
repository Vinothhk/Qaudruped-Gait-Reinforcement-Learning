import time
# from spot_env import QuadrupedEnv
from spot_train import QuadrupedEnv
from stable_baselines3 import PPO

def interactive_test(model_path, test_time=60):
    """Test model with real-time rendering and keyboard control"""
    env = QuadrupedEnv('quadruped_rl/boston_dynamics_spot/scene.xml')
    model = PPO.load(model_path)
    
    obs = env.reset()
    start_time = time.time()
    
    try:
        while time.time() - start_time < test_time:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            # Render at realistic speed
            env.render()
            time.sleep(0.02)  # ~50 FPS
            
            if done:
                obs = env.reset()
                
    finally:
        env.close()

# Usage - will open interactive MuJoCo window
interactive_test("/home/vinoth/rl_project/models/spot_ppo_20250326_233925/best_model.zip", test_time=120)  # 2-minute test
# interactive_test("/home/vinoth/rl_project/models/spot_ppo_20250328_010234/best_model.zip", test_time=120)  # 2-minute test