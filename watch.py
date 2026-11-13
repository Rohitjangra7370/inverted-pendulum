import time
import mujoco.viewer
from stable_baselines3 import PPO
from balance_env import BalanceBotEnv
import numpy as np


print("Loading Physical AI environment...")
env = BalanceBotEnv()

print("Loading the trained brain...")
model = PPO.load("mujoco_balance_bot")

obs, _ = env.reset()

# Launch the MuJoCo 3D visualizer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    print("Visualizer online. Watch it balance.")
    
    while viewer.is_running():
        # 1. The AI looks at the state and decides the optimal torque
        action, _states = model.predict(obs, deterministic=False)
        action = np.array(action).flatten()
        
        # 2. Apply torque and step physics
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Render the frame
        viewer.sync()
        
        # 4. Slow down the loop so it runs in real-time, not hyperspeed
        time.sleep(0.01) 
        
        # If the bot falls or drives off the edge, reset the simulation
        if terminated or truncated:
            print("Resetting environment...")
            obs, _ = env.reset()
            time.sleep(0.5) # Brief pause before restarting