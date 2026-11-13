import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from balance_env import BalanceBotEnv

print("getting mujoco gaang ready!!!")

env = DummyVecEnv([lambda: BalanceBotEnv()])

print("getting shittt to ppo guy")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cpu")

print(" Training is getting ready gaaang hold your seats")
print("-> open a new terminal gaaang to watch the magic run :   tensorboard --logdir ./ppo_tensorboard/")

model.learn(total_timesteps=3000000)

print("saving policy")
model.save("mujoco_balance_bot")
print("done ggnnng policy complete and saved")