import sys
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from balance_env import BalanceBotEnv

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000

env = Monitor(BalanceBotEnv())  # Monitor logs ep_rew_mean / ep_len_mean

# SAC defaults are tuned for this kind of task. ent_coef="auto" learns the exploration level
# (target entropy = -action_dim), so it keeps exploring until it has something better.
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./sac_tensorboard/",
    device="cpu",
    learning_starts=5_000,  # random actions first to fill the replay buffer
    train_freq=1,
    gradient_steps=1,
)

print(f"training SAC for {steps} steps")
model.learn(total_timesteps=steps)
model.save("mujoco_sac")
print("done SAC saved to mujoco_sac.zip")
