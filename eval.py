"""Headless eval: python3 eval.py <policy_name_without_zip> [ppo|sac]"""
import sys
import numpy as np
from stable_baselines3 import PPO, SAC
from balance_env import BalanceBotEnv

name = sys.argv[1] if len(sys.argv) > 1 else "mujoco_balance_bot"
algo = (sys.argv[2] if len(sys.argv) > 2 else "ppo").lower()
model = (SAC if algo == "sac" else PPO).load(name, device="cpu")
env = BalanceBotEnv()

ups = []
for s in range(8):
    obs, _ = env.reset(seed=s)
    n = up = run = best = 0
    first = None
    d = t = False
    while not (d or t):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, d, t, _ = env.step(np.array(a).flatten())
        n += 1
        err = np.arctan2(np.sin(env.data.qpos[1]), np.cos(env.data.qpos[1]))
        if abs(err) < 0.2:
            up += 1
            run += 1
            best = max(best, run)
            if first is None:
                first = n
        else:
            run = 0
    ups.append(up)
    print(f"seed={s} len={n} steps_upright={up} longest_hold={best} first={first} "
          f"final_x={env.data.qpos[0]:+.2f} ended={'cart-limit' if d else 'time'}")
print("mean steps_upright", np.mean(ups))
