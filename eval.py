"""Headless eval of a saved policy.

    python3 eval.py [policy_name_without_zip] [ppo|sac] [n_seeds]

Per seed it counts steps within 0.2 rad of upright, the longest continuous hold, and whether the
pole is still balanced at the end of the episode (last 50 steps all within 0.2 rad).
With 50+ seeds it also prints the balanced-at-end rate by sampled hinge damping, because
reset() randomizes damping per episode and that is what decides success.
"""
import sys
import numpy as np
import mujoco
from stable_baselines3 import PPO, SAC
from balance_env import BalanceBotEnv

name = sys.argv[1] if len(sys.argv) > 1 else "mujoco_sac"
algo = (sys.argv[2] if len(sys.argv) > 2 else "sac").lower()
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8

model = (SAC if algo == "sac" else PPO).load(name, device="cpu")
env = BalanceBotEnv()
hinge = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

ups, damps, balanced = [], [], []
for s in range(n_seeds):
    obs, _ = env.reset(seed=s)
    damping = env.model.dof_damping[hinge]  # sampled by reset()
    n = up = run = best = 0
    first = None
    errs = []
    d = t = False
    while not (d or t):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, d, t, _ = env.step(np.array(a).flatten())
        n += 1
        err = abs(np.arctan2(np.sin(env.data.qpos[1]), np.cos(env.data.qpos[1])))
        errs.append(err)
        if err < 0.2:
            up += 1
            run += 1
            best = max(best, run)
            if first is None:
                first = n
        else:
            run = 0
    ok = all(e < 0.2 for e in errs[-50:])
    ups.append(up)
    damps.append(damping)
    balanced.append(ok)
    if n_seeds <= 20:
        print(f"seed={s} len={n} steps_upright={up} longest_hold={best} first={first} "
              f"balanced_at_end={ok} damping={damping:.3f} final_x={env.data.qpos[0]:+.2f} "
              f"ended={'cart-limit' if d else 'time'}")

balanced, damps = np.array(balanced), np.array(damps)
print(f"mean steps_upright {np.mean(ups):.1f}/500")
print(f"balanced at end: {balanced.sum()}/{n_seeds} = {balanced.mean():.2f}")
if n_seeds >= 50:
    print("balanced at end by hinge damping:")
    for lo, hi in [(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.051)]:
        m = (damps >= lo) & (damps < hi)
        if m.any():
            print(f"  damping {lo:.2f}-{hi:.2f}: {balanced[m].sum():3d}/{m.sum():3d} = {balanced[m].mean():.2f}")
