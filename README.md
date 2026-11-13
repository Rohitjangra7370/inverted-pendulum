# Inverted Pendulum (MuJoCo + PPO)

Cart-pole swing-up learned with Stable-Baselines3 PPO in a custom Gymnasium MuJoCo env.
The pole starts hanging down and must be swung up and balanced by pushing the cart.

## Files
- `balance_env.py` - `BalanceBotEnv`: MuJoCo model (slider cart + hinge pole), 5-value obs, shaped reward
- `train.py` - PPO training (3M steps, CPU), saves `mujoco_balance_bot.zip`
- `watch.py` - load a policy and view it in the MuJoCo viewer
- `hello_mujoco.py` - first MuJoCo experiment (sliding box)
- `mujoco_balance_bot.zip` - 3M-step policy; `mujoco_balance_bot_1p5M_backup.zip` - earlier 1.5M-step policy

## Environment
- Obs: `cart_pos, sin(angle), cos(angle), cart_vel, pole_vel` (angle 0 = upright, pi = hanging)
- Action: cart force in [-1, 1], scaled x10 (`ctrlrange="-10 10"`)
- 25 Hz control (4 physics steps per action), 500-step episodes, cart limit +-3 m
- Reward: `0.3*broad + narrow - 0.01*a^2 - 0.02*x^2`
  - `broad = (cos(angle)+1)/2`: weak slope everywhere
  - `narrow = exp(-(err/0.1)^2) * exp(-(pole_vel/4)^2)`: spike only when near upright AND slow

## Usage
```bash
python3 -m pip install mujoco gymnasium stable-baselines3
python3 train.py          # tensorboard --logdir ./ppo_tensorboard/
python3 watch.py
```

## Status
Not solved. Headless eval over 8 seeds (deterministic): mean 81/500 steps upright, longest continuous hold 26-74 steps
(about 3 s). 4 of 8 seeds never get the pole up (same seeds in both saved policies, so likely an exploration issue).

## Debugging history
1. Original pole cylinder was centred on its hinge: it started 15 cm inside the floor (huge contact forces) and its
   centre of mass sat on the hinge, so there was no gravity tipping. Fixed with a capsule from the hinge upward, collisions off.
2. Switching to swing-up left the old `abs(angle) > 0.4` termination, ending every episode at step 1. Removed.
3. `ctrlrange` clipped force to 1 N while code scaled by 10. Fixed.
4. Negative reward made driving off the track cheaper than staying alive. Reward shifted to non-negative.
5. Jiggling local optimum: added a narrow upright bonus, gated on low pole velocity so fly-throughs score little.

## Next
Try SAC (better exploration for swing-up), or PPO with `ent_coef=0.01`, `gamma=0.995`, `VecNormalize`.
