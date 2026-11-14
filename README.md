# Inverted Pendulum (MuJoCo + RL)

Cart-pole **swing-up**: the pole starts hanging down and must be swung up and balanced by pushing the cart.
Custom Gymnasium + MuJoCo environment, trained with Stable-Baselines3.

PPO could not do it. SAC can: it swings up and holds the pole until the end of the episode in 33% of
randomized episodes (100% when hinge damping is below 0.01). Failures come from high damping, see below.

## Files
| File | What |
|---|---|
| `balance_env.py` | `BalanceBotEnv`: MuJoCo model (slider cart + hinge pole), 5-value obs, shaped reward, domain randomization |
| `train.py` | PPO training (3M steps, CPU), saves `mujoco_balance_bot.zip` |
| `train_sac.py` | SAC training (default 300k steps, CPU), saves `mujoco_sac.zip` |
| `eval.py` | headless eval of a saved policy: `python3 eval.py mujoco_sac sac 300` |
| `watch.py` | view a policy in the MuJoCo viewer: `python3 watch.py mujoco_sac sac` |
| `hello_mujoco.py` | first MuJoCo experiment (sliding box) |
| `mujoco_sac.zip` | SAC policy, 300k steps (best) |
| `mujoco_balance_bot.zip` | PPO policy, 3M steps |
| `mujoco_balance_bot_1p5M_backup.zip` | earlier PPO policy, 1.5M steps, wider reward band |

## Environment
- Obs: `cart_pos, sin(angle), cos(angle), cart_vel, pole_vel` (angle 0 = upright, pi = hanging)
- Action: cart force in [-1, 1], scaled x10 (`ctrlrange="-10 10"`)
- 25 Hz control (4 physics steps of 0.01 s per action), 500-step episodes, cart limit +-3 m (hitting it ends the episode with -10)
- Reset: pole hanging (pi) plus small noise; **pole mass x U(0.7, 1.3) and hinge damping U(0.001, 0.05) are randomized every episode**
- Reward: `0.3*broad + narrow - 0.01*a^2 - 0.02*x^2`
  - `broad = (cos(angle)+1)/2`: weak slope everywhere, guides the swing
  - `narrow = exp(-(err/0.1)^2) * exp(-(pole_vel/4)^2)`: spike only when near upright AND slow, so flying through the top scores little

## Usage
```bash
python3 -m pip install -r requirements.txt
python3 train_sac.py 300000     # tensorboard --logdir ./sac_tensorboard/
python3 eval.py mujoco_sac sac 300
python3 watch.py mujoco_sac sac
```

## Results
Evaluated deterministically on seeds 0-299 (`python3 eval.py <policy> <ppo|sac> 300`).
"Balanced at end" = last 50 steps of the 500-step episode all within 0.2 rad of upright.

| Policy | Mean steps upright (of 500) | Balanced at end |
|---|---|---|
| PPO, 3M steps | 51.9 | 3/300 (1%) |
| **SAC, 300k steps** | **123.0** | **99/300 (33%)** |

SAC by hinge damping (sampled per episode):

| Damping | Balanced at end |
|---|---|
| 0.00-0.01 | 49/49 (100%) |
| 0.01-0.02 | 44/54 (81%) |
| 0.02-0.03 | 6/74 (8%) |
| 0.03-0.04 | 0/63 (0%) |
| 0.04-0.05 | 0/60 (0%) |

Pole mass had no effect (correlation with success 0.07; with damping -0.81). Initial angle sign had no effect.

## Why SAC worked where PPO didn't
- **Entropy bonus:** SAC maximizes `reward + alpha * entropy`, with alpha auto-tuned toward a target entropy. It keeps exploring
  among actions of similar value instead of settling into a habit like "jiggle near the bottom". SB3's PPO default is `ent_coef=0`,
  so its randomness only decays.
- **Replay buffer:** rare lucky swing-ups are replayed thousands of times. PPO uses each sample once.
- **Two critics + target networks:** `min(Q1, Q2)` limits overestimation; slowly-updated targets keep learning stable.
- Cost: one gradient update per env step, so about 100-170 steps/s on CPU against PPO's ~2,400, but far fewer steps needed.

## Where it still fails
All remaining failures are high-damping episodes (>= 0.02): the policy never gets the pole up and just sits near the bottom.
Not verified whether those episodes are feasible within the 10 N force and +-3 m track, or whether the policy is just
under-trained on them (they are about 60% of training episodes, and the policy cannot observe damping).

## Debugging history
1. Pole cylinder was centred on its hinge: it started 15 cm inside the floor (huge contact forces) and its centre of mass sat on
   the hinge, so gravity gave no tipping torque. Fixed with a capsule from the hinge upward, collisions off.
2. Switching to swing-up left the old `abs(angle) > 0.4` termination, ending every episode at step 1. Removed.
3. `ctrlrange` clipped force to 1 N while the code scaled by 10. Fixed.
4. Negative reward made driving off the track cheaper than staying alive. Reward shifted to non-negative.
5. PPO jiggled near the bottom (flat reward there): added the narrow upright bonus, gated on low pole velocity.
6. PPO still only flew through the top (3/300 balanced). Switched to SAC: 99/300.
7. Remaining SAC failures traced to hinge damping, which is randomized per episode.

## Next
- Train SAC longer (1M+ steps) and see whether the 0.02-0.03 damping band improves.
- Narrow the damping range to 0.001-0.02 (where SAC already gets 81-100%), or use a curriculum that widens it.
- Check whether high-damping swing-up is feasible at all (more force, longer track).
- Randomizing `body_mass` does not update `body_inertia`; scale both.
