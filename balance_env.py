import gymnasium as gym
import numpy as np
import mujoco
import time
import mujoco.viewer


class BalanceBotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # The same XML from yesterday's Hello World, but now with a hinge joint (a pendulum)
        xml = """
        <mujoco>
          <option timestep="0.01" gravity="0 0 -9.81"/>
          <worldbody>
            <light pos="0 0 1"/>
            <geom type="plane" size="1 1 0.1"/>
            
            <body pos="0 0 1">
              <joint type="slide" axis="1 0 0" name="slider"/>
              <geom type="box" size="0.1 0.1 0.05" rgba="0 0 1 1" mass="1.0"/>
              
              <body name="pole" pos="0 0 0.05">
                <joint type="hinge" axis="0 1 0" name="hinge"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="1 0 0 1" mass="0.1" contype="0" conaffinity="0"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slider" name="slide_motor" ctrlrange="-10 10" ctrllimited="true"/>
          </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        
        # Action Space: 1 continuous value (motor torque applied to the slider)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: 5 values (cart pos, sin(angle), cos(angle), cart vel, pole vel)
        # sin/cos so the angle has no jump at +-pi during swing-up
        high = np.full(5, np.inf, dtype=np.float32)
        self.max_steps = 500
        self.steps = 0
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self.model.body_mass[pole_id]=0.1*self.np_random.uniform(0.7,1.3)

        hinge_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self.model.dof_damping[hinge_id]=self.np_random.uniform(0.001,0.05)

        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0

        self.data.qpos[1] = np.pi
        self.data.qpos[:] += self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        self.data.qvel[:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    def step(self, action):
        # Apply the AI's chosen torque
        max_force = 10.0
        self.data.ctrl[0] = action[0] * max_force
        
        # --- THE CONTROL FREQUENCY FIX (Action Repeat) ---
        # The AI holds this action for 4 physics frames (25 Hz control loop)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        # -------------------------------------------------
        
        self.steps += 1
        obs = self._get_obs()
        cart_pos = self.data.qpos[0]
        pole_angle = self.data.qpos[1]
        
        terminated = False
        # swing-up: cos(angle) is -1 hanging down, +1 upright. No angle termination.
        # shifted to [0, 1]: 0 hanging down, 1 upright. Never negative, so staying alive beats dying.
        # reward 1: broad, weak. Slope everywhere, 0 hanging -> 1 upright (guides the swing).
        broad = (np.cos(pole_angle) + 1.0) / 2.0
        # reward 2: narrow, strong. Gaussian spike within ~0.2 rad of upright (rewards actually balancing).
        err = np.arctan2(np.sin(pole_angle), np.cos(pole_angle))  # wrapped angle, 0 = upright
        # only counts when the pole is also slow, so flying through the top earns little but balancing earns full
        pole_vel = self.data.qvel[1]
        narrow = np.exp(-(err / 0.1) ** 2) * np.exp(-(pole_vel / 4.0) ** 2)
        reward = float(0.3 * broad + 1.0 * narrow - 0.01 * (action[0] ** 2) - 0.02 * cart_pos ** 2)
        
        if abs(cart_pos) > 3.0:
            reward = -10.0
            terminated = True
        
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],
            np.sin(self.data.qpos[1]),
            np.cos(self.data.qpos[1]),
            self.data.qvel[0], 
            self.data.qvel[1]
        ], dtype=np.float32)


# --- THE TEST HARNESS ---
if __name__ == '__main__':
    print("Testing the Gymnasium Wrapper...")
    env = BalanceBotEnv()
    obs, _ = env.reset()
    print(f"Initial State: {obs}")
    
    total_reward = 0
    for i in range(50):
        # Sample a random action (simulating an untrained, flailing AI)
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        
        print(f"Step {i+1}: Action: {random_action[0]:.2f} | Angle: {obs[1]:.3f} | Reward: {reward:.2f}")
        
        if terminated:
            print(f"Bot fell over at step {i+1}. Total Reward: {total_reward:.2f}")
            break

