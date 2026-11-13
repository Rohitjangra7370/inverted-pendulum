import mujoco
import mujoco.viewer
import time


xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1"/>
    
    <!-- Box starts resting on the floor (Z=0.1) -->
    <body pos="0 0 0.1">
      <!-- 1-DOF sliding joint along the X-axis -->
      <joint name="slider_x" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
  
  <actuator>
    <!-- Motor that applies force to the slider joint -->
    <motor name="motor_x" joint="slider_x" gear="1"/>
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Applying force. Watch the terminal for state data.")
    
    while viewer.is_running():
        # 1. APPLY ACTION: Send a constant force of 0.5 Newtons to the first actuator
        data.ctrl[0] = 0.5 
        
        # 2. STEP PHYSICS: The solver calculates the new dynamics
        mujoco.mj_step(model, data)
        
        # 3. READ STATE: Extract position (qpos) and velocity (qvel)
        position = data.qpos[0]
        velocity = data.qvel[0]
        
        print(f"Position: {position:.3f} m | Velocity: {velocity:.3f} m/s")
        
        viewer.sync()
        time.sleep(0.01)