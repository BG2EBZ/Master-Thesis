import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("museum_shell.xml")
data = mujoco.MjData(model)


def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

# Robot actuators
r_x   = aid("motor_x")
r_y   = aid("motor_y")
r_yaw = aid("motor_yaw")

p_x = aid("person1_motor_x")
p_y = aid("person1_motor_y")
p_yaw = aid("person1_motor_yaw")


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Robot: forward + slight turn
        data.ctrl[r_x] = 0.0
        data.ctrl[r_y] = 0.01
        data.ctrl[r_yaw] = 0.1

        # Person: forward + slight turn
        data.ctrl[p_x] = 0.0
        data.ctrl[p_y] = 0.01
        data.ctrl[p_yaw] = 0.1

        mujoco.mj_step(model, data)
        viewer.sync()