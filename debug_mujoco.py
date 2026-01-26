import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("museum_scene.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

print("Actuators:", model.nu)

data.ctrl[:] = 0.0
data.ctrl[0] = 1.0
for i in range(3000):
    mujoco.mj_step(model, data)
    if i % 500 == 0:
        print(f"Step {i}: robot_x = {data.qpos[0]:.3f}")
    viewer.sync()
    time.sleep(0.01)
