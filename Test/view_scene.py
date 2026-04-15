import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("museum_shell.xml")
data = mujoco.MjData(model)


def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

def qadr(joint_name):
    return model.jnt_qposadr[jid(joint_name)]

# Robot actuators
r_x   = aid("motor_x")
r_y   = aid("motor_y")
r_yaw = aid("motor_yaw")

person_ids = [1, 2, 3, 4, 5]

person_act = {}
person_qpos = {}

for pid in person_ids:
    person_act[pid] = {
        "x":   aid(f"person{pid}_motor_x"),
        "y":   aid(f"person{pid}_motor_y"),
        "yaw": aid(f"person{pid}_motor_yaw"),
    }
    person_qpos[pid] = {
        "x":   qadr(f"person{pid}_x"),
        "y":   qadr(f"person{pid}_y"),
        "yaw": qadr(f"person{pid}_yaw"),
    }

rng = np.random.default_rng(42)

ROOM_MIN = 0.7     # keep away from walls
ROOM_MAX = 9.3     # room A is [0,10]x[0,10]

WALK_SPEED = 0.015
TURN_GAIN = 1.5
TARGET_THRESH = 0.3

# one target per person
targets = {
    pid: rng.uniform([ROOM_MIN, ROOM_MIN], [ROOM_MAX, ROOM_MAX])
    for pid in person_ids
}


with mujoco.viewer.launch_passive(model, data) as viewer:

    # cam_id = mujoco.mj_name2id(
    #     model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overview"
    # )
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # viewer.cam.fixedcamid = cam_id

    # ---- Fix camera ----
    viewer.cam.lookat[:] = [6.0, -5.0, 1.5]   # center of your layout
    viewer.cam.distance = 30.0                # zoom out
    viewer.cam.elevation = -30.0              # look downward
    viewer.cam.azimuth = 90.0                 # rotate around z-axis    

    while viewer.is_running():
        # Robot: forward + slight turn
        data.ctrl[r_x] = 0.0
        data.ctrl[r_y] = 0.01
        data.ctrl[r_yaw] = 0.1

        # People random walking
        for pid in person_ids:
            # current state
            x   = data.qpos[person_qpos[pid]["x"]]
            y   = data.qpos[person_qpos[pid]["y"]]
            yaw = data.qpos[person_qpos[pid]["yaw"]]

            tx, ty = targets[pid]

            dx = tx - x
            dy = ty - y
            dist = np.hypot(dx, dy)

            # reached target → sample new one
            if dist < TARGET_THRESH:
                targets[pid] = rng.uniform(
                    [ROOM_MIN, ROOM_MIN],
                    [ROOM_MAX, ROOM_MAX]
                )
                data.ctrl[person_act[pid]["x"]] = 0.0
                data.ctrl[person_act[pid]["y"]] = 0.0
                data.ctrl[person_act[pid]["yaw"]] = 0.0
                continue

            # desired heading
            desired_yaw = np.arctan2(dx, dy)  # because +y is forward
            yaw_err = (desired_yaw - yaw + np.pi) % (2*np.pi) - np.pi

            # WORLD-frame velocity toward target
            vx = WALK_SPEED * dx / (dist + 1e-6)
            vy = WALK_SPEED * dy / (dist + 1e-6)

            data.ctrl[person_act[pid]["x"]] = vx
            data.ctrl[person_act[pid]["y"]] = vy
            data.ctrl[person_act[pid]["yaw"]] = np.clip(
                TURN_GAIN * yaw_err, -1.5, 1.5
            )

        mujoco.mj_step(model, data)
        viewer.sync()