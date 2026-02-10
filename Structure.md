museum_scene.xml:
- Room geometry (walls, corridors)
- Robot body and geometry
- Human bodies and geometry
- Collision and physical properties

env.py:
- Inherits from `gym.Env`
- Loads MuJoCo `model` and `data`
- Calls `human.update(...)` every timestep
- Handles `reset()`, `step()`, `render()`

human.py
- Individual human behavior
- Internal state:
    - position
    - velocity
    - waypoint
    - behavior mode(wandering, following, listening)
- Implements:
    - random wandering
    - following the robot
    - basic social spacing and repulsion between robot and humans

test_env.py
- Instantiate the environment
- Render the scene
- Run `step`
