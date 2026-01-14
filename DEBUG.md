How shall I set a proper dumpling?

```
    <body name="robot" pos="0 0 0.06">
      <!-- Planar movement: x, y, yaw -->
      <joint name="robot_x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="2"/>
      <joint name="robot_yaw" type="hinge" axis="0 0 1" damping="1"/>
```
