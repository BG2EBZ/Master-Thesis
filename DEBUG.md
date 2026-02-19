Human behaviors:

  following (walking towards the robot)

  wandering (walking randomly)

  listening (go to target position then stand in front of the robot listening robot explanation a display)

  distracted (when humans are following the robot, someone may fall behind)

  overwhelmed (step back or leaving when listen)

  impatient (only person3: move faster than robot and go to the robot's front side while following)

  attact (hit the robot)

  Add more behaviors before learning



Robot behaviors:

  Move (To next display)

  Stop (Explain this display)

  Slow (Reduce speed)

  Speed up (increase speed)

  Call back (Robot stops and turns to the lagging person)

  Move back

  Wait (stop waiting people)

  Add more behaviors before learning


Info schema (current):

- info["events"]
  - entered_listen
  - started_listen_wait
  - completed_listen_wait
  - final_listen_ready
  - overwhelmed_triggered

- info["status"]
  - step_count
  - listen_mode
  - listen_wait: active/counter/steps/remaining/is_final
  - terminated_reason

- info["robot"]
  - pose_xy, goal_xy, dist_to_goal
  - yaw, desired_yaw, mode
  - action: vx/vy/yaw_rate
  - final_waypoint_reached
  - note: goal_xy is the robot's current waypoint, not an XML goal site

- info["humans"]
  - pose_xy, goal_xy
  - actual_yaw, desired_yaw
  - mode, distracted_timer
  - overwhelmed_stage, overwhelmed_leave_timer
  - impatient_timer, impatient_cooldown_timer
  - reached_goal_indices, all_reached
  - action: vx/vy/yaw_rate
  - velocity_components: follow/repulsion/human_robot/total

Quick check snippet:

```python
obs, reward, terminated, truncated, info = env.step(None)
if info["events"]["final_listen_ready"]:
    print("done")
print(info["status"]["listen_wait"]["remaining"])
print(info["robot"]["action"]["vx"])
print(info["humans"]["all_reached"])
```
