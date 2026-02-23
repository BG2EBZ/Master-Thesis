Human behaviors:

- Following (walking towards to the robot)
- Wandering (walking randomly)
- Listening (stopping at target position and attending
to robot's explanation of display)
- Distracted (falling behind the group while
following)
- Overwhelmed (stepping back or leaving the area
when experiencing sensory/information overload in
listening stage)
- Impatient (moving faster than the robot)
- Attack (hitting the robot)
- And more potential behaviors to be added before
learning phase



Robot behaviors:

- Move (proceed to next display/destination)
- Stop (explain current display)
- Slow down (reduce speed)
- Speed up (increase speed)
- Call back (robot stops and turns toward lagging
person)
- Move back (Step back and keep a greater social
distance to humans)
- Wait (stop and wait for people)
- And more potential behaviors to be added before
learning phase


Info schema (current):

- info["events"]
  - entered_listen
  - started_listen_wait
  - completed_listen_wait
  - final_listen_ready
  - overwhelmed_triggered
  - attack_triggered
  - attack_hit
  - callback_triggered
  - callback_completed
  - move_back_triggered
  - move_back_completed

- info["status"]
  - step_count
  - listen_mode
  - listen_wait: active/counter/steps/remaining/is_final
  - callback_active/callback_target_idx/callback_hold_remaining
  - move_back_active/move_back_attacker_idx/move_back_safe_distance/move_back_speed
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
  - impatient_timer
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
