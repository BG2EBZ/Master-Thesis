# Human behaviors:

### Core Behaviors:
- Following (walking in formation around the robot
while moving between displays)

- Wandering (walking randomly before joining the
group)

- Listening (stopping, and facing the robot during
explanation)

- Distracted (falling behind and temporarily ignoring
the robot while following)

- Overwhelmed (backing off, then leaving the area
during listening-stage overload)

- Impatient (moving faster and pushing ahead of
normal following pace)

- Attack (approaching and attempting to hit the
robot)

### Callback Response Behaviors:

- Rejoin (return to following when robot calls back)

- Stay (pause in place for a fixed time after callback)- Happy (short emotion expression after successful
callback rejoin)

- Ignore (continue moving away after callback)- Fear (when attacker is too close)

### Fear Response Behaviors:

- Move back (attacker returns to original position)

- Stay (attacker freezes in place while robot is in
fear state)

- Continue hit (attacker keeps attacking even when
robot shows fear)



# Robot behaviors:

### Navigation Behaviors:

- Move (proceed to the next waypoint/display)
- Stop (stop at display and rotate to face the crowd)

### Interaction Behaviors:

- Explain (enter explanation phase after turning to
people)

- Wait (hold explanation window while humans
settle and events unfold)

- Call back (stop and turn toward a distracted
person, and say "Please follow me")

- Move back (step away from attack threat to
maintain safety distance)

### Emotion States:

- Natural (default state)

- Sad (when humans are distracted or
overwhelmed)

- Happy (short emotion expression after successful
callback rejoin)
- Fear (when attacker is too close)


# Neurodivergent people behaviors profile parameters

All range from 0 to 1

- Sensory Sensitivity: Impact `Overwhelemed` trigger probability, `Overwhelemed` duration

- Social Space Preference: Impact `safety social distance` between humans and robot

- Attention Span: Impact `Distracted` trigger probability, `Distracted` duration, `Callback Response` probability

- Interaction Style: Impact `Attack` probability, `Fear Response` probability




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
  - callback_forced_recovery
  - happy_triggered
  - happy_completed
  - fear_triggered
  - fear_completed
  - move_back_triggered
  - move_back_completed

- info["status"]
  - step_count
  - listen_mode
  - listen_wait: active/counter/steps/remaining/is_final
  - callback_active/callback_target_idx/callback_hold_remaining
  - move_back_active/move_back_attacker_idx/move_back_safe_distance/move_back_speed
  - robot_emotion
  - happy_remaining_steps
  - happy_hold_seconds
  - fear_active
  - fear_attacker_idx
  - fear_distance_threshold
  - speaker_active
  - robot_text_label
  - terminated_reason

- info["robot"]
  - pose_xy, goal_xy, dist_to_goal
  - yaw, desired_yaw, mode
  - emotion
  - action: vx/vy/yaw_rate
  - final_waypoint_reached
  - note: goal_xy is the robot's current waypoint, not an XML goal site

Robot emotion rule:

- fear: any ATTACK human with distance < 0.8m
- sad: any human is DISTRACTED or OVERWHELMED
- happy: triggered by callback_forced_recovery for 1 second
- natural: otherwise
- priority: fear > sad > happy > natural
- note: fear overrides happy/sad; happy timer pauses during fear
- speaker_active: True when listen_wait_active is True (yellow speaking halo visible)
- robot text label priority:
  - fear_active -> `I_need_more_space` (site group 5)
  - any DISTRACTED human (and no fear) -> `Please_follow_me` (site group 4)
  - speaker_active (and no fear, no DISTRACTED) -> `explanation` (site group 3)
  - otherwise -> none

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
