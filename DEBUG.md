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


Potential robot behaviors:

- Change to Silent Mode if someone is sensory overload?
- Yield or Give Way when someone is impatient?
- Quiet Zone Guidance for overwhelmed person?


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
  - callback_response_rejoin
  - callback_response_stay
  - callback_response_ignore
  - happy_triggered
  - happy_completed
  - fear_triggered
  - fear_completed
  - fear_response_move_back
  - fear_response_stay
  - fear_response_continue_hit
  - move_back_triggered
  - move_back_completed

- info["status"]
  - step_count
  - listen_mode
  - listen_wait: active/counter/steps/remaining/is_final
  - callback_active/callback_target_idx/callback_hold_remaining
  - callback_last_response/callback_last_response_target_idx
  - move_back_active/move_back_attacker_idx/move_back_safe_distance/move_back_speed
  - robot_emotion
  - happy_remaining_steps
  - happy_hold_seconds
  - fear_active
  - fear_attacker_idx
  - fear_last_response/fear_last_response_target_idx
  - fear_distance_threshold
  - speaker_active
  - robot_text_label
  - external_action_received/external_action_used
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

FSM modules (decoupled transition tables):

- `museum_env/robot_fsm.py`
  - `Transition` + `apply_transitions(...)`
  - robot transition table for `move/callback/stop/wait/move_back`
  - used by env waiting branch for MOVE_BACK switching

- `museum_env/human_fsm.py`
  - `Transition` + `apply_transitions(...)`
  - human transition table for env-level mode switching
  - callback/fear response transitions provide effects consumed by env

Env structure modules:

- `museum_env/env_stepflows.py`
  - waiting/active step flow orchestration
- `museum_env/env_transitions.py`
  - callback/fear/move_back transition helpers and FSM effects application
- `museum_env/env_visuals.py`
  - robot emotion color, text label visibility, speaking halo sync
- `museum_env/env_info.py`
  - event defaults, snapshot collection, info schema build
- `museum_env/env_runtime.py`
  - runtime state dataclass used by reset initialization

Current execution closure:

- robot wait branch: FSM decision + FSM effects (`set_move_back/hold_move_back/clear_move_back`)
- human callback branch: FSM decision + FSM effects (`set_callback_submode/callback_stay_steps`)
- human fear branch: FSM decision + FSM effects (`restore_listen_anchor/freeze_attack`)
- no shadow/legacy comparison path in env runtime
