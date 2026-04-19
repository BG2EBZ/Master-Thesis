# Master Thesis Museum Env

## Step info schema

`MuseumEnv.step()` now returns a compact `info` structure with four top-level keys:

- `info["events"]`
- `info["state"]`
- `info["robot"]`
- `info["humans"]`

`info["events"]` contains the high-level episode markers:

- `entered_listen`
- `started_listen_wait`
- `completed_listen_wait`
- `final_listen_ready`
- `callback_triggered`
- `callback_completed`
- `callback_success`
- `happy_triggered`
- `happy_completed`

`info["state"]` contains the current orchestration state:

- `step_count`
- `follow_phase`
- `listen_phase`
- `robot_mode`
- `callback_phase`
- `robot_emotion`
- `speaker_active`
- `terminated_reason`

`info["robot"]` contains pose and control summary:

- `pose_xy`
- `goal_xy`
- `dist_to_goal`
- `yaw`
- `action`

`info["humans"]` contains the compact crowd snapshot:

- `pose_xy`
- `goal_xy`
- `mode`
- `profile`
- `reached_goal_indices`
- `perceived_distracted_indices`

The old debug-heavy `metrics`, per-human fuzzy dumps, timers, and callback internals are no longer returned by default.

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["state"]["listen_phase"])
print(info["robot"]["dist_to_goal"])
print(info["humans"]["mode"])

if info["events"]["final_listen_ready"]:
    print("Episode completed")
```

## Run scripts

- Demo: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode demo`
- Fast train loop: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train`
- Record video: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode record --use-timestamp-subfolder`
