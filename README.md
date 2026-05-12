# Master Thesis Museum Env

## Step info schema

`MuseumEnv.step()` now returns a compact `info` structure with five top-level keys:

- `info["events"]`
- `info["episode"]`
- `info["phase"]`
- `info["robot"]`
- `info["crowd"]`

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

`info["episode"]` contains episode-level metadata:

- `step`
- `terminated_reason`

`info["phase"]` contains the current orchestration phase:

- `follow`
- `listen`

`info["robot"]` contains pose and control summary:

- `pose_xy`
- `goal_xy`
- `dist_to_goal`
- `yaw`
- `mode`
- `callback_phase`
- `emotion`
- `speaker_active`
- `action`

`info["crowd"]` contains the compact crowd snapshot:

- `pose_xy`
- `goal_xy`
- `modes`
- `profiles`
- `human_robot_distance`
- `reached_goal_indices`
- `distracted_indices`

The old debug-heavy `metrics`, per-human fuzzy dumps, timers, and callback internals are no longer returned by default.

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["phase"]["listen"])
print(info["robot"]["dist_to_goal"])
print(info["crowd"]["modes"])

if info["events"]["final_listen_ready"]:
    print("Episode completed")
```

## Run scripts

- Demo: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode demo`
- Fast train loop: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train`
- Record video: `/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode record --use-timestamp-subfolder`
