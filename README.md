# Master Thesis Museum Env

## Info schema update

`MuseumEnv.step()` now returns a nested `info` structure with five top-level keys:

- `info["events"]`
- `info["status"]`
- `info["metrics"]`
- `info["robot"]`
- `info["humans"]`

`info["robot"]["goal_xy"]` is sourced from `Robot`'s current waypoint (`Robot.get_current_waypoint()`), not from an XML `goal_site`.

`info["humans"]` includes impatient debugging fields:
- `info["humans"]["impatient_timer"]`
- `info["humans"]["impatient_cooldown_timer"]`

`info["metrics"]["humans"]` includes human-human spacing metrics:
- `info["metrics"]["humans"]["nearest_human_distance"]`
- `info["metrics"]["humans"]["nearest_human_distance_mean_1s"]`
- `info["metrics"]["humans"]["window_seconds"]`
- `info["metrics"]["humans"]["window_steps"]`

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

if info["events"]["final_listen_ready"]:
    print("Final listen completed")

print(info["robot"]["pose_xy"])
print(info["humans"]["action"]["vx"])
print(info["metrics"]["humans"]["nearest_human_distance_mean_1s"])
print(info["status"]["listen_wait"]["remaining"])
```

## Run scripts

- Interactive near real-time demo (default): `python3 test_env.py --mode demo`
- Periodic printing human human distance (nearest): `python3 test_env.py --mode train --rtf-print-every 500 --print-hh-distance-mean-1s`
- Strict real-time alignment demo: `python3 test_env.py --mode demo --realtime-policy strict`
- Fast training run (no render, no sleep): `python3 test_env.py --mode train`
- Video recording with simulation-time playback: `python3 test_env.py --mode record --video-fps 500`
- Legacy recording entry (thin wrapper): `python3 record_env.py`

### Recommended speed-related flags

- `--render-fps 60`: target visual refresh for demo pacing.
- `--sleep-scale 1.0`: `>1.0` speeds up perceived playback, `<1.0` slows it down.
- `--video-fps 500`: for MuJoCo `dt=0.002`, this matches simulation-time playback.
- `--rtf-print-every 500`: print real-time factor periodically.

### Time semantics

- **Simulation time**: `steps * dt` (from MuJoCo, currently `dt=0.002`).
- **Wall-clock time**: actual elapsed real time while script runs.
- **Playback time** (recorded video): controlled by encoded FPS (`--video-fps`).

These three are intentionally decoupled:

- `demo`: tries to keep simulation pace close to real time (stable or strict policy).
- `train`: runs as fast as possible; simulation time and wall-clock time diverge.
- `record`: runs without sleep; playback speed is determined by `video_fps`.
