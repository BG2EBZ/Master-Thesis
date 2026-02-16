# Master Thesis Museum Env

## Info schema update

`MuseumEnv.step()` now returns a nested `info` structure with four top-level keys:

- `info["events"]`
- `info["status"]`
- `info["robot"]`
- `info["humans"]`

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

if info["events"]["final_listen_ready"]:
    print("Final listen completed")

print(info["robot"]["pose_xy"])
print(info["humans"]["action"]["vx"])
print(info["status"]["listen_wait"]["remaining"])
```

## Run scripts

- Interactive run: `python3 test_env.py`
- Video recording: `python3 record_env.py`
