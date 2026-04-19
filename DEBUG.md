# Debug Notes

## Compact runtime contract

The environment now exposes a compact runtime surface:

- `info["events"]`: high-level episode and callback events
- `info["state"]`: orchestrator phase state
- `info["robot"]`: pose, goal, distance, action
- `info["humans"]`: crowd pose, goal, mode, profile, and perceived distracted indices

Key fields to watch while debugging:

- `info["state"]["follow_phase"]`: `None`, `pre_listen_engage`, or `transit_follow`
- `info["state"]["listen_phase"]`: `idle`, `intro`, `wait`, or `paused`
- `info["state"]["robot_mode"]`: `move`, `stop`, or `callback`
- `info["state"]["callback_phase"]`: `turn`, `cue`, or `None`
- `info["events"]["final_listen_ready"]`: terminal success flag

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["state"])
print(info["robot"])
print(info["humans"]["perceived_distracted_indices"])
```

## Manual smoke

Use the simplified runner:

```bash
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode demo
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train --print-every 1000
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode record --use-timestamp-subfolder
```
