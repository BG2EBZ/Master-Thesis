# Debug Notes

## Compact runtime contract

The environment now exposes a compact runtime surface:

- `info["events"]`: high-level episode and callback events
- `info["episode"]`: step counter and termination reason
- `info["phase"]`: follow/listen orchestration state
- `info["robot"]`: pose, goal, mode, emotion, speaker, and action
- `info["crowd"]`: crowd pose, goal, modes, profiles, and distracted indices

Key fields to watch while debugging:

- `info["phase"]["follow"]`: `None`, `pre_listen_engage`, or `transit_follow`
- `info["phase"]["listen"]`: `idle`, `intro`, `wait`, or `paused`
- `info["robot"]["mode"]`: `move`, `stop`, or `callback`
- `info["robot"]["callback_phase"]`: `turn`, `cue`, or `None`
- `info["events"]["final_listen_ready"]`: terminal success flag

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["phase"])
print(info["robot"])
print(info["crowd"]["distracted_indices"])
```

## Manual smoke

Use the simplified runner:

```bash
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode demo
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode train --print-every 1000
/home/tianci/Polimi/workspace/venv/bin/python test_env.py --mode record --use-timestamp-subfolder
```
