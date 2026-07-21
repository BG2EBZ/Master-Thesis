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
- `question_started`
- `question_completed`
- `final_listen_ready`
- `callback_triggered`
- `callback_completed`
- `callback_success`
- `callback_ignored`
- `happy_triggered`
- `happy_completed`

These event fields are boolean edge markers for the current step.

`info["episode"]` contains episode-level metadata:

- `step`
- `terminated_reason`
- terminal-only on the final step:
- `duration_seconds`
- `overwhelmed_triggers`
- `impatient_triggers`
- `distracted_triggers`
- `return`
- `reward_components`

`duration_seconds` is simulated episode duration, computed from `step_count * dt`, not wall-clock runtime.

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

## Reward semantics

`MuseumEnv.step()` now uses episodic reward semantics:

- intermediate steps return `0.0`
- the final scalar reward is emitted only when the episode completes or times out
- `info` keeps the same compact top-level schema shown above
- final-step reward breakdown is exposed under `info["episode"]["reward_components"]`
- each env step still represents `0.05s` of simulated decision time, even though MuJoCo may integrate smaller internal physics substeps

Timing note:

- public decision timestep: `dt = 0.05s`
- MuJoCo integrates finer internal physics substeps underneath that decision timestep
- final `info["episode"]["duration_seconds"]` is based on simulated time, not real execution time

Example:

```python
obs, reward, terminated, truncated, info = env.step(None)

print(info["phase"]["listen"])
print(reward)
print(info["crowd"]["modes"])

if terminated or truncated:
    print("Final episode reward:", reward)
```

## Run scripts

- Demo: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode demo [--seed 2]`
- Fast train loop: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode train [--seed 2]`
- Record video: `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode record --use-timestamp-subfolder [--seed 2]`
- Minimal RWR training: `/home/tianci/Polimi/workspace/venv/bin/python scripts/train_rwr.py`
- Evaluate a learned RWR policy: `/home/tianci/Polimi/workspace/venv/bin/python scripts/eval_baseline.py --learned-params-json artifacts/runs/rwr_20260721_155908/best_params.json`

If running the evaluator from inside `scripts/`, use the parent-relative path:

- `/home/tianci/Polimi/workspace/venv/bin/python eval_baseline.py --learned-params-json ../artifacts/runs/rwr_20260721_155908/best_params.json`

Reproducibility example:

- `/home/tianci/Polimi/workspace/venv/bin/python scripts/run_env.py --mode train --seed 2 --max-steps 5 --print-every 1`

`--seed` is optional and is passed only to the initial `env.reset(...)` for that invocation.

The minimal RWR trainer writes:

- default output directory: `artifacts/runs/rwr_minimal`
- `training_metrics.csv`
- `training_metrics.png`
